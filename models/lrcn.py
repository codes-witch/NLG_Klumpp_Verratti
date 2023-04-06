import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.distributions import Categorical

from .pretrained_models import PretrainedModel

"""
Description of the LRCN model (including forward pass and caption generation). Much of this is also inherited by the GVE 
model and hence used in both cases.
"""


class LRCN(nn.Module):
    def __init__(self, input, word_embed_size, hidden_size,
                 vocab_size, layers_to_truncate=1, dropout_prob=0.5):
        super().__init__()

        # 'input' can either be an integer, in which case it describes the input size (if-condition),
        # or it can be the name of a pretrained model (else-condition). If it is neither,
        # PretrainedModel() will raise an error.
        # self.vision_model is the pretrained vision model (e.g. vgg11), if there is such
        if isinstance(input, int):
            img_feat_size = input
            input_size = input
            self.has_vision_model = False
        else:
            self.vision_model = PretrainedModel(input,
                                                layers_to_truncate=layers_to_truncate)
            img_feat_size = self.vision_model.output_size
            input_size = self.vision_model.input_size
            self.has_vision_model = True
        # construct a word embedding. The parameters are the vocabulary size to be represented, and
        # the desired embedding size
        self.word_embed = nn.Embedding(vocab_size, word_embed_size, padding_idx=0)

        # the first layer takes the embedding of the previously generated word as input, hence the input size is that of
        # a word embedding
        lstm1_input_size = word_embed_size
        # the second layer takes the output of the first layer as well as the image features (passed through
        # 'self.linear1' first), both of which are equal in size to the hidden size
        lstm2_input_size = 2 * hidden_size

        # define the layers with the correct input and output sizes
        self.linear1 = nn.Linear(img_feat_size, hidden_size)
        self.lstm1 = nn.LSTM(lstm1_input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(lstm2_input_size, hidden_size, batch_first=True)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        self.init_weights()  # initialize the weights
        # state dropout probability
        self.dropout_prob = dropout_prob

    def init_weights(self):
        # weights are initialized to values between -0.1 and 0.1, biases to 0
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        self.linear1.weight.data.uniform_(-0.1, 0.1)
        self.linear1.bias.data.fill_(0)
        self.linear2.weight.data.uniform_(-0.1, 0.1)
        self.linear2.bias.data.fill_(0)

    def forward(self, image_inputs, captions, lengths, feat_func=None):
        """
        Implementation of a forward pass.

        Parameters:

            -   image_inputs: the image features
            -   captions: the gold standard captions for training
        """

        # set the featured function to default (identity) if none is explicitly given
        if feat_func is None:
            feat_func = lambda x: x

        # get the embeddings representing the caption
        embeddings = self.word_embed(captions)

        # apply a first dropout in the beginning
        embeddings = F.dropout(embeddings, p=self.dropout_prob, training=self.training)

        # the input either goes through the vision model first (if there is such) or directly into the LRCN
        if self.has_vision_model:
            image_features = self.vision_model(image_inputs)
        else:
            image_features = image_inputs

        # the image features are first passed through the linear layer
        image_features = self.linear1(image_features)
        # apply rectified linear unit function (a function that "cuts off" everything where x<=0)
        image_features = F.relu(image_features)
        # apply dropout for image features too
        image_features = F.dropout(image_features, p=self.dropout_prob, training=self.training)
        # apply the featured function specified above
        image_features = feat_func(image_features)
        image_features = image_features.unsqueeze(1)
        image_features = image_features.expand(-1, embeddings.size(1), -1)

        # This packs the embeddings. Packing padded sequences saves useless computation that working with padding
        # (irrelevant tokens) entails. This is just to agilize the computation for the LSTM
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        # pass through the first lstm layer. Only the captions go through the first LSTM.
        hiddens, _ = self.lstm1(packed)
        # Unpack hidden state to concatenate image features
        unpacked_hiddens, new_lengths = pad_packed_sequence(hiddens, batch_first=True)

        # concatenate output of first layer with image features
        unpacked_hiddens = torch.cat((image_features, unpacked_hiddens), 2)

        # apply dropout after the first layer
        unpacked_hiddens = F.dropout(unpacked_hiddens, p=self.dropout_prob, training=self.training)

        # Pack again to agilize computation
        packed_hiddens = pack_padded_sequence(unpacked_hiddens, lengths,
                                              batch_first=True)

        # pass through the second lstm layer
        hiddens, _ = self.lstm2(packed_hiddens)

        # apply dropout after the second layer
        hiddens = F.dropout(hiddens[0], p=self.dropout_prob, training=self.training)

        # pass through the final linear layer
        outputs = self.linear2(hiddens)

        # return the logits associated with each word as outputs
        return outputs

    def state_dict(self, *args, full_dict=False, **kwargs):
        """
        Return the state dictionary of the model, removing keys that are stored in the vision model if full_dict is
        False.
        """
        state_dict = super().state_dict(*args, **kwargs)
        if self.has_vision_model and not full_dict:
            for key in self.vision_model.state_dict().keys():
                del state_dict['vision_model.{}'.format(key)]
        return state_dict

    def sample(self, logits):
        """
         Sample from the distribution logits
        """
        dist = Categorical(logits=logits)
        sample = dist.sample()
        # return the sample as well as its log probability
        return sample, dist.log_prob(sample)

    def generate_sentence(self, image_inputs, start_word, end_word, states=(None, None),
                          max_sampling_length=50, sample=False, feat_func=None):

        # set the featured function to default if none is explicitly given
        if feat_func is None:
            feat_func = lambda x: x

        sampled_ids = []  # represents the words of the sentence
        # the input either goes through the vision model first (if there is such) or directly into the LRCN
        if self.has_vision_model:
            image_features = self.vision_model(image_inputs)
        else:
            image_features = image_inputs
        # the image features are first passed through the linear layer
        image_features = self.linear1(image_features)
        # apply rectified linear unit function (a function that "cuts off" everything where x<=0)
        image_features = F.relu(image_features)
        # apply the featured function specified above
        image_features = feat_func(image_features)
        image_features = image_features.unsqueeze(1)

        # 'embedded_word' is an embedding of the last generated word so far, starting with the start word
        embedded_word = self.word_embed(start_word)
        embedded_word = embedded_word.expand(image_features.size(0), -1, -1)

        lstm1_states, lstm2_states = states

        end_word = end_word.squeeze().expand(image_features.size(0))
        # the end is reached when the output corresponds to the end word
        reached_end = torch.zeros_like(end_word.data).byte()

        if sample:
            log_probabilities = []
            lengths = torch.zeros_like(reached_end).long()

        i = 0  # index
        # generate words until the end word has been reached
        while not reached_end.all() and i < max_sampling_length:
            # the input is the previous word as an embedding
            lstm1_input = embedded_word

            # pass through the first lstm layer
            lstm1_output, lstm1_states = self.lstm1(lstm1_input, lstm1_states)

            # concatenate output of first layer with image features
            lstm1_output = torch.cat((image_features, lstm1_output), 2)

            # pass both through the second lstm layer
            lstm2_output, lstm2_states = self.lstm2(lstm1_output, lstm2_states)

            # pass through the final linear layer
            outputs = self.linear2(lstm2_output.squeeze(1))

            if sample:
                # if 'sample' is true, the predicted word is sampled from the outputs
                predicted, log_p = self.sample(outputs)
                # the log probability of the prediction is stored as well
                active_batches = (~reached_end)
                log_p *= active_batches.float().to(log_p.device)
                log_probabilities.append(log_p.unsqueeze(1))
                lengths += active_batches.long()
            else:
                # if 'sample' is false, the predicted word is just the most probable
                predicted = outputs.max(1)[1]
            # the end is reached if 1) it was reached beforehand, or 2) the new word is the end word
            reached_end = reached_end | predicted.eq(end_word).data
            # append predicted word to list of previous words
            sampled_ids.append(predicted.unsqueeze(1))
            # generate word embedding from the predicted word to be used as input in the next iteration
            embedded_word = self.word_embed(predicted)
            embedded_word = embedded_word.unsqueeze(1)

            i += 1  # increase index

        sampled_ids = torch.cat(sampled_ids, 1).squeeze()
        if sample:
            # if the words were sampled, log probabilities are returned too
            log_probabilities = torch.cat(log_probabilities, 1).squeeze()
            return sampled_ids, log_probabilities, lengths
        # otherwise, return just the predictions
        return sampled_ids
