import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from functools import partial

from .lrcn import LRCN

"""
GVE model. Most methods are inherited from lrcn.py. The main difference is that for GVE we have labels.
"""


class GVE(LRCN):
    def __init__(self, input, word_embed_size, hidden_size,
                 vocab_size, sentence_classifier, num_classes, layers_to_truncate=1, dropout_prob=0.5):
        # initialize the LRCN with the given parameters
        super().__init__(input, word_embed_size, hidden_size, vocab_size, layers_to_truncate, dropout_prob)

        self.sentence_classifier = sentence_classifier
        self.num_classes = num_classes

        # the input size of the second layer is larger than for LRCN since now there are labels too
        lstm2_input_size = 2 * hidden_size + num_classes
        self.lstm2 = nn.LSTM(lstm2_input_size, hidden_size, batch_first=True)

    def convert_onehot(self, labels):
        """
        Represent the labels as a one-hot vector
        """
        labels_onehot = torch.zeros(labels.size(0),
                                    self.num_classes)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        return labels_onehot

    def get_labels_append_func(self, labels, labels_onehot):
        """
        return a function that can be used as feat_func in lrcn.forward(), but that makes sure that labels are
         concatenated to the image features 'labels' are the labels, 'labels_onehot' is a one-hot representation of them
        """
        # if there is no one-hot representation yet, create one
        if labels_onehot is None:
            labels_onehot = self.convert_onehot(labels)

        # the function to return concatenates image features and label representations
        # both (features and labels) are used where normal LRCN just has image features
        def append_labels(labels_to_append, image_features):
            return torch.cat((image_features,
                              labels_to_append.to(image_features.device)), 1)

        # what is returned is a function corresponding to 'append_labels' with 'labels_onehot' as the first argument
        # (so it still needs 'image_features' as a second argument)
        return partial(append_labels, labels_onehot)

    def forward(self, image_inputs, captions, lengths, labels,
                labels_onehot=None):
        """
        Forward pass
        """
        # get the function as defined in 'get_labels_append_func()'
        feat_func = self.get_labels_append_func(labels, labels_onehot)

        # execute the forward step from 'lrcn.py', using the function
        return super().forward(image_inputs, captions, lengths, feat_func)

    def generate_sentence(self, image_inputs, start_word, end_word,
                          labels, labels_onehot=None, states=(None, None), max_sampling_length=50, sample=False):
        """
        Generate a sentence
        """
        # get the function as defined in 'get_labels_append_func()'
        feat_func = self.get_labels_append_func(labels, labels_onehot)
        # execute the forward step from 'lrcn.py', using the function
        return super().generate_sentence(image_inputs, start_word, end_word, states, max_sampling_length, sample,
                                         feat_func)
