import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

class LRCNTrainer:
    """
    This class defines the training and the evaluation of an epoch for the LRCN model.

    Most methods in this trainer are used for both the lcrn and the gve settings,
    with the gve trainer inheriting from here
    """

    # REQ_EVAL is true when we want to get evaluation scores (Bleu score etc.)
    REQ_EVAL = True

    def __init__(self, args, model, dataset, data_loader, logger, device, checkpoint=None):
        self.model = model
        self.dataset = dataset
        self.data_loader = data_loader
        self.train = args.train
        self.logger = logger
        self.device = device

        model.to(self.device)

        # if no checkpoint has been passed, set hyperparameters and more settings. The current epoch is 0
        if checkpoint is None:
            # define loss function
            self.criterion = nn.CrossEntropyLoss()
            # parameters, all of them require gradients
            self.params = filter(lambda p: p.requires_grad, model.parameters())
            # use Adam as optimizer
            self.optimizer = torch.optim.Adam(self.params, lr=args.learning_rate)
            # number of steps per epoch
            self.total_steps = len(data_loader)
            self.num_epochs = args.num_epochs
            # how often we log
            self.log_step = args.log_step
            self.curr_epoch = 0

        # set the vocabulary and the start and end tokens
        vocab = self.dataset.vocab
        start_word = torch.tensor([vocab(vocab.start_token)],
                device=self.device, dtype=torch.long)
        self.start_word = start_word.unsqueeze(0)
        end_word = torch.tensor([vocab(vocab.end_token)], device=device,
                dtype=torch.long)
        self.end_word = end_word.unsqueeze(0)


    def train_epoch(self):
        """
        The purpose of this function is twofold:
        1. If this is called during training, it trains an epoch and returns the losses.
        2. If it is called in evaluation, it generates captions for evaluation and returns them.
        """
        # Result is list of losses during training
        # and generated captions during evaluation
        result = []

        # go through the data
        for i, (image_input, word_inputs, word_targets, lengths, ids, *excess) in enumerate(self.data_loader):
            # Original Comment: Prepare mini-batch dataset
            image_input = image_input.to(self.device)

            # if called for training, train one epoch and return loss
            if self.train:
                word_inputs = word_inputs.to(self.device)
                word_targets = word_targets.to(self.device)
                word_targets = pack_padded_sequence(word_targets, lengths, batch_first=True)[0]

                # train, compute the loss for the current batch and append to the result; the method used is
                # train_step(), see below
                loss = self.train_step(image_input, word_inputs, word_targets,
                                       lengths, *excess)
                result.append(loss.data.item())

                # compute the step number and log step number and corresponding loss
                step = self.curr_epoch * self.total_steps + i + 1
                self.logger.scalar_summary('batch_loss', loss.data.item(), step)

            # if in evaluation, captions are generated to be returned later, using eval_step() (see below)
            else:
                generated_captions = self.eval_step(image_input, ids, *excess)
                result.extend(generated_captions)

            # print log information after each log_step steps
            if i % self.log_step == 0:
                print("Epoch [{}/{}], Step [{}/{}]".format(self.curr_epoch,
                    self.num_epochs, i, self.total_steps), end='')
                if self.train:
                    print(", Loss: {:.4f}, Perplexity: {:5.4f}".format(loss.data.item(),
                        np.exp(loss.data.item())), end='')
                print()

        # the trainer keeps track of the number of times (epochs) it is called. This is used in the main.py
        # code to determine when it will stop training
        self.curr_epoch += 1

        # after the end of the whole epoch, the total loss is logged again
        if self.train:
            self.logger.scalar_summary('epoch_loss', np.mean(result), self.curr_epoch)

        return result

    def train_step(self, image_input, word_inputs, word_targets, lengths,
            *args):
        # zero out the gradients
        self.model.zero_grad()
        # forward pass
        outputs = self.model(image_input, word_inputs, lengths)
        # compute loss
        loss = self.criterion(outputs, word_targets)
        # back-propagate and optimize
        loss.backward()
        self.optimizer.step()

        return loss

    # this is the evaluation step used in train_epoch()
    def eval_step(self, image_input, ids, *args):

        vocab = self.dataset.vocab
        generated_captions = []
        outputs = self.model.generate_sentence(image_input, self.start_word, self.end_word, *args)
        # the output is a list of word indices. The following nested for-loops are used to convert
        # this into a caption by retrieving the word associated to that index from the vocabulary until an
        # end-of-caption token is reached.
        for out_idx in range(len(outputs)):
            sentence = []
            for w in outputs[out_idx]:
                word = vocab.get_word_from_idx(w.data.item())
                if word != vocab.end_token:
                    sentence.append(word)
                else:
                    break
            generated_captions.append({"image_id": ids[out_idx], "caption": ' '.join(sentence)})
        # generated_captions is a list of dictionaries mapping to the image ID and to its caption
        return generated_captions
