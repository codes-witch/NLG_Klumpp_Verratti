import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

class LRCNTrainer:

    # NOTE: Most methods in this trainer are used for both the lrcn and the gve settings,
    # with the gve trainer inheriting from it

    # NOTE: REQ_EVAL is true when we want to get evaluation scores (BLEU, CIDEr, etc)
    REQ_EVAL = True

    def __init__(self, args, model, dataset, data_loader, logger, device, checkpoint=None):
        self.model = model
        self.dataset = dataset
        self.data_loader = data_loader
        self.train = args.train
        self.logger = logger
        self.device = device

        model.to(self.device)

        # NOTE: If no checkpoint has been passed, set hyperparameters and more settings. The current epoch is 0
        if checkpoint is None:
            # NOTE: Loss
            self.criterion = nn.CrossEntropyLoss()
            # NOTE: parameters
            self.params = filter(lambda p: p.requires_grad, model.parameters())
            # NOTE: Use Adam as optimizer
            self.optimizer = torch.optim.Adam(self.params, lr=args.learning_rate)
            # NOTE TODO not fully sure what this is. does this not depend on batch_size and number of epochs?
            self.total_steps = len(data_loader)
            self.num_epochs = args.num_epochs
            # NOTE: How often we log
            self.log_step = args.log_step
            self.curr_epoch = 0

        # NOTE: set the vocabulary and the start and end tokens
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

        # NOTE and TODO: this goes through all batches, not through all instances, right?
        # NOTE and TODO: Daniela: I think this should be the whole data
        for i, (image_input, word_inputs, word_targets, lengths, ids, *excess) in enumerate(self.data_loader):
            # Prepare mini-batch dataset, send to CPU or CUDA
            image_input = image_input.to(self.device)

            if self.train:
                word_inputs = word_inputs.to(self.device)
                word_targets = word_targets.to(self.device)
                word_targets = pack_padded_sequence(word_targets, lengths, batch_first=True)[0]

                # NOTE: train, compute the loss for the current batch and append to the result; the method used is
                # train_step(), see below
                loss = self.train_step(image_input, word_inputs, word_targets,
                        lengths, *excess)
                result.append(loss.data.item())

                # NOTE: compute the step number and log step number and corresponding loss
                step = self.curr_epoch * self.total_steps + i + 1
                self.logger.scalar_summary('batch_loss', loss.data.item(), step)

            else:
                # NOTE: if in evaluation, captions are generated to be returned later, using eval_step() (see below).
                # NOTE and TODO: Why does this happen here, in train_epoch?
                generated_captions = self.eval_step(image_input, ids, *excess)
                result.extend(generated_captions)

            # NOTE and TODO: The logging happens after each log_step batches (?), but the comments in the original
            # code (the To-do comment) indicate that it is not fully implemented. This does not really have to
            # bother us because the code probably works even without it.

            # NOTE and TODO: (DANIELA Yes. What they do here is simply print out the current loss and perplexity.
            #  It's not logged using Tensorboard. I don't think it's particularly relevant to implement the logging.
            #  It does not affect the main functioning of the code

            # TODO: Add proper logging
            # Print log info
            if i % self.log_step == 0:
                print("Epoch [{}/{}], Step [{}/{}]".format(self.curr_epoch,
                    self.num_epochs, i, self.total_steps), end='')
                if self.train:
                    print(", Loss: {:.4f}, Perplexity: {:5.4f}".format(loss.data.item(),
                        np.exp(loss.data.item())), end='')
                print()

        # NOTE: The trainer counts the epochs it is called. This information is used in the main.py
        # code to define when it will stop training
        self.curr_epoch += 1

        # NOTE: just as after each batch, after the end of the whole epoch, the total loss is logged
        if self.train:
            self.logger.scalar_summary('epoch_loss', np.mean(result), self.curr_epoch)

        return result

    # NOTE: this is a rather standard training step (get output and loss, and optimize depending on the latter)
    def train_step(self, image_input, word_inputs, word_targets, lengths,
            *args):
        # Forward, Backward and Optimize
        self.model.zero_grad()
        outputs = self.model(image_input, word_inputs, lengths)
        loss = self.criterion(outputs, word_targets)
        loss.backward()
        self.optimizer.step()

        return loss

    # NOTE: this is the evaluation step used in train_epoch()
    def eval_step(self, image_input, ids, *args):
        # TODO: max_sampling_length
        vocab = self.dataset.vocab
        generated_captions = []
        outputs = self.model.generate_sentence(image_input, self.start_word, self.end_word, *args)
        # NOTE: The output is a continuous stream of word indices. The following nested for-loops are used to convert
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

        return generated_captions
