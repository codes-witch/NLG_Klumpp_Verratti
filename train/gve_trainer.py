import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

from .lrcn_trainer_notes import LRCNTrainer

class GVETrainer(LRCNTrainer):
    """
    Defines a training step for the GVE model
    """
    REQ_EVAL = True

    def __init__(self, args, model, dataset, data_loader, logger, device, checkpoint=None):
        super().__init__(args, model, dataset, data_loader, logger, device, checkpoint)
        self.rl_lambda = args.loss_lambda

    def train_step(self, image_input, word_inputs, word_targets, lengths,
            labels):
        # Original comment: Forward, Backward and Optimize

        # obtain one-hot-encoded labels
        labels_onehot = self.model.convert_onehot(labels)
        labels_onehot = labels_onehot.to(self.device)
        # zero out the gradient
        self.model.zero_grad()
        # GVE forward pass
        outputs = self.model(image_input, word_inputs, lengths, labels,
                labels_onehot=labels_onehot)

        # Original comment: Reinforce loss
        # Original comment: Sample sentences
        sample_ids, log_ps, lengths = self.model.generate_sentence(image_input, self.start_word,
                self.end_word, labels, labels_onehot=labels_onehot, max_sampling_length=50, sample=True)
        # Original comment: Order sampled sentences/log_probabilities/labels by sentence length (required by LSTM)
        lengths = lengths.cpu().numpy()
        # the sentence classifiers needs its input to be sorted by length in order for the padded sequences to be packed
        sort_idx = np.argsort(-lengths)
        lengths = lengths[sort_idx]
        sort_idx = torch.tensor(sort_idx, device=self.device, dtype=torch.long)
        # sort the labels and logprobs according to length order
        labels = labels[sort_idx]
        labels = labels.to(self.device)
        log_ps = log_ps[sort_idx,:]
        # sort sample_ids
        sample_ids = sample_ids[sort_idx,:]

        # predict a class (bird species) using the sentence classifier
        class_pred = self.model.sentence_classifier(sample_ids, lengths)
        # convert logits to probability distribution
        class_pred = F.softmax(class_pred, dim=1)
        # calculate discriminative and relevance loss according to Hendricks et al.
        rewards = class_pred.gather(1, labels.view(-1,1)).squeeze()
        r_loss = -(log_ps.sum(dim=1) * rewards).sum()
        loss = self.rl_lambda * r_loss / labels.size(0) + self.criterion(outputs, word_targets)

        # back-propagate and optimize
        loss.backward()
        self.optimizer.step()

        return loss

