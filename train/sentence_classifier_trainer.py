import torch
import torch.nn as nn
import numpy as np


class SCTrainer:
    """
    Code for training an epoch and evaluating the sentence classifier.
    """
    REQ_EVAL = False

    # set parameters
    def __init__(self, args, model, dataset, data_loader, logger, device, checkpoint=None):
        self.model = model
        self.dataset = dataset
        self.data_loader = data_loader
        self.train = args.train
        self.logger = logger
        self.device = device

        model.to(self.device)

        # If no checkpoint has been given, set the hyperparameters for training
        if checkpoint is None:
            self.criterion = nn.CrossEntropyLoss()
            # all parameters require gradient
            self.params = filter(lambda p: p.requires_grad, model.parameters())
            # use Adam as the optimizer
            self.optimizer = torch.optim.Adam(self.params, lr=args.learning_rate)
            # number of steps per epoch
            self.total_steps = len(data_loader)
            self.num_epochs = args.num_epochs
            # log every log_step number of steps
            self.log_step = args.log_step
            self.curr_epoch = 0

    def train_epoch(self):
        # Original comment: Result is list of losses during training and generated captions during evaluation
        result = []

        for i, (images, word_inputs, word_targets, lengths, ids, labels) in enumerate(self.data_loader):
            # Original comment: Prepare mini-batch dataset
            if self.train:
                word_targets = word_targets.to(self.device)
                labels = labels.to(self.device)
                # calculate loss
                loss = self.train_step(word_targets, labels, lengths)
                result.append(loss.data.item())
                # calculate current step
                step = self.curr_epoch * self.total_steps + i + 1
                # log current batch loss
                self.logger.scalar_summary('batch_loss', loss.data.item(), step)

            # if we are evaluating, calculate the score of our predictions
            else:
                word_targets = word_targets.to(self.device)
                labels = labels.to(self.device)
                score = self.eval_step(word_targets, labels, lengths)
                result.append(score)

            # print log information after log_step steps
            if i % self.log_step == 0:
                print("Epoch [{}/{}], Step [{}/{}]".format(self.curr_epoch,
                    self.num_epochs, i, self.total_steps), end='')
                if self.train:
                    print(", Loss: {:.4f}, Perplexity: {:5.4f}".format(loss.data.item(),
                                np.exp(loss.data.item())), end='')
                print()

        # keep track of number of epochs trained
        self.curr_epoch += 1

        if self.train:
            self.logger.scalar_summary('epoch_loss', np.mean(result), self.curr_epoch)
        else:
            # calculate accuracy
            result = np.sum(result, axis=0)
            result = result[1] / result[0]
            print("Evaluation Accuracy: {}".format(result))

        return result

    def train_step(self, word_targets, class_labels, lengths):
        # zero out the gradients
        self.model.zero_grad()
        # forward pass
        outputs = self.model(word_targets, lengths)
        # calculate loss, back-propagate and optimize
        loss = self.criterion(outputs, class_labels)
        loss.backward()
        self.optimizer.step()

        return loss

    def eval_step(self, word_targets, class_labels, lengths):
        # predict classes
        outputs = self.model(word_targets, lengths)
        _, predicted = torch.max(outputs.data, 1)
        # number of labels, how many of the predicted class labels were correct (needed for calculating accuracy)
        return [class_labels.size(0), (predicted == class_labels).sum().item()]
