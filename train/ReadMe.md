# Train

This subdirectory contains the code that defines the evaluation and training of an epoch of the LRCN, GVE and
SentenceClassifier models. All code in this subdirectory was written by the original authors. All documentation is ours unless explicitly stated 
(preceded by "Original comment(s)"). 

## Files

### LRCN trainer

Contains functions for the following purposes:

 - Training step: the usual sequence of forward pass, loss calculation, backpropagation and 
optimization. 
 - Evaluation step: generates sentences which are used for evaluation in `main.py`.
 - Training epoch: iterates through all the data and passes it through the training or evaluation step.

### GVE trainer

Extends the `LRCNTrainer` class. Overrides the `train_step` method. A GVE training step entails obtaining class 
predictions from the sentence classifier and calculating two kinds of cost function: discriminative loss and 
relevance loss. The total loss is obtained by combining these two (see Hendricks et al. for details). Finally, 
we back-propagate the error and do an optimization step.

### Sentence classifier

Follows a very similar structure to `lrcn_trainer.py`. The main difference is found in the `eval_step` method, where the
output is the total number of classes and the number of correct class predictions. This information is used in the 
evaluation block of `train_epoch` to compute the accuracy of the classifier.