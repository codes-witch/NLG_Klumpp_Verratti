# Utils

This folder contains several helper methods needed for different parts of the training, caption generation, 
and evaluation process. 
All code in this subdirectory was written by the original authors (but shortened by us to some extent). 
All documentation is ours unless explicitly stated otherwise (preceded by "Original comment(s)").

## Subdirectories

### Data

Contains the classes DataPreparation, CocoDataset, and CubDataset. These are used to set up and access the data
(images, captions, and labels).

### Tokenizer

Contains the PTBTokenizer, which is a widespread standard tokenizer. We did not change anything in this 
directory, so both code and documentation are from the original.

## Files

### Arg Parser

This file is needed for argument parsing when running main.py (for training or evaluation). The arguments are set
to default values or individual input values.

### Logger

The logger is used in main.py for the logging.

### Misc

This file contains a single helper method get_split_str(), which is used to convert the representation of the 
data split (training, validation, or testing) from boolean to string type.

### Transform

The method get_transform is used to define a transformation pipeline based on a vision model. The only 
implemented vision model is VGG.

### Vocabulary

Class representing the vocabulary. This includes the definition of available words (including start, end, 
unknown, and padding tokens) and the indices corresponding to them. This is needed wherever a text is converted 
into or derived from a numerical representation (e.g. for the output in caption generation).
