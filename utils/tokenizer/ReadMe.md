# Tokenizer

All code and all comments are original. These files have been left untouched.

## PTBTokenizer file

Provides a way to run the Stanford PTBTokenizer .jar file using Python.

### Tokenize caption

This method returns a list of tokens for a given caption.

### Tokenize

This method takes the `anns` dictionary from a COCO object as input. It obtains all the captions, tokenizes them and 
returns a dictionary mapping each image to its tokenized caption. 
    