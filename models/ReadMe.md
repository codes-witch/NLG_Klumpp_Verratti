# Models

This folder contains the definitions of the models required for running `main_notes.py` and `evaluation_notes`.
All code in this subdirectory was written by the original authors. All documentation is ours unless explicitly stated 
(preceded by "Original comment(s)"). 

## Files

### Model loader

The model loader `model_loader.py` is in charge of setting the arguments needed for the models LRCN and GVE.
The GVE model (Hendricks et al., 2016) requires a pretrained sentence classifier whose arguments are also set in the
`model_loader.py` file. 

### LRCN

The main contents of this file define the forward pass for the LRCN architecture used in the code. It takes as input the
training captions and the image feature vectors. These are processed separately at the beginning. The embeddings of the 
captions are obtained and passed through an LSTM layer, whereas the image features are passed through a linear layer and
an optional `feat_func`. The outputs of both of these processes are concatenated and further passed through an LSTM 
layer and a final linear layer.

Other functions in this file include weight initialization, a state dictionary, a sampling function from logits and a 
function for sentence generation. 

For sentence generation, the image features are passed through a linear function, a ReLU and an optional `feat_func`.
The generation loop starts with the embedding of the starting token, passed through an LSTM layer and then concatenated 
to the image features. This concatenation then goes through an LSTM layer and, finally, a linear layer. The following
predicted word is obtained either via sampling from the resulting logits or greedily by choosing the maximum logit. The 
embedding of the predicted word is fed to the loop and the process is repeated until we predict the end token or reach a
maximum length.

### GVE

The GVE class extends LRCN and overrides the 

