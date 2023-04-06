import torch

from .lrcn import LRCN
from .gve import GVE
from .sentence_classifier import SentenceClassifier


class ModelLoader:
    """
    ModelLoader to load the LRCN, GVE and SentenceClassifier model.
    """

    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

    def lrcn(self):
        # Set LRCN arguments
        pretrained_model = self.args.pretrained_model
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)
        # number of layers we wish to truncate
        layers_to_truncate = self.args.layers_to_truncate

        lrcn = LRCN(pretrained_model, embedding_size, hidden_size, vocab_size,
                    layers_to_truncate)

        return lrcn

    def gve(self):
        # we want to use the CUB data with labels (bird species)
        self.dataset.set_label_usage(True)
        # set GVE arguments
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)
        input_size = self.dataset.input_size
        num_classes = self.dataset.num_classes

        # GVE model requires a sentence classifier (Generating Visual Explanations, Hendricks et al., 2016)
        sc = self.sc()

        # Load the state dictionary for the sentence classifier from checkpoint, either to CUDA or CPU device
        if torch.cuda.is_available():
            sc.load_state_dict(torch.load(self.args.sc_ckpt))
        else:
            sc.load_state_dict(torch.load(self.args.sc_ckpt, map_location='cpu'))

        # SC is not trained. Its parameters do not require gradient.
        for param in sc.parameters():
            param.requires_grad = False
        sc.eval()

        # initialize the model
        gve = GVE(input_size, embedding_size, hidden_size, vocab_size, sc,
                  num_classes)

        # load the weights from checkpoint
        if self.args.weights_ckpt:
            gve.load_state_dict(torch.load(self.args.weights_ckpt))

        return gve

    def sc(self):
        # we want to use the CUB data with labels (bird species)
        self.dataset.set_label_usage(True)
        # set necessary parameters
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        # number of words in vocabulary
        vocab_size = len(self.dataset.vocab)
        # number of bird species
        num_classes = self.dataset.num_classes

        sc = SentenceClassifier(embedding_size, hidden_size, vocab_size,
                                num_classes)

        return sc
