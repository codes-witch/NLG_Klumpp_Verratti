import torch

#from .lrcn import LRCN
#from .gve import GVE
#from .sentence_classifier import SentenceClassifier

class ModelLoader:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset