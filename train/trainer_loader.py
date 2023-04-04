from .lrcn_trainer_notes import LRCNTrainer
from .gve_trainer import GVETrainer
from .sentence_classifier_trainer import SCTrainer

class TrainerLoader:
    """
    Loading the trainer classes for LRCN, SentenceClassifier and GVE
    """
    lrcn = LRCNTrainer
    gve = GVETrainer
    sc = SCTrainer
