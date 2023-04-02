from .lrcn_trainer import LRCNTrainer
from .gve_trainer import GVETrainer
from .sentence_classifier_trainer import SCTrainer

class TrainerLoader:
    """
    class to load the trainer (for LRCN, GVE or SC)
    """
    lrcn = LRCNTrainer
    gve = GVETrainer
    sc = SCTrainer

# TODO: If we want, we can condense all trainers into one file (which saves a directory)