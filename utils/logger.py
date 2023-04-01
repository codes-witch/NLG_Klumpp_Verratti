from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """
    This code has been left untouched.
    """

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # OWN COMMENT: This is used for logging epoch and batch loss in main_notes.py as well as in the LCRN and
        # SentenceClassifier trainers
        self.writer.add_scalar(tag, value, step)
