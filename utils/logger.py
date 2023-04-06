from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """
    Class for logging scalar variables using tensorboard.
    """
    def __init__(self, log_dir):
        """
        Original comment:
        Create a summary writer logging to log_dir.
        """
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """
        Original comment: Log a scalar variable.
        """
        # This is used for logging epoch and batch loss in main.py as well as in the LCRN and
        # SentenceClassifier trainers
        self.writer.add_scalar(tag, value, step)
