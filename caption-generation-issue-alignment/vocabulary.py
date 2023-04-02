

# Adapted from
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
import pickle

class Vocabulary():
    """Original Comment: Simple vocabulary wrapper.

    Added Comment: The vocabulary is needed in caption generation (see rsa.py) to decode the output. The output will
    be a list of indices which have to be converted into words using get_words_from_index().
    """

    def __init__(self, unknown_token='<UNK>', start_token='<SOS>',
                       end_token='<EOS>', padding_token='<PAD>',
                       add_special_tokens=True):
        self.word2idx = {} # a dictionary (keys: words, values: indices)
        self.idx2word = {} # a dictionary (keys: indices, values: words)
        self.idx = 0 # number of indices
        # set the unknown, start, end, and padding token
        self.unknown_token = unknown_token
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token

        if add_special_tokens:
            # Original Comment: add padding first so that it corresponds to 0
            self.add_word(padding_token)
            self.add_word(start_token)
            self.add_word(end_token)
            self.add_word(unknown_token)

    def add_word(self, word):
        # if the word is not yet represented, add it to both dictionaries (mapped to its index)
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            # increase the number of indices
            self.idx += 1

    def get_word_from_idx(self, idx):
        # if no word is mapped to the index, return the unknown token
        if not idx in self.idx2word:
            return self.unknown_token
        # else, return the word at this index
        return self.idx2word[idx]

    def __call__(self, word):
        # if no index is mapped to the word, return the index of the unknown token
        if not word in self.word2idx:
            return self.word2idx[self.unknown_token]
        # else, return the index for the word
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    @classmethod
    def load(cls, path):
        """
        Method to load the vocabulary from a path.

        :param path: the location of the vocabulary (as a pickle file)
        :return: the vocabulary
        """
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        assert isinstance(vocab, cls) # the vocabulary must be an instance of this vocabulary class
        return vocab