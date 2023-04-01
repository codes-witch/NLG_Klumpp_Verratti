# ORIGINAL COMMENT: Adapted from
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
import pickle

class Vocabulary():
    """ORIGINAL COMMENT: Simple vocabulary wrapper."""
    def __init__(self, unknown_token='<UNK>', start_token='<SOS>',
                       end_token='<EOS>', padding_token='<PAD>',
                       add_special_tokens=True):
        """
        Set instance variables

        Parameters:
            unknown_token: the sequence used for unkonwn words
            start_token: sequence used for start of sentence
            end_token: sequence used for end of sentence
            padding_token: sequence used for padding
            add_special_tokens: Boolean. If True, unknown_token, start_token, end_token and padding_token are added to
            the vocabulary and therefore have indices assigned to them
        """
        # Dict mapping words to an index
        self.word2idx = {}
        # Dict mapping indices to corresponding words
        self.idx2word = {}
        # Current number of words in the vocabulary
        self.idx = 0
        self.unknown_token = unknown_token
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token

        if add_special_tokens:
            # ORIGINAL COMMENT: add padding first so that it corresponds to 0
            self.add_word(padding_token)
            self.add_word(start_token)
            self.add_word(end_token)
            self.add_word(unknown_token)

    def add_word(self, word):
        """
        Adds a word to the vocabulary
        """
        if not word in self.word2idx:
            # updated dictionaries
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            # Keep track of the next index to use
            self.idx += 1

    def get_word_from_idx(self, idx):
        """
        Get a word from the index. If the word is unknown, get index of <UNK>
        """
        if not idx in self.idx2word:
            return self.unknown_token
        return self.idx2word[idx]

    def __call__(self, word):
        """
        If called with a word, get the index. If the word is not in the dictionary, get index of <UNK>
        """
        if not word in self.word2idx:
            return self.word2idx[self.unknown_token]
        return self.word2idx[word]

    def __len__(self):
        """
        len() is calculated as the number of words in the vocabulary
        """
        return len(self.word2idx)

    @classmethod
    def load(cls, path):
        """
        Loads vocabulary from a pickle file
        """
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        assert isinstance(vocab, cls)
        return vocab

    @classmethod
    def save(cls, vocab, path):
        """
        Saves vocabulary in a pickle file
        """
        assert isinstance(vocab, cls)
        with open(path, 'wb') as f:
            pickle.dump(vocab, f)
