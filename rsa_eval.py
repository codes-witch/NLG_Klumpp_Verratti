import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from rsa import IncRSA, BirdDistractorDataset
from rsa import load_model

puncs = set(string.punctuation)
en_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def load_inc_rsa_model(rsa_dataset):
    """
    Loads an RSA model and initializes an IncRSA object

    :param rsa_dataset: the dataset
    :return: the model
    """
    model = load_model(rsa_dataset)
    rsa_model = IncRSA(model, rsa_dataset)
    return rsa_model


class KeywordClassifier(object):
    """
    A class for finding out which issues a caption resolves. Detects when a word that would resolve an issue is in the
    vicinity of the relevant organ. Used for evaluating issue alignment in the issue_alignment and two_issues files.
    """

    def __init__(self, rsa_dataset):
        """
        In general: sets the following instance variable:
        - Rsa_dataset
        - Organ_name_to_match_words: dictionary with synonyms of organ names
        - Organ_name_to_aspect_name: a nested dictionary. Maps organ to possible aspect names and those
       aspect names to possible values
        - Color_match_words: a set of color names
        Rest of comments are original
       """
        self.rsa_dataset = rsa_dataset

        # dictionary with synonyms of organ names
        self.organ_name_to_match_words = {}
        # a nested dictionary, maps organ to possible aspect names and those aspect names to possible values
        self.organ_name_to_aspect_name = {}
        # dict mapping string of the form has_organ_aspect::value to a tuple (organ, aspect, value)
        self.attr_name_to_decomp = {}
        self.segment_name_to_decomp = {}

        # i is an attribute id / index, seg_attr is an attribute name and value
        for i, seg_attr in enumerate(self.rsa_dataset.attr_vocab_ls):
            # split to obtain a list [attribute, value]
            a = seg_attr.split("::")
            # split again to obtain a list ["has", bodypart, aspect]
            # special cases: has_size, has_shape and has_upper_tail_color.
            b = a[0].split("_")
            # b[1] is always a body part except in has_shape and has_size
            organ_name = b[1]

            # for has_shape and has_size, there is no further distinction. Set aspect_name to None for these two
            # segments.
            # (Clarification: "aspect" refers to nouns like "color", "pattern", "shape", "length")
            if len(b) == 2:
                aspect_name = None
            # for compound body part names upper_tail and under_tail, merge them, and take the last element as aspect
            # name
            elif len(b) == 4:
                aspect_name = b[3]
                organ_name = b[1] + '_' + b[2]
            # else: take the last element as aspect name
            else:
                aspect_name = b[2]

            # maps full attribute string of the form has_organ_aspect::value to its decomoposed parts:
            # a tuple (organ, aspect, value)
            self.attr_name_to_decomp[seg_attr] = (organ_name, aspect_name, a[1])
            # maps full segment name of the form has_organ_aspect to an (organ, aspect) tuple
            self.segment_name_to_decomp[seg_attr.split("::")[0]] = (organ_name, aspect_name) #
            # add the name of the organ as a synonym to itself
            self.organ_name_to_match_words[organ_name] = {organ_name}

            # for bill-length
            if aspect_name == 'length':
                # map 'bill' to its possible aspect names. in our case, long(er) and short(er).
                # add organ name to the dictionary if not already there
                if organ_name not in self.organ_name_to_aspect_name:
                    self.organ_name_to_aspect_name[organ_name] = {aspect_name: {'long', 'short', 'longer', 'shorter'}}
                else:
                    self.organ_name_to_aspect_name[organ_name][aspect_name] = {'long', 'short', 'longer', 'shorter'}
                continue

            # has_shape and has_size have aspect_name = None. For all others:
            if aspect_name is not None:
                # attr_set is a set containing all descriptors (descriptive words after '::') for the given organ and
                # aspect
                attr_set = self.get_attr_descriptor_for_organ(organ_name, aspect_name)

                # populate the dictionary for the organ to aspect name. Use attr_set as the possible descriptors
                if organ_name not in self.organ_name_to_aspect_name:
                    self.organ_name_to_aspect_name[organ_name] = {aspect_name: attr_set}
                else:
                    self.organ_name_to_aspect_name[organ_name][aspect_name] = attr_set

                # Original comment: every organ beside head shares same pattern descriptor
                if organ_name != 'head' and aspect_name == 'pattern':
                    # add more descriptors to organ_name_to_aspect_name
                    self.fill_organ_aspect_key_words(organ_name, aspect_name, ['striped', 'stripe',
                                                                               'speckle', 'speckled',
                                                                               'multicolored', 'multicolor',
                                                                               'specks', 'speck',
                                                                               'ornate', 'scattered',
                                                                               'coloring', 'spots', 'spot',
                                                                               'rounded', 'mottled',
                                                                               'tuft', 'webbed', 'puffy',
                                                                               'pointy'])
        # add synonyms to organs
        self.expand()
        # add more possible aspects
        self.aspect_expand()

        # all the organs in this list can be used to refer to "upperparts"
        for name in ['wing', 'throat', 'head', 'forehead', 'nape', 'upper_tail', 'crown', 'breast']:
            self.organ_name_to_match_words['upperparts'].update(self.organ_name_to_match_words[name])

        # all the organs in this list can refer to underparts
        for name in ['leg', 'under_tail', 'belly']:
            self.organ_name_to_match_words['underparts'].update(self.organ_name_to_match_words[name])

        # get the descriptors to describe color
        self.color_match_words = self.get_attr_descriptors("color")
        # add more colors
        self.color_match_words.update(['navy', 'bluish', 'violet', 'scarlet', 'greenish', 'silrumpver', 'teal',
                                       'pinkish', 'colored', 'color', 'multicolored', 'multicolor',
                                       'tan', 'bright', 'dark', 'brown', 'brownish', 'vibrant',
                                       'gray', 'pale', 'russet', 'yellow', 'orange', 'golden',
                                       'coloring', 'toned', 'shiny', 'pink', 'vivid', 'blackish'])

        # add color_match_words to aspect_name_to_match_words
        self.fill_color_key_words()

        # assume 'size' and 'shape' to be organs and give their possible descriptors
        self.organ_name_to_match_words['size'] = ['large', 'small', 'very large', 'medium', 'very small',
                                                  'petite', 'pudgy', 'smal', 'slim', 'huge', 'elongated',
                                                  'skinny', 'sized', 'thick', 'short', 'long', 'shorter', 'longer',
                                                  'puffy']
        self.organ_name_to_match_words['shape'] = ['plump', 'mohawk', 'perching', 'perch', 'gull', 'humming',
                                                   'clinging', 'hawk', 'rounded', 'round', 'puffy']

    def get_attr_descriptors(self, aspect_word):
        """
        Returns a set of colors or other matching words that are found in the attribute list

        :param aspect_word: the aspect for which to get the words
        :return: list of matching words
        """
        # get the attributes mentioning the aspect_word (for example, colors)
        color_attrs = [t.split("::")[0] for t in self.rsa_dataset.q_id_to_segments if aspect_word in t]
        uniq_colors = set()
        # get the descriptors for said attributes (for example, color names)
        for c_a in color_attrs:
            colors = [t.split("::")[1] for t in self.rsa_dataset.attr_vocab_ls if c_a in t]
            uniq_colors.update(colors)

        # return the set of possible values for the attribute
        return uniq_colors

    def get_attr_descriptor_for_organ(self, organ_name, aspect_word, verbose=False):
        """
        Gets all the attribute labels that contain the organ name and the aspect given and puts them in a set

        """
        # get all attributes with the organ and aspect we are looking for
        attrs = [t for t in self.rsa_dataset.attr_vocab_ls if organ_name in t and aspect_word in t]
        if verbose:
            print(attrs)
        uniq_attrs = set()

        for a in attrs:
            a = a.split("::")[1] # get the descriptor (the value of the attribute)
            # "curved", "up" and "down" are all valid descriptors. Add them to the set
            if a == 'curved_(up_or_down)':
                uniq_attrs.add('curved')
                uniq_attrs.add('up')
                uniq_attrs.add('down')
            # if there are several words in the descriptor, add them all to the set of possible descriptors
            elif '_' in a:
                uniq_attrs.update(a.split('_'))
            # get descriptors describing wings (they all have 'wing' as the second element, so simply get the adjective
            # '-wing')
            elif 'wings' in a:
                a = a.split('-')[0]
                uniq_attrs.add(a)
            # in all other cases, just add the descriptor
            else:
                uniq_attrs.add(a)

        return uniq_attrs

    def aspect_expand(self):
        """
        Uses fill_organ_aspect_key_words to expand the dictionary
        """
        self.fill_organ_aspect_key_words('bill', 'shape', ['triangular', 'pointed', 'curved', 'pointy'])
        self.fill_organ_aspect_key_words('bill', 'length', ['large', 'small', 'tiny', 'huge'])
        self.fill_organ_aspect_key_words('tail', 'shape', ['fan'])
        self.fill_organ_aspect_key_words('head', 'pattern', ['streak'])
        self.fill_organ_aspect_key_words('wing', 'shape', ['long', 'large'])

    def fill_organ_aspect_key_words(self, organ_name, aspect_name, keywords):
        """
        Adds all the keywords as possible aspect values in organ_name_to_match_words dictionary,
        using fill_organ_key_words

        :param organ_name: name of the organ
        :param aspect_name: name of the aspect
        :param keywords: list of keywords to add
        """
        self.organ_name_to_aspect_name[organ_name][aspect_name].update(keywords)

    def expand(self):
        """
        Adds synonyms and related words to the organ names in organ_name_to_match_words dictionary, using
        fill_organ_key_words
        """
        # since we lemmatize everything, we do not need to be extensive with all the morphological variations of a word
        self.fill_organ_key_words("leg", ['tarsal', 'tarsals', 'tarsuses', 'tarsus', 'foot', 'thighs',
                                          'feet', 'claws', 'claw', 'legs'])
        self.fill_organ_key_words("wing", ['wingbars', 'wingbar', 'rectricles', 'rectricle', 'retrice',
                                           'gull', 'tip', 'tips', 'primaries', 'primary', 'secondaries',
                                           'secondary', 'converts', 'convert', 'retrices',
                                           'wingspan', 'wingspans'])
        self.fill_organ_key_words('head', ['malar', 'malars', 'cheekpatch', 'eyebrows', 'cheek',
                                           'superciliary', 'eyebrow', 'eyering', 'eyeline', 'eyelines',
                                           'eyerings', 'ring', 'rings'])
        self.fill_organ_key_words('bill', ['beek', 'beaks', 'beak', 'beeks', 'hook', 'bil'])
        self.fill_organ_key_words('under_tail', ['undertail', 'tail', 'rump'])
        self.fill_organ_key_words('belly', ['underbelly', 'stomach', 'plumage', 'feather', 'feathers',
                                            'abdomen', 'side'])
        self.fill_organ_key_words('breast', ['chest', 'stomach', 'plumage', 'feather', 'feathers', 'breasted'])
        self.fill_organ_key_words('upperparts', ['body', 'side', 'sides'])
        self.fill_organ_key_words('forehead', ['forehead'])
        self.fill_organ_key_words('back', ['plumage'])
        self.fill_organ_key_words('primary', ['primaries'])
        self.fill_organ_key_words('throat', ['neck'])
        self.fill_organ_key_words('crown', ['crest'])
        self.fill_organ_key_words('upper_tail', ['rump', 'tail'])
        self.fill_organ_key_words('eye', ['eyebrows', 'superciliary', 'eyebrow', 'eyering', 'eyeline', 'eyelines',
                                          'eyerings', 'ring', 'rings'])

    def fill_organ_key_words(self, name, list_of_words):
        """
        Adds the words here as synonyms to the organ name. This happens in organ_name_to_match_words dict

        :param name: organ name
        :param list_of_words: list of synonyms
        """
        self.organ_name_to_match_words[name].update(list_of_words)

    def fill_color_key_words(self):
        """
        Adds all the colors to the organ_name_to_aspect_name inner dictionary. This happens for all organs, each for
        the aspect 'color', since every organ can be described in terms of its color (in contrast to 'size', 'shape',
        etc., which are only available for some organs)
        """
        for organ_name, aspect_name_match_words in self.organ_name_to_aspect_name.items():
            if 'color' in aspect_name_match_words:
                aspect_name_match_words['color'].update(self.color_match_words)

    def classify_parts(self, part_name, text, tokenize=False):
        """
        Try to find body parts in the given text. Return whether the body part was found (boolean) and the list of
        indices where it occurs. It also gets lemmatized in case we find it that way.

        :param part_name: name of the organ / body part
        :param text: the text in which to search for the organ name (tokenized or untokenized)
        :param tokenize: False if text already is tokenized, True otherwise
        :return: boolean whether the body part is mentioned, list of indices where it is mentioned
        """

        # the name of the body part has to be one of those considered here
        assert part_name in self.organ_name_to_match_words

        # get all words denoting that body part
        keywords = self.organ_name_to_match_words[part_name]

        # tokenize if not already done
        words = nltk.word_tokenize(text) if tokenize else text

        found, ind_list = False, []

        # go through all words in the text
        for i, w in enumerate(words):
            # if the word occurs in its base form in the text, it's fine
            if w in keywords:
                found = True
                ind_list.append(i)
                continue
            # otherwise, we also check whether it occurs in another form (by lemmatizing)
            w = lemmatizer.lemmatize(w.lower())
            if w in keywords:
                found = True
                ind_list.append(i)

        return found, ind_list

    def classify_parts_aspect(self, part_name, aspect_name, text, window_size=3, tokenize=False):
        """
        Identify whether keywords characterizing a specific body part are found within a context window. This is used
        to calculate issue alignment.

        :param part_name: the body part / organ
        :param aspect_name: the aspect (such as 'size', 'color')
        :param text: the text in which to look for keywords (i.e., the caption)
        :param window_size: the size of the context window
        :param tokenize: False if text already is tokenized, True otherwise
        :return: True if keyword is found, False otherwise
        """

        # both body part and aspect must be among those defined before
        assert part_name in self.organ_name_to_match_words
        assert aspect_name in self.organ_name_to_aspect_name[part_name]

        # tokenize if not already done
        if tokenize:
            text = nltk.word_tokenize(text)

        # make sure that tokenization indeed has happened and produced a list
        assert type(text) == list
        # find out whether and where the body part is mentioned
        found, idx_list = self.classify_parts(part_name, text, tokenize=False)

        # get all keywords describing the relevant aspect for the chosen body part
        keywords = self.organ_name_to_aspect_name[part_name][aspect_name]

        # if the body part is not mentioned at all, return False
        if not found:
            return False
        # if it is mentioned, look for keywords
        else:
            # each occurrence of the body part is considered
            for i in idx_list:
                # the context window ranges from window_size words before (or the beginning of the text)
                # to the index where the body part is mentioned
                lookahead_idx = max(i - window_size, 0)
                text_span = text[lookahead_idx:i]
                # within that window, each word is checked for whether it belongs to the relevant keywords
                for t in text_span:
                    # if it is, return True
                    if t in keywords:
                        return True
        # if no pair of keyword and body part within the context window are found, return False
        return False


if __name__ == '__main__':
    nltk.download('stopwords')
    rsa_dataset = BirdDistractorDataset()
    classy = KeywordClassifier(rsa_dataset)

    print(classy.organ_name_to_aspect_name)

