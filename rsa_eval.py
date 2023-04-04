import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn import metrics

import sys

if '/home/anie/AutoGrade/simplepg' in sys.path:
    del sys.path[sys.path.index("/home/anie/AutoGrade/simplepg")]

from rsa_notes import IncRSA, BirdDistractorDataset
from rsa_notes import load_model

puncs = set(string.punctuation)
en_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def load_inc_rsa_model(rsa_dataset):
    model = load_model(rsa_dataset)
    rsa_model = IncRSA(model, rsa_dataset)
    return rsa_model


class KeywordClassifier(object):
    def __init__(self, rsa_dataset):
        # we need:
        # 1. structure that is: {name: match_keywords}
        # 2. We go from organ -> aspect -> attributes
        # where we only do organ -> aspect (segment) first
        # For each segment, then we have:
        # {seg_name/id: [organ_name, aspect_name, attr_name]}

        self.rsa_dataset = rsa_dataset

        # organ first
        self.organ_name_to_match_words = {}
        # aspect second
        self.organ_name_to_aspect_name = {}
        self.attr_name_to_decomp = {}
        self.segment_name_to_decomp = {}

        for i, seg_attr in enumerate(self.rsa_dataset.attr_vocab_ls):
            a = seg_attr.split("::") # DANIELA Get list [segment, value]
            b = a[0].split("_") # DANIELA Get list [verb bodypart (bodypart) noun]
            organ_name = b[1] # DANIELA The second thing is always a body part excep in has_shape and has_size
            if len(b) == 2:
                aspect_name = None
            elif len(b) == 4:
                aspect_name = b[3]
                organ_name = b[1] + '_' + b[2]
            else:
                aspect_name = b[2] # DANIELA aspect is (color|pattern)
            # aspect_name = b[2] if len(b) != 2 else None
            self.attr_name_to_decomp[seg_attr] = (organ_name, aspect_name, a[1]) # DANIELA full string to triple (organ,
                                                                                    # aspect, value)
            self.segment_name_to_decomp[seg_attr.split("::")[0]] = (organ_name, aspect_name) #

            self.organ_name_to_match_words[organ_name] = {organ_name}

            # for bill-length
            if aspect_name == 'length':
                if organ_name not in self.organ_name_to_aspect_name:
                    self.organ_name_to_aspect_name[organ_name] = {aspect_name: {'long', 'short', 'longer', 'shorter'}}
                else:
                    self.organ_name_to_aspect_name[organ_name][aspect_name] = {'long', 'short', 'longer', 'shorter'}
                continue
                # DANIELA organ name to aspect name maps organ names to a dictionary of possible aspect names,
                # and these are mapped to a set of their possible values (at least for length o far)

            # we actually ignored "has_shape" and "has_size"
            if aspect_name is not None:
                attr_set = self.get_attr_descriptor_for_organ(organ_name, aspect_name)
                if organ_name not in self.organ_name_to_aspect_name:
                    self.organ_name_to_aspect_name[organ_name] = {aspect_name: attr_set}
                else:
                    self.organ_name_to_aspect_name[organ_name][aspect_name] = attr_set

                # update pattern keywords!
                if organ_name != 'head' and aspect_name == 'pattern':
                    # add pattern descriptors here
                    # every organ beside head shares same pattern descriptor
                    self.fill_organ_aspect_key_words(organ_name, aspect_name, ['striped', 'stripe',
                                                                               'speckle', 'speckled',
                                                                               'multicolored', 'multicolor',
                                                                               'specks', 'speck',
                                                                               'ornate', 'scattered',
                                                                               'coloring', 'spots', 'spot',
                                                                               'rounded', 'mottled',
                                                                               'tuft', 'webbed', 'puffy',
                                                                               'pointy'])

        # for organ names, most are FINE
        # but not for "upperparts", "underparts", "back", "under_tail", "size", "shape", "primary", 'upper_tail'
        self.expand()
        self.aspect_expand()

        for name in ['wing', 'throat', 'head', 'forehead', 'nape', 'upper_tail', 'crown', 'breast']:
            self.organ_name_to_match_words['upperparts'].update(self.organ_name_to_match_words[name])

        for name in ['leg', 'under_tail', 'belly']:
            self.organ_name_to_match_words['underparts'].update(self.organ_name_to_match_words[name])

        self.color_match_words = self.get_attr_descriptors("color")
        self.color_match_words.update(['navy', 'bluish', 'violet', 'scarlet', 'greenish', 'silrumpver', 'teal',
                                       'pinkish', 'colored', 'color', 'multicolored', 'multicolor',
                                       'tan', 'bright', 'dark', 'brown', 'brownish', 'vibrant',
                                       'gray', 'pale', 'russet', 'yellow', 'orange', 'golden',
                                       'coloring', 'toned', 'shiny', 'pink', 'vivid', 'blackish'])
        self.fill_color_key_words()

        self.organ_name_to_match_words['size'] = ['large', 'small', 'very large', 'medium', 'very small',
                                                  'petite', 'pudgy', 'smal', 'slim', 'huge', 'elongated',
                                                  'skinny', 'sized', 'thick', 'short', 'long', 'shorter', 'longer',
                                                  'puffy']
        self.organ_name_to_match_words['shape'] = ['plump', 'mohawk', 'perching', 'perch', 'gull', 'humming',
                                                   'clinging', 'hawk', 'rounded', 'round', 'puffy']

    def print_descriptors(self, skip_color=True):
        print("all_organs", self.organ_name_to_aspect_name.keys())
        print()
        for organ_name, aspect_name_match_words in self.organ_name_to_aspect_name.items():
            if 'color' in aspect_name_match_words:
                if skip_color:
                    continue
            print(organ_name, aspect_name_match_words)
            print()

    def get_attr_descriptors(self, aspect_word):
        color_attrs = [t.split("::")[0] for t in self.rsa_dataset.q_id_to_segments if aspect_word in t]
        uniq_colors = set()
        for c_a in color_attrs:
            colors = [t.split("::")[1] for t in self.rsa_dataset.attr_vocab_ls if c_a in t]
            uniq_colors.update(colors)

        return uniq_colors

    def get_attr_descriptor_for_organ(self, organ_name, aspect_word, verbose=False):
        attrs = [t for t in self.rsa_dataset.attr_vocab_ls if organ_name in t and aspect_word in t]
        if verbose:
            print(attrs)
        uniq_attrs = set()
        for a in attrs:
            a = a.split("::")[1]
            if a == 'curved_(up_or_down)':
                uniq_attrs.add('curved')
                uniq_attrs.add('up')
                uniq_attrs.add('down')
            elif '_' in a:
                uniq_attrs.update(a.split('_'))
            elif 'wings' in a:
                a = a.split('-')[0] # DANIELA the word after - is always wings
                uniq_attrs.add(a)
            else:
                uniq_attrs.add(a)

        return uniq_attrs

    def aspect_expand(self):
        self.fill_organ_aspect_key_words('bill', 'shape', ['triangular', 'pointed', 'curved', 'pointy'])
        self.fill_organ_aspect_key_words('bill', 'length', ['large', 'small', 'tiny', 'huge'])
        self.fill_organ_aspect_key_words('tail', 'shape', ['fan'])
        self.fill_organ_aspect_key_words('head', 'pattern', ['streak'])
        self.fill_organ_aspect_key_words('wing', 'shape', ['long', 'large'])

    def fill_organ_aspect_key_words(self, organ_name, aspect_name, keywords):
        self.organ_name_to_aspect_name[organ_name][aspect_name].update(keywords)

    def expand(self):
        """
        {'bill': {'bill'},
         'wing': {'wing'},
         'upperparts': {'upperparts'},
         'underparts': {'underparts'},
         'breast': {'breast'},
         'back': {'back'},
         'tail': {'tail'},
         'upper_tail': {'upper_tail'},
         'head': {'head'},
         'throat': {'throat'},
         'eye': {'eye'},
         'forehead': {'forehead'},
         'under_tail': {'under_tail'},
         'nape': {'nape'},
         'belly': {'belly'},
         'size': {'size'},
         'shape': {'shape'},
         'primary': {'primary'},
         'leg': {'leg'},
         'crown': {'crown'}}
        """
        # we lemmatize everything, so it's fine-
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
        self.fill_organ_key_words('forehead', ['forehead'])  # 'eyebrows', 'eyebrow', 'eyering'
        self.fill_organ_key_words('back', ['plumage'])
        self.fill_organ_key_words('primary', ['primaries'])
        self.fill_organ_key_words('throat', ['neck'])
        self.fill_organ_key_words('crown', ['crest'])
        self.fill_organ_key_words('upper_tail', ['rump', 'tail'])
        self.fill_organ_key_words('eye', ['eyebrows', 'superciliary', 'eyebrow', 'eyering', 'eyeline', 'eyelines',
                                          'eyerings', 'ring', 'rings'])

    def fill_organ_key_words(self, name, list_of_words):
        # we collect a lot of them and fill them up
        self.organ_name_to_match_words[name].update(list_of_words)

    def fill_color_key_words(self):
        # iterate through ALL
        for organ_name, aspect_name_match_words in self.organ_name_to_aspect_name.items():
            if 'color' in aspect_name_match_words:
                aspect_name_match_words['color'].update(self.color_match_words)

    def classify_parts(self, part_name, text, tokenize=False):
        # basically we just try different ways to change the text and match
        assert part_name in self.organ_name_to_match_words

        keywords = self.organ_name_to_match_words[part_name] # get all aspects we might want to match

        words = nltk.word_tokenize(text) if tokenize else text

        found, ind_list = False, []

        # then we lemmatize text
        for i, w in enumerate(words):
            if w in keywords:
                found = True
                ind_list.append(i)
                continue
            w = lemmatizer.lemmatize(w.lower())
            if w in keywords:
                found = True
                ind_list.append(i)

        return found, ind_list

    def classify_parts_aspect(self, part_name, aspect_name, text, window_size=3, tokenize=False):
        # if no tokenize, we expect a list of words
        assert part_name in self.organ_name_to_match_words
        assert aspect_name in self.organ_name_to_aspect_name[part_name]

        # here we first identify the location of body parts
        # then we look ahead for a fixed window (3-5 words) for the aspect match
        if tokenize:
            text = nltk.word_tokenize(text)

        assert type(text) == list
        found, idx_list = self.classify_parts(part_name, text, tokenize=False)

        keywords = self.organ_name_to_aspect_name[part_name][aspect_name]

        if not found:
            return False
        else:
            for i in idx_list:
                # check previous 5 words
                # a = [0,1,2,3,4]
                # a[1:3] = [1, 2]
                lookahead_idx = max(i - window_size, 0)
                text_span = text[lookahead_idx:i]
                for t in text_span:
                    if t in keywords:
                        return True

        return False


if __name__ == '__main__':
    nltk.download('stopwords')
    rsa_dataset = BirdDistractorDataset()
    classy = KeywordClassifier(rsa_dataset)

    print(classy.organ_name_to_aspect_name)

