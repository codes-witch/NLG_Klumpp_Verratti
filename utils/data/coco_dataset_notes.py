import os
from collections import Counter
from enum import Enum
import pickle
import json
from PIL import Image

import torch
import torch.utils.data as data
import numpy as np
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from utils.vocabulary import Vocabulary
from utils.tokenizer.ptbtokenizer import PTBTokenizer


# Original Comment:
# Adapted from
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
class CocoDataset(data.Dataset):

    """
    Original Comment: COCO Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    # set COCO-specific paths
    dataset_prefix = 'coco'
    image_path = '{}2014'
    caption_path = 'annotations/captions_{}2014.json'
    vocab_file_name = 'coco_vocab.pkl'
    tokens_file_name = 'coco_tokens_{}.pkl'
    class_labels_path = 'annotations/instances_{}2014.json'

    # for COCO, two data splits are available:
    # 'train' must be contained (as for CUB)
    # there is no 'test' since attributes are not specified in the data (hence the
    # gold standard for evaluation of attribute coverage and issue alignment is missing)
    DATA_SPLITS = set(['train', 'val'])

    # punctuations to be removed from the sentences
    PUNCTUATIONS = ["''", "'", "``", "`", "(", ")", "{", "}",
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    class ID_BASE(Enum):
        CAPTIONS = 0
        IMAGES = 1


    def __init__(self, root, split='train', vocab=None, tokenized_captions=None,
            transform=None):
        """
        Original Comment:
        Args:
            root: directory of coco data
            split: one of ['train', 'val']
        """

        # cls is needed to use the classmethods defined below
        cls = self.__class__
        # the split must be one of those defined above (i.e., 'train' or 'val')
        assert split in cls.DATA_SPLITS
        self.split = split

        self.root = root

        # the paths are defined depending on the root directory
        self.caption_path = os.path.join(self.root, cls.caption_path.format(split))
        self.image_path = os.path.join(self.root, cls.image_path.format(split))
        self.tokens_path = os.path.join(self.root, cls.tokens_file_name.format(split))
        self.vocab_path = os.path.join(self.root, cls.vocab_file_name)
        self.labels_path = os.path.join(self.root, cls.class_labels_path.format(split))

        # get tokenized captions (either by loading or by constructing them)
        # the get_tokenized_captions method is a classmethod defined below
        if tokenized_captions is None:
            tokenized_captions = cls.get_tokenized_captions(self.caption_path,
                    self.tokens_path)

        # get the vocabulary using the path to the training data,
        # which is just the caption path and the tokenized captions in training but has to be specified separately
        # otherwise
        if vocab is None:
            # gets all captions from the training data and tokenizes them.
            if split != 'train':
                cap_path_train = os.path.join(self.root, cls.caption_path.format('train'))
                tokens_path_train = os.path.join(self.root, cls.tokens_file_name.format('train'))
                tokens_train = cls.get_tokenized_captions(cap_path_train,
                                                          tokens_path_train)
            else:
                cap_path_train = self.caption_path
                tokens_train = tokenized_captions
            # vocabulary is obtained from the tokens
            vocab = cls.get_vocabulary(self.vocab_path, cap_path_train, tokens_train)

        # if we are training, we do so with all available captions. Otherwise, we generate captions for each image
        if split == 'train':
            self.ids_based_on = cls.ID_BASE.CAPTIONS
        else:
            self.ids_based_on = cls.ID_BASE.IMAGES

        self.coco = COCO(self.caption_path)

        # get the IDs from the chosen ID base
        if self.ids_based_on == self.ID_BASE.CAPTIONS:
            self.ids = list(self.coco.anns.keys())
        elif self.ids_based_on == self.ID_BASE.IMAGES:
            self.ids = list(self.coco.imgs.keys())
        else:
            raise ValueError("Chosen base for COCO IDs is not implemented")

        self.return_labels = False
        # set vocabulary, tokens and transform
        self.vocab = vocab
        self.tokens = tokenized_captions
        self.transform = transform

    # if we want to return labels, load them.
    def set_label_usage(self, return_labels):
        if return_labels and not hasattr(self, 'class_labels'):
            self.load_class_labels(self.labels_path)
        self.return_labels = return_labels

    def load_class_labels(self, category_path, use_supercategories=False):
        # initialize coco object
        coco = COCO(category_path)
        # map id (supercategory or int id) to label
        id_to_label = {}
        # maps images to labels
        class_labels = {}
        i = 0

        # key is a number, info is a dictionary with {'supercategory': a high-level category of the object,
        # 'id': id of the object (same as the key), 'name': name of object
        for key, info in coco.cats.items():
            # if we are using the supercategories, cat will be a supercategory. Otherwise, it will be the ID
            if use_supercategories:
                cat = info["supercategory"]
            else:
                cat = key
            # if this is the first time we encounter this ID, add it to the id_to_label dict.
            # The labels are integers starting form 0
            if cat not in id_to_label:
                id_to_label[cat] = i
                i += 1
            # get label number
            label_id = id_to_label[cat]
            # populate class_labels dictionary, mapping images to list of label ids
            for img in coco.catToImgs[key]:
                if img not in class_labels:
                    class_labels[img] = [label_id]
                elif label_id not in class_labels[img]:
                    class_labels[img].append(label_id)

        # Original comment: Add label for all images that have no label
        l = 'not labeled'
        for img in self.coco.imgs.keys():
            if img not in class_labels:
                if l not in id_to_label:
                    id_to_label[l] = i
                    i += 1
                class_labels[img] = [id_to_label[l]]

        self.class_labels = class_labels
        self.num_classes = len(id_to_label)

    # gets the transformed image features
    def get_image(self, img_id):
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.image_path, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    # returns a random class label from the possible labels of the image
    def get_class_label(self, img_id):
        img_labels = self.class_labels[img_id]
        rand_idx = np.random.randint(len(img_labels))
        class_label = torch.LongTensor([int(img_labels[rand_idx])])
        return class_label

    def __getitem__(self, index):
        """
        Returns tuple: image features, target caption, image or annotation ID (depending on base) and class label (if
        return_labels is True.
        """
        coco = self.coco
        vocab = self.vocab
        base_id = self.ids[index]

        # set annotation and image IDs
        if self.ids_based_on == self.ID_BASE.CAPTIONS:
            ann_id = base_id
            img_id = coco.anns[ann_id]['image_id']
        # if the IDs are based on the images, get only one annotation and its ID
        elif self.ids_based_on == self.ID_BASE.IMAGES:
            img_id = base_id
            img_anns = coco.imgToAnns[img_id]
            rand_idx = np.random.randint(len(img_anns))
            ann_id = img_anns[rand_idx]['id']

        # if we want the labels, obtain a random (possible) label for this image
        if self.return_labels:
            class_label = self.get_class_label(img_id)

        # tokens of current annotation
        tokens = self.tokens[ann_id]
        # image features
        image = self.get_image(img_id)

        # make target caption with starting and ending tokens
        caption = []
        caption.append(vocab(vocab.start_token))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab(vocab.end_token))
        target = torch.Tensor(caption)

        # return the tuples
        if self.return_labels:
            return image, target, base_id, class_label
        else:
            return image, target, base_id

    # the length is the number of (caption or image) IDs
    def __len__(self):
        return len(self.ids)

    def eval(self, captions, checkpoint_path, score_metric='CIDEr'):
        captions_path = checkpoint_path + "-val-captions.json"
        # write the captions to a file
        with open(captions_path, 'w') as f:
            json.dump(captions, f)
        # a result API object for evaluation
        cocoRes = self.coco.loadRes(captions_path)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        # evaluate: computes Bleu, METEOR, ROUGE_L, CIDEr and SPICE scores
        cocoEval.evaluate()
        # write the evalImgs list and the evaluation dictionary, mapping metrics to scores
        json.dump(cocoEval.evalImgs, open(checkpoint_path + "-val-metrics-imgs.json", 'w'))
        json.dump(cocoEval.eval,     open(checkpoint_path + "-val-metrics-overall.json", 'w'))

        print(cocoEval.eval.items())
        # return the score for the given metric
        return cocoEval.eval[score_metric]


    @staticmethod
    def collate_fn(data):
        """
        Original comment:
        Creates mini-batch tensors from the list of tuples (image, caption).

        We should build custom collate_fn rather than using default collate_fn,
        because merging caption (including padding) is not supported in default.
        Args:
            data: list of tuple (image, caption).
                - image: torch tensor of shape (3, 256, 256).
                - caption: torch tensor of shape (?); variable length.
        Returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
        """
        # Original comment: Sort a data list by caption length (descending order).
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions, ids, *labels = zip(*data)

        # return_labels is True if there are any labels in the data. Otherwise, it's false
        if len(labels) > 0:
            return_labels = True
            # concatenate labels[0] along the rows
            labels = torch.cat(labels[0], 0)
        else:
            return_labels = False

        # Original comment: Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Original comment: Merge captions (from tuple of 1D tensor to 2D tensor).
        # last index of each caption
        lengths = [len(cap)-1 for cap in captions]
        word_inputs = torch.zeros(len(captions), max(lengths)).long()
        word_targets = torch.zeros(len(captions), max(lengths)).long()

        for i, cap in enumerate(captions):
            # last index of the current caption
            end = lengths[i]
            # for each word_input index, its target output will be that in its corresponding position in word_targets
            word_inputs[i, :end] = cap[:-1]
            word_targets[i, :end] = cap[1:]

        if return_labels:
            return images, word_inputs, word_targets, lengths, ids, labels
        else:
            return images, word_inputs, word_targets, lengths, ids


    @classmethod
    def tokenize(cls, caption):
        # tokenizes a caption
        t = PTBTokenizer()
        return t.tokenize_caption(caption)


    @classmethod
    def build_tokenized_captions(cls, json):
        # the PTBTokenizer can be found in /utils/tokenizer/ptbtokenizer.py
        # The captions are taken from the annotations in the dataset and tokenized
        coco = COCO(json)
        t = PTBTokenizer()
        tokenized_captions = t.tokenize(coco.anns)
        return tokenized_captions


    @classmethod
    def get_tokenized_captions(cls, caption_path, target_path):
        # Original comment: Load or construct tokenized captions.

        # Our comment: if there is something at target_path, we can get the captions from there,
        # else build_tokenized_captions (see above) is used to get them, and they are stored
        # at target_path
        if os.path.exists(target_path):
            with open(target_path, 'rb') as f:
                tokens = pickle.load(f)
        else:
            tokens = cls.build_tokenized_captions(caption_path)
            with open(target_path, 'wb') as f:
                pickle.dump(tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved the tokenized captions to '{}'".format(target_path))
        return tokens


    @classmethod
    def build_vocab(cls, json, tokenized_captions, threshold):
        # the vocabulary contains all words that occur above a certain threshold
        # a counter is used to count how often each type occurs in all captions
        print("Building vocabulary")
        coco = COCO(json)
        counter = Counter()
        # IDs of all captions
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            # keep track of the number of times each token appears
            tokens = tokenized_captions[id]
            counter.update(tokens)

        # Original comment: If the word frequency is less than 'threshold', then the word is discarded.
        words = [word for word, cnt in counter.items() if cnt >= threshold]

        # Original comment: Creates a vocab wrapper and add some special tokens.
        vocab = Vocabulary()

        # Original comment: Adds the words to the vocabulary.
        for word in words:
            vocab.add_word(word)

        print("Total vocabulary size: %d" %len(vocab))
        return vocab


    @classmethod
    def get_vocabulary(cls, vocab_path, captions_path, tokenized_captions, threshold=1):
        # load vocabulary if possible.
        if os.path.exists(vocab_path):
            vocab = Vocabulary.load(vocab_path)
        # if not vocabulary file exists, build the vocabulary and save it.
        else:
            vocab = cls.build_vocab(captions_path, tokenized_captions, threshold)
            Vocabulary.save(vocab, vocab_path)
            print("Saved the vocabulary to '%s'" %vocab_path)
        return vocab
