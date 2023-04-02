from collections import defaultdict

import random
import torch
import torch.nn.functional as F
from os.path import join as pjoin
from tqdm import tqdm

from models.model_loader import ModelLoader
from utils.data.data_prep import DataPreparation
from train.trainer_loader import TrainerLoader
import utils.arg_parser

from torch.distributions import Categorical

import numpy as np
import pickle

from scipy.stats import entropy

import matplotlib
import matplotlib.pyplot as plt

def get_args(argstring=None, verbose=True):
    """
    Parses the argstring and returns the arguments (an argsparse.Namespace object). If no argstring is given,
    model: gve, dataset: cub, eval: ./checkpoints/gve-cub-D2020-03-14-T18-04-04-G0-best-ckpt.pth
    """
    if argstring is None:
        argstring = "--model gve --dataset cub --eval ./checkpoints/gve-cub-D2020-03-14-T18-04-04-G0-best-ckpt.pth"
    args = utils.arg_parser.get_args(argstring)
    if verbose:
        # Print arguments
        utils.arg_parser.print_args(args)

    return args

def logsumexp(inputs, dim=None, keepdim=False):
    """
    Function to calculate logsumexp in a numerically stable way. (From the original rsa_utils.py)

    Takes input of any dimension. If no dimension over which to sum is specified, the inputs get flattened.
    One can specify whether we want to keep the original dimensions or not. If we do not, we squeeze over the
    dimension 'dim'.

    Logsumexp is used for avoiding underflow. The Softmax is its gradient.
    """
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


class BirdDistractorDataset(object):

    """
    Class representing the BirdDistractorDataset used to construct similar and distractor cells in issue-sensitive
    captioning.
    """

    def __init__(self, cell_select_strategy=None, argstring=None, return_labels=True, randomized=True):
        """
        Sets instance variables for the class. See in-line comments for explanations
        """
        # cell_select_strategy can be None or 'random'.
        # 'random' will mean that the cell list will be shuffled
        self.cell_select_strategy = cell_select_strategy

        # Parse the argstring
        self.args = args = get_args(argstring)

        # device: cuda if available; else, CPU
        self.device = device = torch.device('cuda:{}'.format(args.cuda_device) if
                                            torch.cuda.is_available() and not args.disable_cuda else 'cpu')

        # split_to_data: dictionary mapping 'train', 'val' and 'test' to their corresponding data
        self.split_to_data = self.get_train_val_test_loader(args)

        # the path to the image folder
        self.image_folder = self.split_to_data['train'].image_path

        # image_id_to_split:  dictionary mapping image IDs to the split where the image is (train, test or val).
        self.image_id_to_split = self.get_image_id_to_split()

        # load the attributes:
        # filename_to_cub_img_id and cub_img_id_to_filename are dictionaries. The names are self-explanatory
        # attribute_matrix is a 2D numpy matrix where rows are images and columns are attributes. The attribute is
        # in the image iff attribute_matrix[image, attribute] == 1

        # If not randomized, load the image-attribute matrix normally. Each image's CUB ID (starting from 0) corresponds
        # to its index in the matrix
        if not randomized:
            self.filename_to_cub_img_id, self.cub_img_id_to_filename, self.attribute_matrix = self.load_attribute_map(
                pjoin(self.image_folder, "attributes", "attribute_matrix.npy"))

        # If the images are randomized, we need two further dicts: one for mapping the random index to the CUD ID
        # (obtained from a file in ./data/cub/attributes) and one for the opposite mapping. Here we populate all dicts
        else:
            self.filename_to_cub_img_id, self.cub_img_id_to_filename, \
            self.attribute_matrix, self.random_idx_to_img_id = self.load_attribute_map_randomized(
                pjoin(self.image_folder, "attributes", "randomized_attribute_matrix.npy"),
                pjoin(self.image_folder, "attributes", "random_idx_to_file_idx.json"))
            self.img_id_to_random_idx = {}
            # reverse the mapping in random_idx_to_img_id
            for k, v in self.random_idx_to_img_id.items():
                self.img_id_to_random_idx[v] = k

        # If true, the index of the rows of the attribute_matrix does not correspond with the CUD IDs.
        self.randomized = randomized

        # attr_vocab_ls: List of strings. Maps index (attribute ID from 0) to name of the attribute
        # attr_to_attid: dict from string of attribute to its ID starting from 0
        self.attr_vocab_ls, self.attr_to_attid = self.get_attribute_vocab()

        # labels_path: path to a pickled dict mapping image IDs (file names) to the class (bird species) it belongs to
        self.labels_path = pjoin(self.image_folder, "CUB_label_dict.p")

        # segments_to_attr_id: dictionary. "Segment" refers to the first part of attribute (like has_bill_shape).
        # Maps segment to a list of attribute IDs that correspond to it.
        # attr_id_to_segment does the opposite mapping
        # q_id_to_segments: List of all segments. Works as a dictionary mapping ID (idx) to segment
        self.segments_to_attr_id, self.attr_id_to_segment, self.q_id_to_segments = self.get_attr_segments()

        # sets self.return_labels to the given boolean value
        self.set_label_usage(return_labels)

    def get_attr_segments(self):
        """
        Returns a tuple
            -   segments_to_attr_id: maps segment to a list of attribute IDs that contain it.
            -   attr_id_to_segment: maps each attribute ID to the segment contained in the attribute
            -   q_id_to_segment: a list of segments used for mapping ID to the segment name.
        """
        # defaultdict(list): when a key is not in the dictionary, return an empty list
        segments_to_attr_id = defaultdict(list)
        # initialize dict
        attr_id_to_segment = {}

        # iterate over all attribute names and their index
        for i, attr_name in enumerate(self.attr_vocab_ls):
            # get segment (first part of the attribute name)
            seg_name = attr_name.split("::")[0]
            # a new index of attribute is added to the list corresponding to the segment
            segments_to_attr_id[seg_name].append(i)
            # map the attribute index to the segment name
            attr_id_to_segment[i] = seg_name

        # list of segment names. We can use it as a map. The ID of the segment (or "question") is the index and we
        # access the segment with it.
        q_id_to_segments = []
        # iterate over all segment names and append to list
        for k in segments_to_attr_id.keys():
            q_id_to_segments.append(k)

        return segments_to_attr_id, attr_id_to_segment, q_id_to_segments

    def load_attribute_map_randomized(self, attribute_matrix_path=None, random_idx_to_file_idx_path=None):
        """
        Loads the image-attribute matrix and dictionaries to access it appropriately
        Parameters:
            -   attribute_matrix_path: path to a numpy file containing the matrix
            -   random_idx_to_file_idx_path: path to a json file containing a dictionary mapping the randomized image
            index in the matrix to the image file index in the attributes/images.txt file
        Returns:
            -   filename_to_cub_img_id: dictionary
            -   cub_img_id_to_filename: dictionary
            -   attribute_matrix: 2D matrix. Rows represent images and columns represent attributes.
            matrix[img, attr] = 1 iff the image contains the attribute
            -   random_idx_to_file_idx: since the matrix rows are randominzed, a dictionary mapping the random index to
            the original CUB image ID
        """
        # initialize dicts
        filename_to_cub_img_id = {}
        cub_img_id_to_filename = {}

        # the lines have the following form:
        # index_from_1 image_id_filename
        with open(pjoin(self.image_folder, "attributes", "images.txt")) as f:
            for line in f:
                cub_img_id, filename = line.strip().split()
                # populate dictionaries: filename maps to ID from zero and vice versa
                filename_to_cub_img_id[filename] = int(cub_img_id) - 1
                cub_img_id_to_filename[int(cub_img_id) - 1] = filename

        # if a path has been given for the matrix, simply load it along with the random mapping dictionary and return
        if attribute_matrix_path is not None:
            attribute_matrix = np.load(attribute_matrix_path)
            random_idx_to_file_idx = pickle.load(open(random_idx_to_file_idx_path, 'rb'))
            return filename_to_cub_img_id, cub_img_id_to_filename, attribute_matrix, random_idx_to_file_idx

        # If no path has been given, we have to create the matrix and the random mapping

        # List of all ints from 0 to 11787, in order. Serves as the indices for all filenames
        indices = list(range(len(filename_to_cub_img_id)))
        import random
        random.seed(12)

        # shuffle the list. Now it functions as a map from original index to new random index.
        # file_idx_to_random_file_idx[9] -> 2000 means that the filename originally at 5 should now be at index 2000
        random.shuffle(indices)
        file_idx_to_random_file_idx = indices
        random_idx_to_file_idx = {}

        # iniialize a matrix of all zeros
        attribute_matrix = np.zeros((len(filename_to_cub_img_id), 312))

        # the image_attribute_labels.txt file contains information about the attributes that are present in each image.
        # each line follows the following structure:
        # <int:image_id> <int:attribute_id> <bool/int:is_present> <int:certainty_id> <str:time>
        with open(pjoin(self.image_folder, "attributes", "image_attribute_labels.txt")) as f:
            # tqdm used in order to get a progress bar
            for line in tqdm(f, total=3677856):
                # ORIGINAL COMMENT: some lines have error, we fix it
                if len(line.strip().split()) > 5:
                    tups = line.strip().split()
                    # These specific lines have errors. Hardcode solution
                    if tups[0] == '2275' or tups[0] == '9364':
                        cub_img_id = int(tups[0]) - 1
                        att_id = int(tups[1]) - 1
                        is_present = int(tups[2])
                        random_cub_img_id = file_idx_to_random_file_idx[cub_img_id]
                        random_idx_to_file_idx[random_cub_img_id] = cub_img_id
                        attribute_matrix[random_cub_img_id, att_id] = is_present
                        continue

                # get information from split line
                cub_img_id, att_id, is_present, certainty, time = line.strip().split()

                # get CUD image ID from 0
                cub_img_id = int(cub_img_id) - 1

                # Get the actual (random) position of the image
                random_cub_img_id = file_idx_to_random_file_idx[cub_img_id]
                # Record the mapping random -> original
                random_idx_to_file_idx[random_cub_img_id] = cub_img_id

                # get att_id from 0
                att_id = int(att_id) - 1
                # get 0 or 1
                is_present = int(is_present)
                # Fill the matrix at the random index for the image and the att_id with 0 or 1
                attribute_matrix[random_cub_img_id, att_id] = is_present

        return filename_to_cub_img_id, cub_img_id_to_filename, attribute_matrix, random_idx_to_file_idx

    def load_attribute_map(self, attribute_matrix_path=None):
        """
        Loads the image-attribute matrix and dictionaries to access it appropriately
        Parameters:
            -   attribute_matrix_path: path to a numpy file containing the matrix
        ______
        Returns:
            -   filename_to_cub_img_id: dictionary
            -   cub_img_id_to_filename: dictionary
            -   attribute_matrix: 2D matrix. Rows represent images and columns represent attributes.
            matrix[img, attr] = 1 iff the image contains the attribute
        """
        # Initialize dictionaries
        filename_to_cub_img_id = {}
        cub_img_id_to_filename = {}

        # This file's lines contain:
        # <cub_image_id (from 1)>    <image_id_filename>
        with open(pjoin(self.image_folder, "attributes", "images.txt")) as f:
            for line in f:
                cub_img_id, filename = line.strip().split()
                # Populate dictionaries with opposite mappings. Make indices start from 0
                filename_to_cub_img_id[filename] = int(cub_img_id) - 1
                cub_img_id_to_filename[int(cub_img_id) - 1] = filename

        # If no attribute matrix is already given, we have to build it from the image_attribute_labels.txt file
        # The lines have this structure:
        # <int:image_id> <int:attribute_id> <bool/int:is_present> <int:certainty_id> <str:time>
        if attribute_matrix_path is None:
            # Initialize an all-zero matrix to fill in
            attribute_matrix = np.zeros((len(filename_to_cub_img_id), 312))
            with open(pjoin(self.image_folder, "attributes", "image_attribute_labels.txt")) as f:
                # Use tqdm for a progress bar. The total represents the number of lines in the file (number of binary
                # attributes (312 binary attributes per image)
                for line in tqdm(f, total=3677856):
                    # ORIGINAL COMMENTS:
                    # some lines have error
                    # we fix it
                    if len(line.strip().split()) > 5:
                        tups = line.strip().split()
                        # These specific lines have errors. Hardcode solution
                        if tups[0] == '2275' or tups[0] == '9364':
                            cub_img_id = int(tups[0]) - 1
                            att_id = int(tups[1]) - 1
                            is_present = int(tups[2])
                            attribute_matrix[cub_img_id, att_id] = is_present
                            continue

                    # get information from split line
                    cub_img_id, att_id, is_present, certainty, time = line.strip().split()

                    # get IDs from 0 to easily access in matrix
                    cub_img_id = int(cub_img_id) - 1
                    att_id = int(att_id) - 1

                    # Boolean: 0 or 1
                    is_present = int(is_present)
                    # Fill in matrix with whether the attribute is in the image or not
                    attribute_matrix[cub_img_id, att_id] = is_present

        # If a matrix file was already given, simply load it
        else:
            attribute_matrix = np.load(attribute_matrix_path)

        return filename_to_cub_img_id, cub_img_id_to_filename, attribute_matrix

    def get_attribute_vocab(self):
        """
        Returns
            -   ls: a list that works as a map between attribute ID/index (from 0) to attribute name
            -   map: a dictionary mapping attribute names/labels to their index/ID
        """

        # Initialize a list of 312 strings (we have 312 binary attributes) to prevent index out of range
        ls = [""] * 312
        # Initialize dict
        map = {}

        # Open the file. It has <int:attribute_id (from 1)> <str:attribute_label>
        with open(pjoin(self.image_folder, "attributes", "attributes_labels.txt")) as f:
            for line in f:
                # split and get information
                att_id, att_text = line.strip().split()
                # make the mappings. They are opposite mappings. Make sure to set the IDs to start from 0
                ls[int(att_id) - 1] = att_text
                map[att_text] = int(att_id) - 1

        return ls, map

    def get_image_id_to_split(self):
        """
        Returns:
            -   dict: a dictionary mapping the image filename to the data split where it is
        """
        dic = {}

        for split in ['train', 'val', 'test']:
            # The image IDs are in text files called train.txt, val.txt and test.txt.
            with open(pjoin(self.image_folder, "{}.txt".format(split))) as f:
                # All lines of each file get added to the dictionary and are mapped to the corresponding split
                for line in f:
                    dic[line.strip()] = split
        return dic

    def get_image(self, img_id):
        """
        Parameter:
            -   img_id: a string ID for the image, that is, the filename. E.g. "030.Fish_Crow/Fish_Crow_0073_25977.jpg"
        _____
        Returns:
            -   a tensor with the image features (size 8192)
        """
        # get the split where the given image is
        split = self.image_id_to_split[img_id]
        # get the features
        return self.split_to_data[split].get_image(img_id)

    def get_train_val_test_loader(self, args):
        """
        Parameters:
            -   args: parsed arguments containing information for dataset, data_path, pretrained_model, batch_size and
            num_workers
        _____
        Returns:
            -   a dictionary mapping the 'train', 'val' or 'test' to the corresponding data.
        """
        split_to_data = {}
        for split in ['train', 'val', 'test']:
            # Initialize DataPreparation object
            data_prep = DataPreparation(args.dataset, args.data_path)
            # Get the dataset
            dataset, _ = data_prep.get_dataset_and_loader(split, args.pretrained_model,
                                                          batch_size=args.batch_size,
                                                          num_workers=args.num_workers)
            # map split name to dataset
            split_to_data[split] = dataset

        return split_to_data

    def get_top_k_from_cell(self, cell, max_cap_per_cell, cell_select_strategy=None):
        """
        TODO idk what exactly this is for
        """
        # If the select strategy is not random, simply get the first k from the cell
        if cell_select_strategy == None:
            return cell[:max_cap_per_cell]
        # If it is random, shuffle the cell and return the first k after shuffling
        elif cell_select_strategy == 'random':
            random.shuffle(cell)
            return cell[:max_cap_per_cell]
        # Throw exception if it's something other than None or 'random'
        else:
            raise NotImplemented

    def clone(self, b):
        """
        Returns a clone of the given numpy array
        """
        a = np.empty_like(b)
        a[:] = b
        return a

    def get_valid_segment_qs(self, img_id):
        """
        Parameters:
            -   img_id: the filename of the image
        _____
        Returns:
            -   valid_seg_ids: a list of the question/segment IDs that have an answer for this image
            -   labels: the segment labels of the segment IDs
        """

        # Get image index in the matrix
        attr_img_pos = self.filename_to_cub_img_id[img_id]
        if self.randomized:
            attr_img_pos = self.img_id_to_random_idx[attr_img_pos]

        valid_seg_ids = []
        labels = []

        # for all segment names
        for question_id in range(len(self.q_id_to_segments)):
            # get the attribute IDs that contain the segment
            focus_attr_ids = self.segments_to_attr_id[self.q_id_to_segments[question_id]]
            # access those attributes for the given image
            attr_vec = self.attribute_matrix[attr_img_pos, focus_attr_ids]

            # if any of the attributes is true, it means that this segment/question has an answer for this image
            # It's therefore a "valid question" to ask about the image.
            if sum(attr_vec) > 0:
                # Add the question ID to the valid segments
                valid_seg_ids.append(question_id)
                # add the label of the segment to the valid labels
                labels.append(self.q_id_to_segments[question_id])

        return valid_seg_ids, labels

    def get_valid_qs(self, img_id, verbose=True):
        """
        Parameters:
            -   img_id: the filename of the image
            -   verbose: boolean. If true, prints the indices and labels being saved
        _____
        Returns:
            -   valid_attr_inds: the indices of the binary attributes that are true for the given image
            -   labels: the names/labels of the valid attributes
        """
        # get the position of the image in the matrix
        attr_img_pos = self.filename_to_cub_img_id[img_id]
        if self.randomized:
            attr_img_pos = self.img_id_to_random_idx[attr_img_pos]

        # access the row for the image
        attrs = self.attribute_matrix[attr_img_pos]
        # find the attributes that are true for the image. This is a list of their matrix indices (aka their IDs)
        valid_attr_inds = attrs.nonzero()[0]

        labels = []
        # For all attribute IDs, get their label and append them to the label list
        for v_ind in valid_attr_inds:
            labels.append(self.attr_vocab_ls[v_ind])
            if verbose:
                print("qid: {}, attr: {}".format(v_ind, self.attr_vocab_ls[v_ind]))

        return valid_attr_inds, labels

    def map_img_pos_to_img_id(self, ls):
        """
        Takes a list of CUB image IDs (indices) and returns their corresponding filenames (image IDs)
        """
        return [self.cub_img_id_to_filename[cub_id] for cub_id in ls]

    def map_img_id_to_img_pos(self, ls):
        """
        Takes a list of filenames (image IDs) and returns their corresponding CUB image IDs (indices)
        """
        return [self.filename_to_cub_img_id[file_name] for file_name in ls]

    def set_label_usage(self, return_labels):
        """
        Sets the instance variable return_labels to a Boolean value.
        """
        # Load class_labels if not already available
        if return_labels and not hasattr(self, 'class_labels'):
            self.load_class_labels(self.labels_path)
        self.return_labels = return_labels

    def load_class_labels(self, class_labels_path):
        """
        Load class labels
        Parameter:
            -   class_label_path: path to a pickled dictionary mapping image IDs (filenames) to class
        """
        # get mapping from filenames to classes (in our case, bird species)
        with open(class_labels_path, 'rb') as f:
            label_dict = pickle.load(f, encoding='latin1')

        # set the number of classes as an instance variable.
        # In our case, 200 classes whose labels are numbers from 1 to 200
        self.num_classes = len(set(label_dict.values()))
        # set the mapping from filename to class as an instance variable
        self.class_labels = label_dict

    def get_class_label(self, img_id):
        """
        Gets the label of the class corresponding to img_id as a tensor
        Parameter:
            -   img_id: image filename
        _____
        Returns:
            -   class_label: a tensor for the class label of the given image. Possible values range from 0 to 199 (both
            inclusive)
        """
        class_label = torch.LongTensor([int(self.class_labels[img_id]) - 1])
        return class_label

    def get_batch(self, list_img_ids):
        """
        Parameters:
            -   list_img_ids: a list of images filenames
        _____
        Returns:
            -   images: a tensor sized [n, 8192] where n is the number of images
            -   labels: a tensor sized [n] where n is the number of images (or an empty list if return_labels is False)
        """
        # list for image-feature tensors
        images = []
        # list for class-label tensors
        labels = []

        # get features for each image ID
        for img_id in list_img_ids:
            split = self.image_id_to_split[img_id]
            images.append(self.split_to_data[split].get_image(img_id))

            # If we wish to return the labels, get the class label tensor for the given image
            if self.return_labels:
                class_label = self.get_class_label(img_id)
                labels.append(class_label)

        # Stack the tensors. Each image should be a tensor of size [1, 8192]. Stack along dimension 0
        images = torch.stack(images, 0)
        # Concatenate along dimension 0
        labels = torch.cat(labels, 0)

        return images, labels

    def get_caption_by_img_id(self, img_id, join_str=False):
        """
        Parameters:
            -   img_id: ID of the image whose captions we want to obtain
            -   join_str: True if we wish to obtain a list of full strings and not a list of lists of tokens
        """
        # Find what split the given image is in and get the dataset for that split
        split = self.image_id_to_split[img_id]
        dataset = self.split_to_data[split]

        # base_id = img_id (Commented out by us)

        # get the image annotations for the given ID
        img_anns = dataset.coco.imgToAnns[img_id]
        # get the annotation IDs
        ann_ids = [img_anns[rand_idx]['id'] for rand_idx in range(len(img_anns))]

        # obtain the annotation tokens
        tokens = [dataset.tokens[ann_id] for ann_id in ann_ids]

        if join_str:
            # get full strings from lists of tokens
            tokens = [' '.join(t) for t in tokens]

        return tokens
    """
    def get_captions_for_attribute(self, attr_ids, limit=5, print_ready=True):
        # TODO not used

        # ORIGINAL COMMENTS
        # attr_id: needs to be the "binary" attribute ID
        # retrieve images that have the attribute
        # attr_id: can be a number, OR a list of indices
        imgs = []

        for random_img_pos in range(self.attribute_matrix.shape[0]):
            if self.randomized:
                img_pos = self.random_idx_to_img_id[random_img_pos]
            else:
                img_pos = random_img_pos

            attr_vec = self.attribute_matrix[random_img_pos, attr_ids]
            if sum(attr_vec) >= 1:
                imgs.append(img_pos)

        img_cell = self.get_top_k_from_cell(imgs, max_cap_per_cell=limit)

        # ORIGINAL COMMENTS
        # get captions
        # map from img_pos to img_id (strings)
        img_ids = self.map_img_pos_to_img_id(img_cell)
        print("img_ids:", img_ids)
        img_name_to_caption = {}

        for img_id in img_ids:
            captions = self.get_caption_by_img_id(img_id, True)
            img_name_to_caption[img_id] = captions

        # we print it out
        if print_ready:
            for img_id, captions in img_name_to_caption.items():
                print('img name: {}'.format(img_id))
                for i, c in enumerate(captions):
                    print(i, ":", c)

                print()

            return

        return img_name_to_caption

    def get_captions_for_segment(self, seg_id, limit=5, print_ready=True):
        # TODO not used
        focus_attr_ids = self.segments_to_attr_id[self.q_id_to_segments[seg_id]]

        imgs = []

        for random_img_pos in range(self.attribute_matrix.shape[0]):
            if self.randomized:
                img_pos = self.random_idx_to_img_id[random_img_pos]
            else:
                img_pos = random_img_pos

            attr_vec = self.attribute_matrix[random_img_pos, focus_attr_ids]
            if sum(attr_vec) >= 1:
                imgs.append(img_pos)

        img_cell = self.get_top_k_from_cell(imgs, max_cap_per_cell=limit)

        img_ids = self.map_img_pos_to_img_id(img_cell)
        img_name_to_caption = {}

        for img_id in img_ids:
            captions = self.get_caption_by_img_id(img_id, True)
            img_name_to_caption[img_id] = captions

        if print_ready:
            for img_id, captions in img_name_to_caption.items():
                print('img name: {}'.format(img_id))
                for i, c in enumerate(captions):
                    print(i, ":", c)

                print()

            return

        return img_name_to_caption
    """
def load_model(rsa_dataset, verbose=True):

    print("Loading Model ...")
    ml = ModelLoader(rsa_dataset.args, rsa_dataset.split_to_data['train'])
    model = getattr(ml, rsa_dataset.args.model)()
    if verbose:
        print(model, '\n')
        print("Loading Model Weights...")

    if torch.cuda.is_available():
        evaluation_state_dict = torch.load(rsa_dataset.args.eval_ckpt)
    else:
        evaluation_state_dict = torch.load(rsa_dataset.args.eval_ckpt, map_location='cpu')

    model_dict = model.state_dict(full_dict=True)
    model_dict.update(evaluation_state_dict)
    model.load_state_dict(model_dict)
    model.eval()

    return model


class RSA(object):
    """
    ORIGINAL COMMENTS:
    RSA through matrix normalization
    We can compute RSA through the following steps:
    Step 1: add image prior: + log P(i) to the row
    Step 2: Column normalize
    - Pragmatic Listener L1: L1(i|c) \propto S0(c|i) P(i)
    Step 3: Multiply the full matrix by rationality parameter (0, infty), when rationality=1, no changes (similar to temperature)
    Step 4: add speaker prior: + log P(c_t|i, c_<t) (basically add the original literal matrix) (very easy)
            OR add a unconditioned speaker prior: + log P(c) (through a language model, like KenLM)
    Step 5: Row normalization
    - Pragmatic Speaker S1: S1(c|i) \propto L1(i|c) p(c), where p(c) can be S0
    The reason for additions is e^{\alpha log L1(i|c) + log p(i)}, where \alpha is rationality parameter
    """

    def __init__(self):
        # can be used to add KenLM language model
        # The "gigaword" one takes too long to load
        pass
    """
    def build_literal_matrix(self, orig_logprob, distractor_logprob):
        
        :param orig_logprob: [n_sample]
        :param distractor_logprob: [num_distractors, n_sample]
        :return: We put orig_logprob as the FIRST row
                [num_distractors+1 , n_sample]
        TODO this has no usages
        
        return torch.cat([orig_logprob.unsqueeze(0), distractor_logprob], dim=0)

    def compute_pragmatic_speaker(self, literal_matrix,
                                  rationality=1.0, speaker_prior=False, lm_logprobsf=None,
                                  return_diagnostics=False):
        
        Do the normalization over logprob matrix
        literal_matrix: [num_distractor_images+1, captions]
        :param literal_matrix: should be [I, C]  (num_images, num_captions)
                               Or [I, Vocab] (num_images, vocab_size)
        :param speaker_prior: turn on, we default to adding literal matrix
        :param speaker_prior_lm_mat: [I, Vocab] (a grammar weighting for previous tokens)
        :return:
               A re-weighted matrix [I, C/Vocab]
        TODO has no usages. All comments are ORIGINAL
        

        # step 2
        s0 = literal_matrix.clone()
        norm_const = logsumexp(literal_matrix, dim=0, keepdim=True)
        l1 = literal_matrix.clone() - norm_const
        # step 3
        l1 *= rationality
        # step 4
        if speaker_prior:
            # we add speaker prior
            # this needs to be a LM with shared vocabulary
            if lm_logprobsf is not None:
                s1 = l1 + lm_logprobsf[0]
            else:
                s1 = l1 + s0 # NOTE cost
        # step 5
        norm_const = logsumexp(s1, dim=1, keepdim=True)  # row normalization
        s1 = s1 - norm_const

        if return_diagnostics:
            return s1, l1, s0

        return s1
    """
    def compute_entropy(self, prob_mat, dim, keepdim=True):
        """
        Returns the entropy for the given probability matrix.
        Parameters:
            -    probmat: a probability matrix
            -   dim: dimesion across which the entropy will be calculated
            -   keepdim: Boolean. If True, the output has the same dimensions as the input. Otherwise, it will have one
             fewer dimension.
        """
        return -torch.sum(prob_mat * torch.exp(prob_mat), dim=dim, keepdim=keepdim)

    def compute_pragmatic_speaker_w_similarity(self, literal_matrix, num_similar_images,
                                               rationality=1.0, speaker_prior=False, lm_logprobsf=None,
                                               entropy_penalty_alpha=0.0, return_diagnostics=False):

        # The literal matrix contains the probabilities of S_0 for the next word.
        s0_mat = literal_matrix
        # TODO why do we take the index at 0?
        # TODO I think this gets the target image?
        prior = s0_mat.clone()[0]

        # since all the calculations are happening in log-space, this is the same as dividing S_0(w|i) by the
        # normalizing factor (represented here by logsumexp)
        # TODO there should be multiplication by the prior (or addition in log-space), both in S0_mat and in logsumexp.
        #  Where is it?
        l1_mat = s0_mat - logsumexp(s0_mat, dim=0, keepdim=True)


        same_cell_prob_mat = l1_mat[:num_similar_images + 1] - logsumexp(l1_mat[:num_similar_images + 1], dim=0)
        l1_qud_mat = same_cell_prob_mat.clone()

        entropy = self.compute_entropy(same_cell_prob_mat, 0, keepdim=True)  # (1, |V|)

        utility_2 = entropy

        utility_1 = logsumexp(l1_mat[:num_similar_images + 1], dim=0, keepdim=True)  # [1, |V|]

        utility = (1 - entropy_penalty_alpha) * utility_1 + entropy_penalty_alpha * utility_2

        s1 = utility * rationality

        # apply rationality
        if speaker_prior:
            if lm_logprobsf is None:
                s1 += prior
            else:
                s1 += lm_logprobsf[0]  # lm rows are all the same  # here is two rows summation

        if return_diagnostics:
            # ORIGINAL COMMENTS:
            # We return RSA-terms only; on the oustide (Debugger), we re-assemble for snapshots of computational process
            # s0, L1, u1, L1*, u2, u1+u2, s1
            # mat, vec, vec, mat, vec, vec, vec
            return s0_mat, l1_mat, utility_1, l1_qud_mat, entropy, utility_2, utility, s1 - logsumexp(s1, dim=1,
                                                                                                      keepdim=True)

        return s1 - logsumexp(s1, dim=1, keepdim=True)


class IncRSA(RSA):
    def __init__(self, model, rsa_dataset, lm_model=None):
        super().__init__()
        self.model = model
        self.rsa_dataset = rsa_dataset

        args = self.rsa_dataset.args

        trainer_creator = getattr(TrainerLoader, args.model)
        evaluator = trainer_creator(args, model, rsa_dataset.split_to_data['val'], [],
                                    None, rsa_dataset.device)
        evaluator.train = False

        self.evaluator = evaluator
        self.device = self.evaluator.device

    def sentence_decode(self, sampled_ids):
        outputs = sampled_ids
        vocab = self.evaluator.dataset.vocab


        generated_captions = []
        for out_idx in range(len(outputs)):
            sentence = []
            for w in outputs[out_idx]:
                word = vocab.get_word_from_idx(w.data.item())
                if word != vocab.end_token:
                    sentence.append(word)
                else:
                    break
            generated_captions.append(' '.join(sentence))

        return generated_captions

    # def semantic_speaker_from_features(self, image_input, labels):
    #     image_input = image_input.to(self.device)
    #
    #     sample_ids = self.model.generate_sentence(image_input, self.evaluator.start_word,
    #                                               self.evaluator.end_word, labels, labels_onehot=None,
    #                                               max_sampling_length=50, sample=False)
    #
    #     # if only one ID was passed, we have a 1-dimensional tensor which sentence_decode can't handle
    #     if len(sample_ids.shape) == 1:
    #         # make the sample_ids 2-dimensional
    #         sample_ids = sample_ids.unsqueeze(0)
    #
    #     return self.sentence_decode(sample_ids)

    def semantic_speaker(self, image_id_list=[], decode_strategy="greedy", image_input=None, labels=None):
        """
        Parameters:
            -   image_id_list: list of image IDs to get the captions of
            -   decode_strategy: can only be greedy in the current implementation
        """
        # image_id here is a string!

        if image_input is None and labels is None:
            # get the image features and their labels
            image_input, labels = self.rsa_dataset.get_batch(image_id_list)
        if decode_strategy == 'greedy':
            # write the features to the device in use
            image_input = image_input.to(self.device)
            # generate a sentence - obtains the IDs of the words
            sample_ids = self.model.generate_sentence(image_input, self.evaluator.start_word,
                                                      self.evaluator.end_word, labels, labels_onehot=None,
                                                      max_sampling_length=50, sample=False)
        else:
            raise Exception("not implemented")

        # if only one ID was passed, we have a 1-dimensional tensor which sentence_decode can't handle
        if len(sample_ids.shape) == 1:
            # make the sample_ids 2-dimensional
            sample_ids = sample_ids.unsqueeze(0)

        return self.sentence_decode(sample_ids)


    def greedy_pragmatic_speaker(self, img_id, question_id, rationality,
                                 speaker_prior, entropy_penalty_alpha,
                                 max_cap_per_cell=5, cell_select_strategy=None,
                                 no_similar=False, verbose=True, return_diagnostic=False, segment=False,
                                 subset_similarity=False):
        # collect_stats: debug mode (or eval mode); collect RSA statistics to understand internal workings
        # TODO only called in the debugger - not really used
        if max_cap_per_cell == 0:
            return self.semantic_speaker([img_id], decode_strategy="greedy")

        dis_cell, sim_cell, quality = self.rsa_dataset.get_cells_by_partition(img_id, question_id, max_cap_per_cell,
                                                                              cell_select_strategy,
                                                                              no_similar=no_similar,
                                                                              segment=segment,
                                                                              subset_similarity=subset_similarity)

        image_id_list = [img_id] + sim_cell + dis_cell
        with torch.no_grad():
            if not return_diagnostic:
                captions = self.greedy_pragmatic_speaker_free(image_id_list, len(sim_cell),
                                                              rationality, speaker_prior, entropy_penalty_alpha)
            else:
                captions, diags = self.greedy_pragmatic_speaker_free(image_id_list, len(sim_cell),
                                                                     rationality, speaker_prior, entropy_penalty_alpha,
                                                                     return_diagnostic=True)

        if return_diagnostic:
            return captions[0], quality, diags

        return captions[0], quality

    def fill_list(self, items, lists):
        # this is a pass-by-reference update
        for item, ls in zip(items, lists):
            ls.append(item)

    def greedy_pragmatic_speaker_free(self, image_id_list, num_sim, rationality,
                                      speaker_prior, entropy_penalty_alpha, lm_logprobsf=None,
                                      max_sampling_length=50, sample=False, return_diagnostic=False):
        """
        ORIGINAL COMMENTS:
            We always assume image_id_list[0] is the target image
            image_id_list[:num_sim] are the within cell
            image_id_list[num_sim:] are the distractors
        Parameters
            -   image_id_list: a list of image IDs where the first one is the target and the rest are similar or dissimilar
            images
            -   num_sim: int stating how many similar images are in the list
            -   rationality: RSA rationality hyperparameter
            -   speaker_prior: prior over images TODO not sure
            -   max_sampling_length: integer for maximum length of the caption
            -   sample: whether we will decode greedily or sample from predicted distribution
            -   return_diagnostic: whether we wish to obtain intermediate values from the RSA computation
        _____
        Returns:
            -   decoded captions for the images
            -   intermediate RSA calculations if return_diagnostic == True
        """

        # image features and class labels
        image_input, labels = self.rsa_dataset.get_batch(image_id_list)
        image_inputs = image_input.to(self.device)


        # set start and end words
        start_word = self.evaluator.start_word
        end_word = self.evaluator.end_word

        # will append one-hot encoded class labels to image features once image features are passed.
        feat_func = self.model.get_labels_append_func(labels, None)
        image_features = image_inputs

        # Pass through linear layer
        image_features = self.model.linear1(image_features)
        # Pass through ReLU
        image_features = F.relu(image_features)
        # Concatenate the one-hot class encoding
        image_features = feat_func(image_features)
        # Adds a new dimension at index 1
        # TODO why? something to do with channels and convolutional neural networks?
        image_features = image_features.unsqueeze(1)

        # get the embedding of the start word and repeat it so that each image gets the same start word embedding
        embedded_word = self.model.word_embed(start_word)
        embedded_word = embedded_word.expand(image_features.size(0), -1, -1)

        init_states = (None, None)
        lstm1_states, lstm2_states = init_states

        end_word = end_word.squeeze().expand(image_features.size(0))
        # Keep track of when we have reached the end for each caption
        reached_end = torch.zeros_like(end_word.data).byte()

        sampled_ids = []

        if return_diagnostic:
            # ORIGINAL COMMENT: their length is the time step length
            s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list, u_list, s1_list = [], [], [], [], [], [], [], []

        # ORIGINAL COMMENT: greedy loop, over time step
        i = 0
        while not reached_end.all() and i < max_sampling_length:
            # we get the loop started with the embedding of the start word. For further iterations it will be the
            # previous predicted word
            lstm1_input = embedded_word

            # First LSTM pass
            lstm1_output, lstm1_states = self.model.lstm1(lstm1_input, lstm1_states)
            # concatenate the output of the lstm with the image features once more
            lstm1_output = torch.cat((image_features, lstm1_output), 2)

            # Second LSTM pass. LSTM1 output used as input.
            lstm2_output, lstm2_states = self.model.lstm2(lstm1_output, lstm2_states)

            # get rid of size-1 dimensions and do a linear transformation
            outputs = self.model.linear2(lstm2_output.squeeze(1))

            # ORIGINAL COMMENT: all our RSA computation is in log-prob space
            log_probs = F.log_softmax(outputs, dim=-1)  # ORIGINAL COMMENT: log(softmax(x))

            # We will use the logprobs as our literal matrix for the RSA computation.
            literal_matrix = log_probs

            # If we don't need the diagnostics, simply get the pragmatic speaker's predictions. Otherwise, get the
            # diagnostics information too
            if not return_diagnostic:
                pragmatic_array = self.compute_pragmatic_speaker_w_similarity(literal_matrix, num_sim,
                                                                              rationality=rationality,
                                                                              speaker_prior=speaker_prior,
                                                                              entropy_penalty_alpha=entropy_penalty_alpha,
                                                                              lm_logprobsf=lm_logprobsf)
            else:
                s0_mat, l1_mat, utility_1, l1_qud_mat, entropy, utility_2, combined_u, pragmatic_array = self.compute_pragmatic_speaker_w_similarity(
                    literal_matrix, num_sim,
                    rationality=rationality,
                    speaker_prior=speaker_prior,
                    entropy_penalty_alpha=entropy_penalty_alpha,
                    lm_logprobsf=lm_logprobsf,
                    return_diagnostics=True)
                # update the lists with intermediate results
                self.fill_list([s0_mat, l1_mat, utility_1, l1_qud_mat, entropy, utility_2, combined_u, pragmatic_array],
                               [s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list, u_list, s1_list])

            # ORIGINAL COMMENTS:
            # pragmatic_array:
            # torch.Size([1, 3012])

            # ORIGINAL COMMENTS:
            # pragmatic array becomes the computational output,
            # but we need to repeat it for all
            # beam search / diverse beam search this part is easier to handle.
            outputs = pragmatic_array.expand(len(image_id_list), -1)  # expand along batch dimension
            # ORIGINAL COMMENT: rsa augmentation ends

            # if sample, we sample to predict. Otherwise, we just get the most probable next word (greedy)
            if sample:
                predicted, log_p = self.sample(outputs)
                # a tensor with zeros and ones for which captions are still active (have not been finished)
                active_batches = (~reached_end)
                # the log probabilities for the finished captions are multiplied by 0 so that we don't keep generating
                log_p *= active_batches.float().to(log_p.device)

            else:
                predicted = outputs.max(1)[1]

            # Update the captions that have reached the end: also those that now have predicted EOS
            reached_end = reached_end | predicted.eq(end_word).data
            # IDs of the words sampled
            sampled_ids.append(predicted.unsqueeze(1))
            # Get embeddings
            embedded_word = self.model.word_embed(predicted)
            embedded_word = embedded_word.unsqueeze(1)

            i += 1

        sampled_ids = torch.cat(sampled_ids, 1).squeeze()

        # return the decoded sentences (not word IDs but actual words) and, if required, the diagnostic information
        if return_diagnostic:
            return self.sentence_decode(sampled_ids), [s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list,
                                                       u_list, s1_list]

        return self.sentence_decode(sampled_ids)



"""
Essentially this is almost exactly the same as IncRSA
Except we add some new method, such as trace, visualize etc.
This is a stateful solution, but makes interaction easy enough.
"""
#TODO none of this is called.

"""
class IncRSADebugger(IncRSA):

    def greedy_pragmatic_speaker(self, img_id, question_id, rationality,
                                 speaker_prior, entropy_penalty_alpha,
                                 max_cap_per_cell=5, cell_select_strategy=None,
                                 no_similar=False, verbose=True, return_diagnostic=True,
                                 segment=True, subset_similarity=True):

        # we will automatically store last sentence

        return_diagnostic = True
        if max_cap_per_cell == 0:
            sent = super().greedy_pragmatic_speaker(img_id, question_id, rationality,
                                                    speaker_prior, entropy_penalty_alpha, max_cap_per_cell,
                                                    cell_select_strategy,
                                                    no_similar, verbose, return_diagnostic)
            return sent
        else:
            sent, quality, diags = super().greedy_pragmatic_speaker(img_id, question_id, rationality,
                                                                    speaker_prior, entropy_penalty_alpha,
                                                                    max_cap_per_cell, cell_select_strategy,
                                                                    no_similar, verbose, return_diagnostic,
                                                                    segment, subset_similarity)

        self.sent = sent
        self.quality = quality
        self.diags = diags
        self.question_id = question_id
        self.rationality = rationality

        return sent, quality

    def run_full_checks(self):
        # s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list, u_list, s1_list
        self.check_s0_row_stochastic(self.diags[0])
        self.check_l1_column_stochastic(self.diags[1])
        self.check_u1_sum_of_partial_prob(self.diags[2])
        self.check_l1_qud_column_normalized(self.diags[3])
        self.check_u2_entropy_correct(self.diags[4], self.diags[3])
        self.check_s1_row_normalized(self.diags[-1])

    def check_s0_row_stochastic(self, s0_mat):
        rand_time_idx = np.random.randint(len(s0_mat))
        print("S0 - The following value should be 1:", torch.exp(logsumexp(s0_mat[rand_time_idx][0])))

    def check_l1_column_stochastic(self, l1_mat):
        rand_time_idx = np.random.randint(len(l1_mat))

        print("L1 - The following value should be 1:", torch.exp(logsumexp(l1_mat[rand_time_idx][:, 0])))

    def check_u1_sum_of_partial_prob(self, u1_vec):
        # if the summed partial prob should be smaller than 1
        rand_time_idx = np.random.randint(len(u1_vec))
        rand_v_idx = np.random.randint(u1_vec[rand_time_idx].shape[1])

        for v_idx in range(u1_vec[rand_time_idx].shape[1]):
            assert torch.exp(u1_vec[rand_time_idx][0, rand_v_idx]) < 1

        print("U1 - The following value should be less than 1:", torch.exp(u1_vec[rand_time_idx][0, rand_v_idx]))

    def check_l1_qud_column_normalized(self, l1_qud_mat):
        rand_time_idx = np.random.randint(len(l1_qud_mat))
        print("L1 QuD - The following value should be 1:", torch.exp(logsumexp(l1_qud_mat[rand_time_idx][:, 0])))

    def check_u2_entropy_correct(self, u2_vec, l1_qud_mat):
        # use Scipy to compute entropy, check if it's the same
        rand_time_idx = np.random.randint(len(u2_vec))
        rand_v_idx = np.random.randint(l1_qud_mat[rand_time_idx].shape[1])

        v_prob = torch.exp(l1_qud_mat[rand_time_idx][:, rand_v_idx]).cpu().numpy()
        h = entropy(v_prob)

        print("U2 - The following two values should equal {} == {}".format(
            h, u2_vec[rand_time_idx][0, rand_v_idx]
        ))

    def check_s1_row_normalized(self, s1_mat):
        rand_time_idx = np.random.randint(len(s1_mat))
        print("S0 - The following value should be 1:", torch.exp(logsumexp(s1_mat[rand_time_idx][0])))

    def compute_rank(self, a):
        temp = a.argsort()[::-1]
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(a))

        # rank: each position corresponds to the original item (rank of each item, same order as original list)
        # temp: [smallest to largest] (order of original list)
        return ranks, temp

    def compute_stats(self, torch_ls, focus_idx):
        # return ranking/medium/mean/min/max
        a = torch_ls.squeeze().cpu().numpy()
        ranks, _ = self.compute_rank(a)

        return ranks[focus_idx], [np.mean(a), np.median(a), np.min(a), np.max(a), np.std(a)]

    def get_index_from_word(self, word):
        vocab = self.evaluator.dataset.vocab
        return vocab(word)

    def get_word_from_index(self, idx):
        vocab = self.evaluator.dataset.vocab
        return vocab.get_word_from_idx(idx)

    def stats_to_str(self, stats):
        concat_str = ""
        for tup in zip("mean/median/min/max/std".split('/'), stats):
            concat_str += "{}: {:.3f} ".format(tup[0], tup[1])

        return concat_str

    def numeric_space(self, matrix, prob_space):
        if prob_space:
            return torch.exp(matrix)
        else:
            return matrix

    def get_ranked_item_index(self, diag_list_idx, timestep, rank_indices, is_item_word=True):
        
        Used to examine "U1 word "spotted" has value -0.256 ranked 3th/3012"
        What are the words higher than "spotted"? Do they have lower entropy? We need to trace those!
        s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list, u_list, s1_lis
        :param diag_list_idx: put in 0 to len(self.diags)
        :param rank_indices: the top-k items that we want (i.e., [0,1,2,3]) or [2, 5])
        :param is_item_word: if we are looking for a word, we return word; otherwise we return index
        :return: [(word/index, value)]
        
        # we return index
        item_list = self.diags[diag_list_idx]
        item_unit = item_list[timestep]
        if diag_list_idx in [0, 1, -1]:
            values = item_unit[0, :].squeeze().cpu().numpy()
        else:
            values = item_unit.squeeze().cpu().numpy()

        ranks, temp = self.compute_rank(values)
        top_k = temp[rank_indices]

        if is_item_word:
            return [self.get_word_from_index(k) for k in top_k], values[top_k]
        else:
            return top_k, values[top_k]

    def compute_rsa_decision_path_for_word(self, timestep, focus_word=None, verbose=True, prob_space=False,
                                           return_rank=False):
        
        Conclusion: Ranking, and global stats, with negative words should satisfy ALL our debugging needs.
        We can quantify "failure" cases using the stats we collected here
        Basic requirements:
        1. In normal RSA:
            1). For S0, we print the relative ranking of word of choice in row, medium/mean/min/max/CI of the list
            2). For our focus word, in L1, we display word prob in target, and word probs in distractor, display it's ranking (in this limited group), medium/mean/min/max of the list
                -- p(i|u); if our "u" cannot even signal "i", then semantic model S0 failure!
                -- if word prob that we want is not the highest in target, then our S0 failed already (this is a "relative" measure)
                -- Assert: target w needs to be 1st
            3). For our focus word, in pre-prior S1, we compute it's row-stats again (to see if it's increased)
                -- p(u|i); without prior
                -- If target w is not the first, we know how big the gap is between w and w* (because we return max)
            4). For our focus word, in post-prior S1, we compute it's row-stats again (to see if it's increased)
                -- p(i|u)p(u|i)
                -- If target w is not the first, we know how big the gap is between w and w* (because we return max)
                -- between (3) and (4), we see the effect of prior
        2. In QuD-Entropy RSA
            1). Same as 1.1) S0
                -- word ranking; global stats
            2). Same as 1.2) L1
                -- word ranking; global stats
            3). U1 -- QuD RSA thing (summed from L1).
                -- word ranking; global stats
            4). U2/L1* -- Entropy thing (computed from L1*). Relative ranking of the word, stats of the dist.
                -- word ranking; global stats
                -- for negative word, we'll just track them seperately
            5). same as 1.3) pre-prior S1
                -- might need a magnitude comparison with prior
            6). same as 1.4) post-prior S1
        :arg timestep: it can be any word in the vocab list (doesn't have to be present in caption)
        :arg focus_word: we can investigate why this "focus_word" did not appear at timestep t!
        :arg negative_words: a list of negative words we send in, words that SHOULD NOT appear in the current caption
        :return a packaged stats for future decision making (like check what's broken)
    """
"""
        if focus_word is None:
            focus_word = self.sent.split()[timestep]

        focus_word_idx = self.get_index_from_word(focus_word)
        # 1). S0 (row)
        s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list, u_list, s1_list = self.diags
        s0_mat = self.numeric_space(s0_list[timestep], prob_space)

        s0_rank, s0_stats = self.compute_stats(s0_mat[0, :], focus_word_idx)
        if verbose:
            print('S0 word "{}" has value {} ranked {}th/{}, stats: {}'.format(focus_word, s0_mat[0, focus_word_idx],
                                                                               s0_rank, len(s0_mat[0, :]),
                                                                               self.stats_to_str(s0_stats)))

        # 2). L1 (column)
        l1_mat = self.numeric_space(l1_list[timestep], prob_space)
        l1_rank, l1_stats = self.compute_stats(l1_mat[:, focus_word_idx],
                                               0)  # here the focus_index is the "target" image p(i|u)
        if verbose:
            print('L1 word "{}" has value {} ranked {}th/{}, stats: {}'.format(focus_word, l1_mat[0, focus_word_idx],
                                                                               l1_rank, len(l1_mat[:, focus_word_idx]),
                                                                               self.stats_to_str(l1_stats)))

        # 3). U1 (would be the same as L1 in normal RSA setting)
        u1_vec = self.numeric_space(u1_list[timestep].squeeze(), prob_space)
        u1_rank, u1_stats = self.compute_stats(u1_vec, focus_word_idx)
        if verbose:
            print('U1 word "{}" has value {} ranked {}th/{}, stats: {}'.format(focus_word, u1_vec[focus_word_idx],
                                                                               u1_rank, len(u1_vec),
                                                                               self.stats_to_str(u1_stats)))

        # 4). Entropy
        ent_vec = self.numeric_space(entropy_list[timestep].squeeze(), False)
        ent_rank, ent_stats = self.compute_stats(ent_vec, focus_word_idx)
        if verbose:
            print('Entropy word "{}" has value {} ranked {}th/{}, stats: {}'.format(focus_word,
                                                                                    ent_vec[focus_word_idx],
                                                                                    ent_rank, len(ent_vec),
                                                                                    self.stats_to_str(ent_stats)))

        # 5). U2
        u2_vec = self.numeric_space(u2_list[timestep].squeeze(), False)
        u2_rank, u2_stats = self.compute_stats(u2_vec, focus_word_idx)
        if verbose:
            print('U2 word "{}" has value {} ranked {}th/{}, stats: {}'.format(focus_word, u2_vec[focus_word_idx],
                                                                               u2_rank, len(u2_vec),
                                                                               self.stats_to_str(u2_stats)))

        # 5). Alpha impact (this is pre-prior S1)
        # multiply with rationality (temperature); higher rationality means
        # higher rationality means lower influence of this
        u_vec = self.numeric_space(u_list[timestep].squeeze(), False)
        u_rank, u_stats = self.compute_stats(u_vec, focus_word_idx)
        if verbose:
            print('f(U1, U2) word "{}" has value {} ranked {}th/{}, stats: {}'.format(focus_word, u_vec[focus_word_idx],
                                                                                      u_rank, len(u_vec),
                                                                                      self.stats_to_str(u_stats)))

        # 6). Post-prior S1
        s1_vec = self.numeric_space(s1_list[timestep].squeeze(), prob_space)
        s1_rank, s1_stats = self.compute_stats(s1_vec, focus_word_idx)
        if verbose:
            print('S1 word "{}" has value {} ranked {}th/{}, stats: {}'.format(focus_word, s1_vec[focus_word_idx],
                                                                               s1_rank, len(s1_vec),
                                                                               self.stats_to_str(s1_stats)))

        if return_rank:
            # evolution of ranks through RSA computation
            # this is the "decision path"
            # S0, L1, QuD-L1 (U1), f(U1, U2), S1 ~= alpha * f(U1, U2) + prior
            # the ranks in here are all out of vocab rank

            # L1 is still the rank over vocab space
            l1_utt_rank, _ = self.compute_stats(l1_mat[0, :], focus_word_idx)

            # in terms of raw value, we can have all of them... but for now, we can skip
            return [s0_rank, l1_utt_rank, u1_rank, u2_rank, u_rank, s1_rank], []

    def set_plt_style(self):
        params = {'backend': 'pdf',
                  'axes.titlesize': 10,
                  'axes.labelsize': 10,
                  'font.size': 10,
                  'legend.fontsize': 10,
                  'xtick.labelsize': 10,
                  'ytick.labelsize': 10,
                  'font.family': 'DejaVu Serif',
                  'font.serif': 'Computer Modern',
                  }
        matplotlib.rcParams.update(params)

    def visualize_words_decision_paths_at_timestep(self, timestep, words, cmap_name=None):
        # build a line-graph of ranks
        # each line is a word at the timestep

        self.set_plt_style()

        try:
            self.diags
        except:
            print("Need to run 'greedy_pragmatic_speaker' over an image first!")
            return

        word_rank = {}
        word_final_rank = {}
        for w in words:
            r, _ = self.compute_rsa_decision_path_for_word(timestep, w, verbose=False, return_rank=True)
            word_rank[w] = -np.array(r)  # we use "negative" rank
            word_final_rank[w] = r[-1]

        # gradient = np.linspace(0, 1, 256)
        # gradient = np.vstack((gradient, gradient))

        # https://stackoverflow.com/questions/9750699/how-to-display-only-a-left-and-bottom-box-border-in-matplotlib

        plt.figure(figsize=(12, 8))
        # plt.set_cmap("RdBu")
        ax = plt.gca()
        # ax.set_prop_cycle('color', [plt.cm.get_cmap('RdBu')(i) for i in np.linspace(0, 1, len(word_rank))])
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.xticks(range(6), ['S0', 'L1', 'U1', 'U2', 'f(U1, U2)', 'S1'])
        # colors = [plt.cm.get_cmap("RdBu")(i) for i in np.linspace(0, 1, len(word_rank))]
        if cmap_name is None:
            colors = [plt.cm.get_cmap("coolwarm")(i) for i in np.linspace(0, 1, len(word_rank))]
        else:
            colors = [plt.cm.get_cmap(cmap_name)(i) for i in np.linspace(0, 1, len(word_rank))]

        for i, (w, r) in enumerate(word_rank.items()):
            plt.plot(range(6), r, marker='o', markersize=3, linewidth=1, color=colors[i], label='"{}"'.format(w))
            # ax.annotate('"{}"'.format(w), xy=(list(range(6))[-1], r[-1]), xytext=(10, 0), textcoords='offset points', va='center')

        # post y-axis modification
        start, end = ax.get_ylim()
        plt.yticks(plt.yticks()[0], [str(int(n)) + "th" for n in -plt.yticks()[0]])
        ax.set_ylim(bottom=start, top=end)

        plt.title("Timestep {}".format(timestep))
        plt.legend()

        plt.show()

        return word_final_rank

    def visualize_word_decision_path_at_timesteps(self, word, cmap_name=None):
        # each line is a time step
        # we scan through the time steps
        timesteps = range(len(self.sent.split()))

        self.set_plt_style()

        try:
            self.diags
        except:
            print("Need to run 'greedy_pragmatic_speaker' over an image first!")
            return

        timestep_rank = {}
        timestep_final_rank = {}
        for t in timesteps:
            r, _ = self.compute_rsa_decision_path_for_word(t, word, verbose=False, return_rank=True)
            timestep_rank[t] = -np.array(r)
            timestep_final_rank[t] = r[-1]

        plt.figure(figsize=(12, 8))
        # plt.set_cmap("RdBu")
        ax = plt.gca()
        # ax.set_prop_cycle('color', [plt.cm.get_cmap('RdBu')(i) for i in np.linspace(0, 1, len(word_rank))])
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.xticks(range(6), ['S0', 'L1', 'U1', 'U2', 'f(U1, U2)', 'S1'])
        # colors = [plt.cm.get_cmap("RdBu")(i) for i in np.linspace(0, 1, len(word_rank))]
        if cmap_name is None:
            colors = [plt.cm.get_cmap("coolwarm")(i) for i in np.linspace(0, 1, len(timestep_rank))]
        else:
            colors = [plt.cm.get_cmap(cmap_name)(i) for i in np.linspace(0, 1, len(timestep_rank))]

        for i, (t, r) in enumerate(timestep_rank.items()):
            plt.plot(range(6), r, marker='o', markersize=3, linewidth=1, color=colors[i], label='step {}'.format(t))
            # ax.annotate('At step {}'.format(t), xy=(list(range(6))[-1], r[-1]), xytext=(10, 0), textcoords='offset points',
            #             va='center')

        # post y-axis modification
        start, end = ax.get_ylim()
        plt.yticks(plt.yticks()[0], [str(int(n)) + "th" for n in -plt.yticks()[0]])
        ax.set_ylim(bottom=start, top=end)

        plt.title('Word "{}"'.format(word))
        plt.legend()

        plt.show()

        return timestep_final_rank


if __name__ == '__main__':

    rsa_dataset = BirdDistractorDataset()
    # rsa_dataset.get_valid_qs("195.Carolina_Wren/Carolina_Wren_0029_186212.jpg")
    # rsa_dataset.filename_to_cub_img_id["195.Carolina_Wren/Carolina_Wren_0029_186212.jpg"]
    # rsa_dataset.attribute_matrix[11470, 117]
    # rsa_dataset.attribute_matrix[0, 117]
    # rsa_dataset.attribute_matrix[[0, 2, 3], 117]


    # import IPython
    #
    # IPython.embed()

    # pdb.set_trace()

    model = load_model(rsa_dataset)
    incr = IncRSA(model, rsa_dataset)
    # print(incr.sentence_decode([torch.tensor([400, 60]), torch.tensor([1]), torch.tensor([2])]))
    #print(incr.semantic_speaker(['003.Sooty_Albatross/Sooty_Albatross_0075_796352.jpg', '007.Parakeet_Auklet/Parakeet_Auklet_0047_795956.jpg']))
    print(incr.semantic_speaker(['003.Sooty_Albatross/Sooty_Albatross_0075_796352.jpg']))
"""