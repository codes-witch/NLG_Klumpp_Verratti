import os
import random
import torch
from os.path import join as pjoin
from tqdm import tqdm
import utils.arg_parser
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image

from rsa_eval import load_inc_rsa_model
from rsa_notes import BirdDistractorDataset

def get_args(argstring=None, verbose=False):
    """
    Parse the arguments if there is an argstring. Otherwise, resort to the GVE model, CUB dataset from the checkpoint
     ./checkpoints/gve-cub-D2020-03-14-T18-04-04-G0-best-ckpt.pth

    If verbose, print all arguments
    """
    if argstring is None:
        argstring = "--model gve --dataset cub --eval ./checkpoints/gve-cub-D2020-03-14-T18-04-04-G0-best-ckpt.pth"
    args = utils.arg_parser.get_args(argstring)
    if verbose:
        utils.arg_parser.print_args(args)

    return args


class CUBPartitionDataset(object):
    """
    A class to manage partitions of the CUB dataset.

    Uses the following instance variables:
        From parameters:
        -   cell_select_strategy: No bearing in the current implementation.
        -   arguments obtained from the argstring passed. If no argstring is present, resort to the GVE model, CUB dataset
        and load from the checkpoint at ./checkpoints/gve-cub-D2020-03-14-T18-04-04-G0-best-ckpt.pth

        Others:
        -   device: CUDA if available, otherwise CPU
        -   image_folder is the CUB folder in data, where we will find all the images, matrices and pickle files we need
        -   attr_issue_matrix: Rows represent all images and columns represent the 17 most common issues.
        -   mat_idx_to_imgid: a list of image IDs (filenames). Maps the index at which they are to be found in the
        attr_issue_matrix to the image ID itself. In short, maps index -> ID
        -   imgid_to_mat_idx: a dictionary to find the index at which an image is in the attr_to_issue_matrix.
        Maps ID -> index
        -   issue_vocab: List of all the 17 issue names. Maps the indices of the matrix columns to the issue name.


    """
    def __init__(self, cell_select_strategy=None, argstring=None, return_labels=True):
        """
        Set all instance variables (see class description)
        """

        self.cell_select_strategy = cell_select_strategy

        self.args = args = get_args(argstring)
        self.device = torch.device('cuda:{}'.format(args.cuda_device) if
                                            torch.cuda.is_available() and not args.disable_cuda else 'cpu')

        self.image_folder = './data/cub/'

        # load from data files
        self.attr_issue_matrix = np.load(pjoin(self.image_folder, "strict_cap_issue_matrix.npz"))['data']
        self.mat_idx_to_imgid = json.load(open(pjoin(self.image_folder, "mat_idx_to_imgid.json")))
        self.imgid_to_mat_idx = json.load(open(pjoin(self.image_folder, "imgid_to_mat_idx.json")))
        self.issue_vocab = json.load(open(pjoin(self.image_folder, "issue_vocab.json")))


    def get_cells_by_partition(self, img_id, issue_id, max_cap_per_cell=60, cell_select_strategy=None,
                               no_similar=False):

        """
        Parameters:
            -   img_id: a string with the name of the image file. E.g. '144.Common_Tern/Common_Tern_0108_149672.jpg'
            -   issue_id: an int from 0 to 16 corresponding to the issue by which we wish to create the partition.
            -   max_cap_per_cell: an int defining how many images at most will be sampled for each cell
            -   cell_select_strategy: has no bearing on our current implementation
            -   no_similar: a boolean. If True, the cell for similar images will be an empty list

        _____

        Returns: a tuple (list, list, 0, list, list)
            -   tuple[0]: IDs for the images in the dissimilar cell
            -   tuple[1]: IDs for the images in the similar cell
            -   tuple[2]: Always 0. This seems to have no bearing in the current implementation
            -   tuple[3]: indices of the matrix row corresponding to dissimilar images
            -   tuple[4]: indices of the matrix row corresponding to similar images
        """
        # Check for validity of issue_id parameter
        assert type(issue_id) == int and 0 <= issue_id <= len(self.issue_vocab)
        # Make sure that the given issue is one of the resolvable issues for the given image
        assert issue_id in self.get_valid_issues(img_id)[0], "question_id not valid!"

        # Find the images which contain the issue and save them in sim_img_indices
        # Images that don't contain the issue are saved in dis_img_indices
        # Both are lists of the image indices
        sim_img_indices = self.attr_issue_matrix[:, issue_id].nonzero()[0].tolist()
        dis_img_indices = (self.attr_issue_matrix[:, issue_id] == 0).nonzero()[0].tolist()

        # Get some randomly-sampled indices from both similar & dissimilar issues
        sampled_sim_img_indices = random.sample(sim_img_indices, max_cap_per_cell)
        sampled_dis_img_indices = random.sample(dis_img_indices, max_cap_per_cell)

        # Get the IDs from the sampled image indices
        sim_img_ids = [self.mat_idx_to_imgid[i] for i in sampled_sim_img_indices]
        dis_img_ids = [self.mat_idx_to_imgid[i] for i in sampled_dis_img_indices]

        if no_similar:
            sim_img_ids = []

        return dis_img_ids, sim_img_ids, 0, sampled_dis_img_indices, sampled_dis_img_indices

    def get_valid_issues(self, img_id):
        """
        Parameters:
            - img_id: a string with the name of the image file. E.g. '144.Common_Tern/Common_Tern_0108_149672.jpg'

        Returns a tuple (list, list).
            -   tuple[0] is a list of ints representing the issue IDs of the issues that are resolvable for the given image
            ID
            -   tuple[1] is a list of strings with the labels (names) of the resolvable issues for the given image ID
        """

        # Get the row corresponding to the image ID from the matrix
        issue_row = self.attr_issue_matrix[self.imgid_to_mat_idx[img_id]]
        # Get list of IDs for non-zero issues in that row (resolvable issues)
        valid_issue_ids = issue_row.nonzero()[0].tolist()

        # Get all the labels of the valid issue IDs
        labels = []
        for issue_id in valid_issue_ids:
            labels.append(self.issue_vocab[issue_id])

        return valid_issue_ids, labels

    def clone(self, b):
        """
        Returns a copy of the given matrix
        """
        a = np.empty_like(b)
        a[:] = b
        return a


def generate_caption_for_test(save_file_prefix, max_cap_per_cell=40, rationality=20, entropy_penalty_alpha=0.4,
                              no_retry=False, no_similar=False, s_avg=False):

    """
    Generates pragmatic speaker captions for the images of the test set. Generates Json files with dictionaries

    Parameters:
        -   save_file_prefix: Prefix of the file where the captions will be saved
        -   max_cap_per_cell: an int defining how many images at most will be sampled for each cell
        -   rationality: RSA hyperparameter representing optimality
        -   entropy_penalty_alpha: hyperparameter from 0 to 1 weighing the importance of entropy (u2) vs. u1_C
        -   no_similar: whether no similar images are being passed. If False, We are dealing with S1_C or S1_C+H
    """

    open(save_file_prefix + "_gen_captions.json", 'w').close()
    open(save_file_prefix + "_sampled_partitions.json", 'w').close()

    # Create necessary objects
    cub_partition = CUBPartitionDataset()
    rsa_dataset = BirdDistractorDataset(randomized=True)
    # load the model
    rsa_model = load_inc_rsa_model(rsa_dataset)

    # Get all image IDs for the test set
    test_ids = []
    # TODO change test file name!
    with open(pjoin(cub_partition.image_folder, 'test_small.txt')) as f:
        for line in f:
            test_ids.append(line.strip())

    # Initialize dictionaries
    img_id_to_caption = {}
    img_id_to_partition_idx = {}

    for imgid in tqdm(test_ids):

        # Resolvable issues for the current image
        img_issues, issue_names = cub_partition.get_valid_issues(imgid)

        # A dict with an embedded dict. Mapping: image ID -> issue ID -> caption
        img_id_to_caption[imgid] = {}

        # A dict with an embedded dict.
        # Mapping: image ID -> issue ID -> Partition as a list of lists: [dissimilar img indices, similar img indices]
        img_id_to_partition_idx[imgid] = {}

        # Populating the dictionaries:
        for issue_id, issue_name in zip(img_issues, issue_names):
            # Get the cells for the partition
            dis_cell2, sim_cell2, _, dis_indices, sim_indices = cub_partition.get_cells_by_partition(
                imgid, issue_id, max_cap_per_cell=max_cap_per_cell)

            if no_similar:
                sim_cell2 = []

            # OWN CODE!
            if s_avg: # average all features of similar images and call semantic speaker with the result
                img_features, labels = rsa_dataset.get_batch(sim_cell2)
                _, label = rsa_dataset.get_batch([imgid])
                avg = img_features.mean(dim=0, keepdim=True)

                # print(avg)

                cap = rsa_model.semantic_speaker(image_input=avg, labels=label)[0]

            else:
                # Get the caption for current issue for current image. This handles all types of pragmatic speaker:
                # S1 is generated when no_similar = True (information of whether i is in the same cell as i' is irrelevant)
                # S1_C is generated if no_similar = False and entropy_penalty_alpha is 0 (U_1 * 1 + U_2 * 0 = U_1)
                # S1_C+H is generated if no_similar = False and entropy_penalty_alpha is different from 0
                cap = rsa_model.greedy_pragmatic_speaker_free(
                        [imgid] + sim_cell2 + dis_cell2,
                        num_sim=len(sim_cell2), rationality=rationality,
                        speaker_prior=True, entropy_penalty_alpha=entropy_penalty_alpha)[0]

            # Save caption and partition in corresponding dictionaries
            img_id_to_caption[imgid][issue_id] = cap
            img_id_to_partition_idx[imgid][issue_id] = [dis_indices, sim_indices]

    # Save the populated dictionaries in respective Json files
    json.dump(img_id_to_caption, open(save_file_prefix+"_gen_captions.json", 'w'))
    json.dump(img_id_to_partition_idx, open(save_file_prefix+"_sampled_partitions.json", 'w'))

def generate_literal_caption_for_test(save_file_prefix):
    """
        Generates literal speaker captions for the images of the test set. Generates Json file with a dictionary

        Parameters:
            -   save_file_prefix: Prefix of the file where the captions will be saved
    """
    # Initialize necessary objects
    cub_partition = CUBPartitionDataset()
    rsa_dataset = BirdDistractorDataset(randomized=True)
    rsa_model = load_inc_rsa_model(rsa_dataset)

    # Create file
    open(save_file_prefix + "_gen_captions.json", 'w').close()

    test_ids = []
    # TODO change test name for running late
    with open(pjoin(cub_partition.image_folder, 'test_small.txt')) as f:
        for line in f:
            test_ids.append(line.strip())

    # Initialize the dictionary. It will be a dict with an embedded dict. Mapping: image ID -> issue ID -> caption
    img_id_to_caption = {}
    # img_id_to_partition_idx = {} (Unnecessary - deleted by us)

    # Populate the dictionary.
    for imgid in tqdm(test_ids):
        img_id_to_caption[imgid] = {}
        # img_id_to_partition_idx[imgid] = {} (TODO Unnecessary - deleted by us)
        img_issues, issue_names = cub_partition.get_valid_issues(imgid)

        # get the caption from the semantic speaker
        cap = rsa_model.semantic_speaker([imgid])[0]

        # Since this is a semantic speaker, the caption is the same for all issues - it is not issue-sensitive
        for issue_id, issue_name in zip(img_issues, issue_names):
            img_id_to_caption[imgid][issue_id] = cap

    # Once populated, save the dictionary in a JSON file
    json.dump(img_id_to_caption, open(save_file_prefix + "_gen_captions.json", 'w'))

if __name__ == '__main__':
    # both S1-Q, and S1-QH get "retry"
    parser = argparse.ArgumentParser()
    parser.add_argument('--rationality', type=float, default=10, help="raitionality")
    parser.add_argument('--entropy', type=float, default=0.4, help="raitionality")
    parser.add_argument('--max_cell_size', type=int, default=40, help="cell size")

    parser.add_argument('--exp_num', type=int, help="which evaluation experiment to run; this helps parallelization")
    parser.add_argument('--root_dir', type=str, default="./results/", help="format is ./results/{}, no slash aft+er")
    parser.add_argument('--file_prefix', type=str, default="{}", help="prefix hyperparameter for the run")
    parser.add_argument('--run_time', type=int, default=4, help="format is ./results/{}, no slash after")
    args = parser.parse_args()

    time = args.exp_num

    # If one wishes to only run one experiment, set the number in the parameter (--exp_num)
    # The rationality has been hardcoded to be the one used in the paper for each experiment,
    # but it can easily be changed

    os.makedirs(pjoin(args.root_dir, "random_run_{}".format(time), "test.txt"), exist_ok=True)
    save_dir = pjoin(args.root_dir, "random_run_{}".format(time))

    if args.exp_num == 0:
        generate_literal_caption_for_test(save_dir + "/S0")

    if args.exp_num == 1:
        generate_caption_for_test(save_dir + "/S1", max_cap_per_cell=args.max_cell_size,
                                  rationality=3,
                                  entropy_penalty_alpha=0, no_similar=True)
    if args.exp_num == 2:
        generate_caption_for_test(save_dir + "/S1_Q", max_cap_per_cell=args.max_cell_size,
                                  rationality=10,
                                  entropy_penalty_alpha=0)

    if args.exp_num == 3:
        generate_caption_for_test(save_dir + "/S1_QH", max_cap_per_cell=args.max_cell_size,
                                  rationality=10,
                                  entropy_penalty_alpha=args.entropy)

    if args.exp_num == 4:
        generate_caption_for_test(save_dir + "/S0_AVG", s_avg=True)

    # If you wish to run all experiments in one run, uncomment the code below and comment out the one above. Set exp_num
    # to 5 to run all 4 experiments.

    #
    # for time in range(args.run_time):
    #
    #     args.exp_num = time
    #
    #     # Create the directory for this run
    #     os.makedirs(pjoin(args.root_dir, "random_run_{}".format(time), "test.txt"), exist_ok=True)
    #
    #     save_dir = pjoin(args.root_dir, "random_run_{}".format(time))
    #
    #     # Generate literal speaker captions
    #     if args.exp_num == 0:
    #         generate_literal_caption_for_test(save_dir + "/S0")
    #
    #     # Generate pragmatic speaker captions (insensitive to issues)
    #     if args.exp_num == 1:
    #         # manually set rationality to replicate paper
    #         args.rationality = 3
    #         generate_caption_for_test(save_dir + "/S1", max_cap_per_cell=args.max_cell_size,
    #                                           rationality=args.rationality,
    #                                           entropy_penalty_alpha=0, no_similar=True)
    #
    #     # Generate issue-sensitive pragmatic speaker captions (S1_C)
    #     if args.exp_num == 2:
    #         # manually set rationality to replicate paper
    #         args.rationality = 10
    #         generate_caption_for_test(save_dir + "/S1_C", max_cap_per_cell=args.max_cell_size,
    #                                   rationality=args.rationality,
    #                                   entropy_penalty_alpha=0)
    #
    #     # Generate issue-sensitive pragmatic speaker captions with penalization for misleading captions (S1_C+H)
    #     if args.exp_num == 3:
    #         # manually set rationality to replicate paper
    #         args.rationality = 10
    #         generate_caption_for_test(save_dir + "/S1_QH", max_cap_per_cell=args.max_cell_size,
    #                                   rationality=args.rationality,
    #                                   entropy_penalty_alpha=args.entropy)
    #
    #     # Generate captions with S0_AVG
    #     if args.exp_num == 4:
    #         generate_caption_for_test(save_dir + "/S0_AVG", s_avg=True)
