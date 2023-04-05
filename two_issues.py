import os
from os.path import join as pjoin
from tqdm import tqdm
import json

from rsa_eval import KeywordClassifier
from rsa_eval import load_inc_rsa_model
from rsa import BirdDistractorDataset
from evaluation import CUBPartitionDataset

def generate_caption_for_test_two_issues(save_file_prefix, max_cap_per_cell=400, rationality=20, # max capacity is higher
                                         entropy_penalty_alpha=0.4, no_retry=False, no_similar=False,
                                         distractors="wide", s_avg=False):
    """
    Modified from generate_caption_for_test() in evaluation.py

    Generates pragmatic speaker captions for two issues for the images of the test set.
    Generates Json files with dictionaries

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
    with open(pjoin(cub_partition.image_folder, 'test.txt')) as f:
        for line in f:
            test_ids.append(line.strip())

    # Initialize dictionaries
    img_id_to_caption = {}
    img_id_to_partition_idx = {}

    for imgid in tqdm(test_ids):

        # Resolvable issues for the current image
        img_issues, issue_names = cub_partition.get_valid_issues(imgid)

        # A dict with two nested embedded dictionaries. Mapping: image ID -> first issue ID -> second issue ID -> caption
        img_id_to_caption[imgid] = {}

        # A dict with two nested embedded dictionaries.
        # Mapping: image ID -> first issue ID -> second issue ID -> Partition as a list of lists: [dissimilar img indices, similar img indices]
        img_id_to_partition_idx[imgid] = {}

        # Populating the dictionaries:
        # the outer loop goes through all resolvable issues (i.e., as the first issue)
        for issue_id_1, issue_name_1 in zip(img_issues, issue_names):
            # Get the cells for the first partition
            dis_cell2_1, sim_cell2_1, _, dis_indices_1, sim_indices_1 = cub_partition.get_cells_by_partition(
                imgid, issue_id_1, max_cap_per_cell=max_cap_per_cell)

            if no_similar:
                sim_cell2_1 = []

            # initialize the embedded dictionaries for the first issue
            img_id_to_caption[imgid][issue_id_1] = {}
            img_id_to_partition_idx[imgid][issue_id_1] = {}

            # get the index of the first issue
            index = img_issues.index(issue_id_1)

            # range through all resolvable issues after the first issue (as the second issue)
            for issue_id_2, issue_name_2 in zip(img_issues[index+1:], issue_names[index+1:]): # we only need the issues that come after the current
                # Get the cells for the second partition
                dis_cell2_2, sim_cell2_2, _, dis_indices_2, sim_indices_2 = cub_partition.get_cells_by_partition(
                    imgid, issue_id_2, max_cap_per_cell=max_cap_per_cell)

                if no_similar:
                    sim_cell2_2 = []

                # the combined sim cell is the intersection
                sim_cell2_comb = list(set(sim_cell2_1) & set(sim_cell2_2))
                sim_indices_comb = list(set(sim_indices_1) & set(sim_indices_2))
                # in narrow condition, get the intersection of distractor cells
                if distractors == "narrow":
                    dis_cell2_comb = list(set(dis_cell2_1) & set(dis_cell2_2))
                    dis_indices_comb = list(set(dis_indices_1) & set(dis_indices_2))
                # in wide condition, get the combined distractor cell from a union of intersections
                else:
                    a = list(set(dis_cell2_1) & set(dis_cell2_2))
                    b = list(set(dis_cell2_1) & set(sim_cell2_2))
                    c = list(set(sim_cell2_1) & set(dis_cell2_2))
                    dis_cell2_comb = a+b+c
                    a_ind = list(set(dis_indices_1) & set(dis_indices_2))
                    b_ind = list(set(dis_indices_1) & set(sim_indices_2))
                    c_ind = list(set(sim_indices_1) & set(dis_indices_2))
                    dis_indices_comb = a_ind+b_ind+c_ind

                if s_avg:
                    # just as with one issue, we can imagine several ways to calculate the average:

                    # in the commented-out code below, we average all features of similar images and call semantic speaker
                    # with the result.

                    # img_features, labels = rsa_dataset.get_batch(sim_cell2_comb)
                    # _, label = rsa_dataset.get_batch([imgid])
                    # avg = img_features.mean(dim=0, keepdim=True)
                    # cap = rsa_model.semantic_speaker(image_input=avg, labels=label)[0]

                    # here, we pass all similar images to the semantic speaker together.
                    cap = rsa_model.semantic_speaker(sim_cell2_comb)[0]

                    # for differences between the two options above, see report for discussion.

                else:
                    # Get the caption for current issues for current image. This handles all types of pragmatic speaker:
                    # S1 is generated when no_similar = True (information of whether i is in the same cell as i' is irrelevant)
                    # S1_C is generated if no_similar = False and entropy_penalty_alpha is 0 (U_1 * 1 + U_2 * 0 = U_1)
                    # S1_C+H is generated if no_similar = False and entropy_penalty_alpha is different from 0
                    cap = rsa_model.greedy_pragmatic_speaker_free([imgid] + sim_cell2_comb + dis_cell2_comb,
                                                              num_sim=len(sim_cell2_comb), rationality=rationality,
                                                              speaker_prior=True,
                                                              entropy_penalty_alpha=entropy_penalty_alpha)[0]

                # Save caption and partition in corresponding dictionaries
                img_id_to_caption[imgid][issue_id_1][issue_id_2] = cap
                img_id_to_partition_idx[imgid][issue_id_1][issue_id_2] = [dis_indices_comb, sim_indices_comb]

    # Save the populated dictionaries in respective Json files
    json.dump(img_id_to_caption, open(save_file_prefix + "_gen_captions.json", 'w'))
    json.dump(img_id_to_partition_idx, open(save_file_prefix + "_sampled_partitions.json", 'w'))

def generate_literal_caption_for_test_two_issues(save_file_prefix):
    """
        Modified from generate_literal_caption_for_test() in evaluation.py. This works exactly identical,
        except that it returns a dictionary of nested dictionaries instead of a dictionary of simple dictionaries.

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
    with open(pjoin(cub_partition.image_folder, 'test.txt')) as f:
        for line in f:
            test_ids.append(line.strip())
    # Initialize the dictionary. It will be a dict with two nested embedded dicts.
    # Mapping: image ID -> first issue ID -> second issue ID -> caption
    img_id_to_caption = {}

    # Populate the dictionary
    for imgid in tqdm(test_ids):
        img_id_to_caption[imgid] = {}
        img_issues, issue_names = cub_partition.get_valid_issues(imgid)

        # get the caption from the semantic speaker
        cap = rsa_model.semantic_speaker([imgid])[0]

        # Since this is a semantic speaker, the caption is the same for all pairs of issues - it is not issue-sensitive
        for issue_id_1, issue_name_1 in zip(img_issues, issue_names):
            img_id_to_caption[imgid][issue_id_1] = {}
            index = img_issues.index(issue_id_1) # the index of the current issue in the list of resolvable issues

            for issue_id_2, issue_name_2 in zip(img_issues[index+1:], issue_names[index+1:]):
                img_id_to_caption[imgid][issue_id_1][issue_id_2] = cap

    # Once populated, save the dictionary in a JSON file
    json.dump(img_id_to_caption, open(save_file_prefix + "_gen_captions.json", 'w'))

def generate_captions(root_dir="./results/", id="S0", condition="S0", distractors="wide", rationality=10, entropy=0.4,
                      max_cell_size=400):
    """
    This is based on the main section of evaluation.py. Calling it generates the captions for the chosen condition.
    """

    # specify directory for storing the generated captions
    os.makedirs(pjoin(root_dir, "random_run_{}".format(id), "test.txt"), exist_ok=True)
    save_dir = pjoin(root_dir, "random_run_{}".format(id))

    # Generate literal speaker captions
    if condition=="S0":
        generate_literal_caption_for_test_two_issues(save_dir + "/S0")

    # Generate pragmatic speaker captions (insensitive to issues)
    if condition=="S1":
        generate_caption_for_test_two_issues(save_dir + "/S1", max_cap_per_cell=max_cell_size, rationality=rationality,
                                             entropy_penalty_alpha=0, no_similar=True, distractors="narrow")

    # Generate issue-sensitive pragmatic speaker captions (S1_C)
    if condition=="S1_C":
        generate_caption_for_test_two_issues(save_dir + "/S1_C_" + distractors, max_cap_per_cell=max_cell_size,
                                             rationality=rationality, entropy_penalty_alpha=0, distractors=distractors)

    # Generate issue-sensitive pragmatic speaker captions with penalization for misleading captions (S1_C+H)
    if condition=="S1_CH":
        generate_caption_for_test_two_issues(save_dir + "/S1_CH_" + distractors, max_cap_per_cell=max_cell_size,
                                             rationality=rationality, entropy_penalty_alpha=entropy,
                                             distractors=distractors)

    # Generate captions with S0_AVG
    if condition=="S0_Avg":
        generate_caption_for_test_two_issues(save_dir + "/S0_Avg", max_cap_per_cell=max_cell_size, s_avg=True)

def evaluate_captions(path):
    """
    Modified from evaluate_captions() in issue_alignment.py

    Method to evaluate issue alignment for all captions in a file

    This weights all image-issue-issue triples equally, in contrast to evaluate_captions_by_image().
    """

    # instantiate the keyword classifier
    rsa_dataset = BirdDistractorDataset(randomized=True)
    kc = KeywordClassifier(rsa_dataset)

    # all issues to be looked at
    issues = [('wing','pattern'),('belly','pattern'),('breast','pattern'),('nape','color'),('upper_tail','color'),('under_tail','color'),('back','color'),('leg','color'),('throat','color'),('crown','color'),('bill','shape'),('eye','color'),('wing','color'),('bill','color'),('breast','color'),('belly','color'),('bill','length')]

    sum_full_true_pos = 0  # both issues are addressed
    sum_part_true_pos = 0  # only one issue is addressed
    sum_false_neg = 0  # neither is addressed
    sum_false_pos = 0  # number of other issues that are addressed

    # load captions
    with open(path) as input_file:
        captions = json.loads(input_file.read())

    # go through all images
    for img in captions:
        # go through all possible issues for that image
        for issue_1 in captions[img]:
            # go through all captions for that image and issue
            for issue_2 in captions[img][issue_1]:
                cap = captions[img][issue_1][issue_2]

                # parts and aspects can be extracted from the list of issues
                part_1 = issues[int(issue_1)][0]
                aspect_1 = issues[int(issue_1)][1]
                part_2 = issues[int(issue_2)][0]
                aspect_2 = issues[int(issue_2)][1]

                # check whether one or both issues are resolved
                if kc.classify_parts_aspect(part_1, aspect_1, cap, tokenize=True):
                    if kc.classify_parts_aspect(part_2, aspect_2, cap, tokenize=True):
                        sum_full_true_pos += 1
                    else:
                        sum_part_true_pos += 1
                else:
                    if kc.classify_parts_aspect(part_2, aspect_2, cap, tokenize=True):
                        sum_part_true_pos += 1
                    else:
                        sum_false_neg += 1

                # go through all other issues in the list that should not be resolved by the caption
                for other_i in range(17):
                    if other_i != int(issue_1) and other_i != int(issue_2):
                        other_part = issues[other_i][0]
                        other_aspect = issues[other_i][1]
                        if kc.classify_parts_aspect(other_part, other_aspect, cap, tokenize=True):
                            sum_false_pos += 1

    precision = (sum_full_true_pos * 2 + sum_part_true_pos) / (sum_full_true_pos * 2 + sum_part_true_pos + sum_false_pos)
    recall_strict = sum_full_true_pos / (sum_full_true_pos + sum_part_true_pos + sum_false_neg)
    recall_lax = (sum_full_true_pos + sum_part_true_pos) / (sum_full_true_pos + sum_part_true_pos + sum_false_neg)

    return precision, recall_strict, recall_lax

def evaluate_captions_by_image(path):
    """
    Modified from evaluate_captions_by_image() in issue_alignment.py

    Alternative evaluation method, where image-issue-issue triples are not weighted equally, but instead, precision
    and recall scores are calculated for each image and then averaged
    """

    # instantiate the keyword classifier
    rsa_dataset = BirdDistractorDataset(randomized=True)
    kc = KeywordClassifier(rsa_dataset)

    # all issues to be looked at
    issues = [('wing','pattern'),('belly','pattern'),('breast','pattern'),('nape','color'),('upper_tail','color'),('under_tail','color'),('back','color'),('leg','color'),('throat','color'),('crown','color'),('bill','shape'),('eye','color'),('wing','color'),('bill','color'),('breast','color'),('belly','color'),('bill','length')]

    sum_full_true_pos = 0  # both issues are addressed
    sum_part_true_pos = 0  # only one issue is addressed
    sum_false_neg = 0  # neither is addressed
    sum_false_pos = 0  # number of other issues that are addressed
    precisions = [] # list of precisions for each image
    recalls_strict = [] # list of strict recalls for each image
    recalls_lax = [] # list of lax recalls for each image

    # load captions
    with open(path) as input_file:
        captions = json.loads(input_file.read())

    # go through all images
    for img in captions:
        # go through all possible issues for that image
        for issue_1 in captions[img]:
            # go through all captions for that image and issue
            for issue_2 in captions[img][issue_1]:
                cap = captions[img][issue_1][issue_2]

                # parts and aspects can be extracted from the list of issues
                part_1 = issues[int(issue_1)][0]
                aspect_1 = issues[int(issue_1)][1]
                part_2 = issues[int(issue_2)][0]
                aspect_2 = issues[int(issue_2)][1]

                # check whether one or both issues are resolved
                if kc.classify_parts_aspect(part_1, aspect_1, cap, tokenize=True):
                    if kc.classify_parts_aspect(part_2, aspect_2, cap, tokenize=True):
                        sum_full_true_pos += 1
                    else:
                        sum_part_true_pos += 1
                else:
                    if kc.classify_parts_aspect(part_2, aspect_2, cap, tokenize=True):
                        sum_part_true_pos += 1
                    else:
                        sum_false_neg += 1

                # go through all other issues in the list that should not be resolved by the caption
                for other_i in range(17):
                    if other_i != int(issue_1) and other_i != int(issue_2):
                        other_part = issues[other_i][0]
                        other_aspect = issues[other_i][1]
                        if kc.classify_parts_aspect(other_part, other_aspect, cap, tokenize=True):
                            sum_false_pos += 1

        # if there is at least one resolvable issue for the image, calculate precision and recalls
        if len(captions[img]) > 0:
            # if neither of the target issues were resolved, or no issues were resolved at all,
            # the precision is set to 0
            if sum_full_true_pos+sum_part_true_pos > 0:
                precisions.append((sum_full_true_pos * 2 + sum_part_true_pos) /
                                  (sum_full_true_pos * 2 + sum_part_true_pos + sum_false_pos))
            else:
                precisions.append(0)

            recalls_strict.append(sum_full_true_pos / (sum_full_true_pos + sum_part_true_pos + sum_false_neg))
            recalls_lax.append((sum_full_true_pos + sum_part_true_pos) /
                               (sum_full_true_pos + sum_part_true_pos + sum_false_neg))

        # set all sums back to 0
        sum_full_true_pos = 0
        sum_part_true_pos = 0
        sum_false_neg = 0
        sum_false_pos = 0

    # overall precision and recall scores are the average of the scores for individual images
    precision = sum(precisions) / len(precisions)
    recall_strict = sum(recalls_strict) / len(recalls_strict)
    recall_lax = sum(recalls_lax) / len(recalls_lax)

    return precision, recall_strict, recall_lax

if __name__ == '__main__':

    """
    Running this main section executes the whole caption generation and evaluation process (for all conditions
    considered in the report, under both score calculation methods).
    """

    dir = "./results_two_issues/"

    # generate captions
    generate_captions(root_dir=dir, id="S0", condition="S0")
    generate_captions(root_dir=dir, id="S1", condition="S1", rationality=3) # same rationality setting as for one issue
    generate_captions(root_dir=dir, id="S1_Cn", condition="S1_C", distractors="narrow", rationality=10) # same rationality as for one issue
    generate_captions(root_dir=dir, id="S1_Cw", condition="S1_C", distractors="wide", rationality=10) # same rationality as for one issue
    generate_captions(root_dir=dir, id="S1_CHn", condition="S1_CH", distractors="narrow", rationality=10, entropy=0.4) # same rationality as for one issue
    generate_captions(root_dir=dir, id="S1_CHw", condition="S1_CH", distractors="wide", rationality=10, entropy=0.4) # same rationality as for one issue
    generate_captions(root_dir=dir, id="S0_Avg", condition="S0_Avg" )

    # evaluate captions for the default calculation method
    s0 = evaluate_captions("./results_two_issues/random_run_S0/S0_gen_captions.json")
    s1 = evaluate_captions("./results_two_issues/random_run_S1/S1_gen_captions.json")
    s1cn = evaluate_captions("./results_two_issues/random_run_S1_Cn/S1_C_narrow_gen_captions.json")
    s1cw = evaluate_captions("./results_two_issues/random_run_S1_Cw/S1_C_wide_gen_captions.json")
    s1chn = evaluate_captions("./results_two_issues/random_run_S1_CHn/S1_CH_narrow_gen_captions.json")
    s1chw = evaluate_captions("./results_two_issues/random_run_S1_CHw/S1_CH_wide_gen_captions.json")
    s0avg = evaluate_captions("./results_two_issues/random_run_S0_Avg/S0_Avg_gen_captions.json")

    for s, s_name in zip([s0, s1, s1cn, s1cw, s1chn, s1chw, s0avg], ["S0", "S1", "S1_C, narrow", "S1_C, wide", "S1_CH, narrow", "S1_CH, wide", "S0_AVG"]):
        print(s_name)
        print("Precision: " + str(s[0]))
        print("Recall, strict: " + str(s[1]))
        print("Recall, lax: " + str(s[2]))

    # evaluate captions for each image separately and average then
    s0_alt = evaluate_captions_by_image("./results_two_issues/random_run_S0/S0_gen_captions.json")
    s1_alt = evaluate_captions_by_image("./results_two_issues/random_run_S1/S1_gen_captions.json")
    s1cn_alt = evaluate_captions_by_image("./results_two_issues/random_run_S1_Cn/S1_C_narrow_gen_captions.json")
    s1cw_alt = evaluate_captions_by_image("./results_two_issues/random_run_S1_Cw/S1_C_wide_gen_captions.json")
    s1chn_alt = evaluate_captions_by_image("./results_two_issues/random_run_S1_CHn/S1_CH_narrow_gen_captions.json")
    s1chw_alt = evaluate_captions_by_image("./results_two_issues/random_run_S1_CHw/S1_CH_wide_gen_captions.json")
    s0avg_alt = evaluate_captions_by_image("./results_two_issues/random_run_S0_Avg/S0_Avg_gen_captions.json")

    for s, s_name in zip([s0_alt, s1_alt, s1cn_alt, s1cw_alt, s1chn_alt, s1chw_alt, s0avg_alt],
                         ["S0", "S1", "S1_C, narrow", "S1_C, wide", "S1_CH, narrow", "S1_CH, wide", "S0_AVG"]):
        print(s_name)
        print("Precision: " + str(s[0]))
        print("Recall, strict: " + str(s[1]))
        print("Recall, lax: " + str(s[2]))
