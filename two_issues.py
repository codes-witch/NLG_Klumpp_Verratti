import os
from os.path import join as pjoin
from tqdm import tqdm
import json

from rsa_eval import KeywordClassifier
from rsa_eval import load_inc_rsa_model
from rsa import BirdDistractorDataset
from evaluation import CUBPartitionDataset

# based on generate_caption_for_test in evaluation.py, modified for two issues
def generate_caption_for_test_two_issues(save_file_prefix, max_cap_per_cell=400, rationality=20, # max capacity is higher
                                         entropy_penalty_alpha=0.4, no_retry=False, no_similar=False,
                                         distractors="wide"):
    open(save_file_prefix + "_gen_captions.json", 'w').close()
    open(save_file_prefix + "_sampled_partitions.json", 'w').close()

    cub_partition = CUBPartitionDataset()
    rsa_dataset = BirdDistractorDataset(randomized=True)
    rsa_model = load_inc_rsa_model(rsa_dataset)

    test_ids = []
    with open(pjoin(cub_partition.image_folder, 'test.txt')) as f:
        for line in f:
            test_ids.append(line.strip())

    img_id_to_caption = {}
    img_id_to_partition_idx = {}

    for imgid in tqdm(test_ids):

        img_issues, issue_names = cub_partition.get_valid_issues(imgid)
        img_id_to_caption[imgid] = {}
        img_id_to_partition_idx[imgid] = {}

        for issue_id_1, issue_name_1 in zip(img_issues, issue_names):
            dis_cell2_1, sim_cell2_1, _, dis_indices_1, sim_indices_1 = cub_partition.get_cells_by_partition(
                imgid, issue_id_1, max_cap_per_cell=max_cap_per_cell)

            if no_similar:
                sim_cell2_1 = []

            img_id_to_caption[imgid][issue_id_1] = {}
            img_id_to_partition_idx[imgid][issue_id_1] = {}
            index = img_issues.index(issue_id_1) # the index of the current issue in the list of resolvable issues

            for issue_id_2, issue_name_2 in zip(img_issues[index+1:], issue_names[index+1:]): # we only need the issues that come after the current
                dis_cell2_2, sim_cell2_2, _, dis_indices_2, sim_indices_2 = cub_partition.get_cells_by_partition(
                    imgid, issue_id_2, max_cap_per_cell=max_cap_per_cell)

                if no_similar:
                    sim_cell2_2 = []

                sim_cell2_comb = list(set(sim_cell2_1) & set(sim_cell2_2))
                sim_indices_comb = list(set(sim_indices_1) & set(sim_indices_2))
                if distractors == "narrow":
                    dis_cell2_comb = list(set(dis_cell2_1) & set(dis_cell2_2))
                    dis_indices_comb = list(set(dis_indices_1) & set(dis_indices_2))
                else: # "wide" condition
                    a = list(set(dis_cell2_1) & set(dis_cell2_2))
                    b = list(set(dis_cell2_1) & set(sim_cell2_2))
                    c = list(set(sim_cell2_1) & set(dis_cell2_2))
                    dis_cell2_comb = a+b+c
                    a_ind = list(set(dis_indices_1) & set(dis_indices_2))
                    b_ind = list(set(dis_indices_1) & set(sim_indices_2))
                    c_ind = list(set(sim_indices_1) & set(dis_indices_2))
                    dis_indices_comb = a_ind+b_ind+c_ind

                cap = rsa_model.greedy_pragmatic_speaker_free([imgid] + sim_cell2_comb + dis_cell2_comb,
                                                              num_sim=len(sim_cell2_comb), rationality=rationality,
                                                              speaker_prior=True,
                                                              entropy_penalty_alpha=entropy_penalty_alpha)[0]

                img_id_to_caption[imgid][issue_id_1][issue_id_2] = cap
                img_id_to_partition_idx[imgid][issue_id_1][issue_id_2] = [dis_indices_comb, sim_indices_comb]

    json.dump(img_id_to_caption, open(save_file_prefix + "_gen_captions.json", 'w'))
    json.dump(img_id_to_partition_idx, open(save_file_prefix + "_sampled_partitions.json", 'w'))

# this is exactly the same as the original generate_literal_caption_for_test in evaluation.py,
# except that it returns a dictionary of dictionaries instead of a simple dictionary
def generate_literal_caption_for_test_two_issues(save_file_prefix):
    cub_partition = CUBPartitionDataset()
    rsa_dataset = BirdDistractorDataset(randomized=True)
    rsa_model = load_inc_rsa_model(rsa_dataset)

    open(save_file_prefix + "_gen_captions.json", 'w').close()

    test_ids = []
    with open(pjoin(cub_partition.image_folder, 'test.txt')) as f:
        for line in f:
            test_ids.append(line.strip())

    img_id_to_caption = {}
    img_id_to_partition_idx = {}

    for imgid in tqdm(test_ids):
        img_id_to_caption[imgid] = {}
        img_id_to_partition_idx[imgid] = {}
        img_issues, issue_names = cub_partition.get_valid_issues(imgid)

        cap = rsa_model.semantic_speaker([imgid])[0]
        for issue_id_1, issue_name_1 in zip(img_issues, issue_names):
            img_id_to_caption[imgid][issue_id_1] = {}
            index = img_issues.index(issue_id_1) # the index of the current issue in the list of resolvable issues

            for issue_id_2, issue_name_2 in zip(img_issues[index+1:], issue_names[index+1:]):
                img_id_to_caption[imgid][issue_id_1][issue_id_2] = cap

    json.dump(img_id_to_caption, open(save_file_prefix + "_gen_captions.json", 'w'))

# based on the main section of evaluation.py
def generate_captions(root_dir="./results/", id="S0", condition="S0", distractors="wide", rationality=10, entropy=0.4,
                      max_cell_size=400):
    os.makedirs(pjoin(root_dir, "random_run_{}".format(id), "test.txt"), exist_ok=True)

    save_dir = pjoin(root_dir, "random_run_{}".format(id))

    if condition=="S0":
        generate_literal_caption_for_test_two_issues(save_dir + "/S0")
    if condition=="S1":
        generate_caption_for_test_two_issues(save_dir + "/S1", max_cap_per_cell=max_cell_size, rationality=rationality,
                                             entropy_penalty_alpha=0, no_similar=True, distractors="narrow")
    if condition=="S1_Q":
        generate_caption_for_test_two_issues(save_dir + "/S1_Q_" + distractors, max_cap_per_cell=max_cell_size,
                                             rationality=rationality, entropy_penalty_alpha=0, distractors=distractors)
    if condition=="S1_QH":
        generate_caption_for_test_two_issues(save_dir + "/S1_QH_" + distractors, max_cap_per_cell=max_cell_size,
                                             rationality=rationality, entropy_penalty_alpha=entropy,
                                             distractors=distractors)
    return None

# evaluate issue alignment for all captions in a file
def evaluate_captions(path):

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

    return (precision, recall_strict, recall_lax)

if __name__ == '__main__':

    # Running this main section executes the whole caption generation and evaluation; however, this is not necessarily
    # recommended because it takes very long.

    dir = "./results_two_issues/"
    generate_captions(root_dir=dir, id="S0", condition="S0")
    generate_captions(root_dir=dir, id="S1", condition="S1", rationality=3) # same rationality setting as for one issue
    generate_captions(root_dir=dir, id="S1_Qn", condition="S1_Q", distractors="narrow", rationality=10) # same rationality as for one issue
    generate_captions(root_dir=dir, id="S1_Qm", condition="S1_Q", distractors="wide", rationality=10) # same rationality as for one issue
    generate_captions(root_dir=dir, id="S1_QHn", condition="S1_QH", distractors="narrow", rationality=10, entropy=0.4) # same rationality as for one issue
    generate_captions(root_dir=dir, id="S1_QHm", condition="S1_QH", distractors="wide", rationality=10, entropy=0.4) # same rationality as for one issue

    s0 = evaluate_captions("./results_two_issues/random_run_S0/S0_gen_captions.json")
    s1 = evaluate_captions("./results_two_issues/random_run_S1/S1_gen_captions.json")
    s1qn = evaluate_captions("./results_two_issues/random_run_S1_Qn/S1_Q_narrow_gen_captions.json")
    s1qm = evaluate_captions("./results_two_issues/random_run_S1_Qm/S1_Q_wide_gen_captions.json")
    s1qhn = evaluate_captions("./results_two_issues/random_run_S1_QHn/S1_QH_narrow_gen_captions.json")
    s1qhm = evaluate_captions("./results_two_issues/random_run_S1_QHm/S1_QH_wide_gen_captions.json")

    for s, s_name in zip([s0, s1, s1qn, s1qm, s1qhn, s1qhm], ["S0", "S1", "S1_Q, narrow", "S1_Q, wide", "S1_QH, narrow", "S1_QH, wide"]):
        print(s_name)
        print("Precision: " + str(s[0]))
        print("Recall, strict: " + str(s[1]))
        print("Recall, lax: " + str(s[2]))