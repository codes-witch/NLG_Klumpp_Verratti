"""
This file contains the code we used for evaluating issue alignment. It is not based on anything from Nie et al.'s
repository. Running the main method prints precision and recall for all conditions we evaluated (given that all the
files are in the right location).

"""

import json

from evaluation_notes import BirdDistractorDataset
from rsa_eval import KeywordClassifier


def evaluate_captions(path):
    """
    Method to evaluate issue alignment for all captions in a file
    
    This weights all image-issue pairs equally, in contrast to evaluate_captions_by_image().

    """

    # instantiate the dataset and keyword classifier
    rsa_dataset = BirdDistractorDataset(randomized=True)
    kc = KeywordClassifier(rsa_dataset)

    # all issues to be looked at
    issues = [('wing', 'pattern'), ('belly', 'pattern'), ('breast', 'pattern'), ('nape', 'color'),
                ('upper_tail', 'color'), ('under_tail', 'color'), ('back', 'color'), ('leg', 'color'),
                ('throat', 'color'), ('crown', 'color'), ('bill', 'shape'), ('eye', 'color'), ('wing', 'color'),
                ('bill', 'color'), ('breast', 'color'), ('belly', 'color'), ('bill', 'length')]

    sum_true_pos = 0  # target issue is addressed
    sum_false_neg = 0  # target issue is not addressed
    sum_false_pos = 0  # number of other issues that are addressed

    # load captions
    with open(path) as input_file:
        captions = json.loads(input_file.read())

    # go through all images
    for img in captions:
        # go through all captions for that image and issue
        for issue in captions[img]:
            cap = captions[img][issue]

            # parts and aspects can be extracted from the list of issues
            part = issues[int(issue)][0]
            aspect = issues[int(issue)][1]

            # check whether the issue is resolved
            if kc.classify_parts_aspect(part, aspect, cap, tokenize=True):
                sum_true_pos+=1
            else:
                sum_false_neg+=1

            # go through all other issues in the list that should not be resolved by the caption
            for other_i in range(17):
                if other_i != int(issue):
                    other_part = issues[other_i][0]
                    other_aspect = issues[other_i][1]
                    if kc.classify_parts_aspect(other_part, other_aspect, cap, tokenize=True):
                        sum_false_pos += 1

    precision = sum_true_pos / (sum_true_pos + sum_false_pos)
    recall = sum_true_pos / (sum_true_pos + sum_false_neg)
    f1_score = 2*sum_true_pos / (2*sum_true_pos + sum_false_pos + sum_false_neg)

    # return a tuple (precision, recall, F1 score)
    return precision, recall, f1_score


def evaluate_captions_by_image(path):
    """
    Alternative evaluation method, where image-issue pairs are not weighted equally, but instead, precision, 
    recall, and F1 score are calculated for each image and then averaged
    """

    # instantiate the dataset and keyword classifier
    rsa_dataset = BirdDistractorDataset(randomized=True)
    kc = KeywordClassifier(rsa_dataset)

    # all issues to be looked at
    issues = [('wing', 'pattern'), ('belly', 'pattern'), ('breast', 'pattern'), ('nape', 'color'),
                ('upper_tail', 'color'), ('under_tail', 'color'), ('back', 'color'), ('leg', 'color'),
                ('throat', 'color'), ('crown', 'color'), ('bill', 'shape'), ('eye', 'color'), ('wing', 'color'),
                ('bill', 'color'), ('breast', 'color'), ('belly', 'color'), ('bill', 'length')]

    sum_true_pos = 0  # target issue is addressed
    sum_false_neg = 0  # target issue is not addressed
    sum_false_pos = 0  # number of other issues that are addressed
    precisions = [] # list of precisions for each image
    recalls = [] # list of recalls for each image
    f1_scores = [] # list of F1 scores for each image

    # load captions
    with open(path) as input_file:
        captions = json.loads(input_file.read())

    # go through all images
    for img in captions:
        # go through all captions for that image and issue
        for issue in captions[img]:
            cap = captions[img][issue]

            # parts and aspects can be extracted from the list of issues
            part = issues[int(issue)][0]
            aspect = issues[int(issue)][1]

            # check whether the issue is resolved
            if kc.classify_parts_aspect(part, aspect, cap, tokenize=True):
                sum_true_pos+=1
            else:
                sum_false_neg+=1

            # go through all other issues in the list that should not be resolved by the caption
            for other_i in range(17):
                if other_i != int(issue):
                    other_part = issues[other_i][0]
                    other_aspect = issues[other_i][1]
                    if kc.classify_parts_aspect(other_part, other_aspect, cap, tokenize=True):
                        sum_false_pos += 1

        # if there is at least one resolvable issue for the image, calculate precision, recall, and F1 score
        if len(captions[img])>0:
            # if sum_true_pos==0 and sum_false_pos!=0, the precision is 0
            # if both sum_true_pos==0 and sum_false_pos==0, we set the precision to 0 as well (although it cannot be properly calculated)
            if sum_true_pos > 0:
                precisions.append(sum_true_pos/(sum_true_pos+sum_false_pos))
            else:
                precisions.append(0)
            recalls.append(sum_true_pos/(sum_true_pos+sum_false_neg))
            f1_scores.append(2 * sum_true_pos / (2 * sum_true_pos + sum_false_pos + sum_false_neg))

        # set all sums back to 0
        sum_true_pos=0
        sum_false_neg=0
        sum_false_pos=0
    
    # overall precision, recall, and F1 score are the average of the scores values for individual images
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f1_score = sum(f1_scores) / len(f1_scores)

    # return a tuple (precision, recall, F1 score)
    return precision, recall, f1_score


if __name__ == '__main__':

    # get scores for all conditions for equal weighting of image-issue pairs
    s0 = evaluate_captions("./results/random_run_0/S0_gen_captions.json")
    s1 = evaluate_captions("./results/random_run_1/S1_gen_captions.json")
    s1c = evaluate_captions("./results/random_run_2/S1_C_gen_captions.json")
    s1ch = evaluate_captions("./results/random_run_3/S1_CH_gen_captions.json")
    s0avg = evaluate_captions("./results/random_run_4/S0_AVG_gen_captions.json")

    # print scores
    for s, s_name in zip([s0, s1, s1c, s1ch, s0avg], ["S0", "S1", "S1_C", "S1_CH", "S0_AVG"]):
        print(s_name)
        print("Precision: " + str(s[0]))
        print("Recall: " + str(s[1]))
        print("F1 Score: " + str(s[2]))
                                                      
    # alternative method of score calculation
    s0_alt = evaluate_captions_by_image("./results/random_run_0/S0_gen_captions.json")
    s1_alt = evaluate_captions_by_image("./results/random_run_1/S1_gen_captions.json")
    s1c_alt = evaluate_captions_by_image("./results/random_run_2/S1_C_gen_captions.json")
    s1ch_alt = evaluate_captions_by_image("./results/random_run_3/S1_CH_gen_captions.json")
    s0avg_alt = evaluate_captions_by_image("./results/random_run_4/S0_AVG_gen_captions.json")


    # print scores
    for s, s_name in zip([s0_alt, s1_alt, s1c_alt, s1ch_alt, s0avg_alt], ["S0", "S1", "S1_C", "S1_CH", "S0_AVG"]):
        print(s_name)
        print("Precision: " + str(s[0]))
        print("Recall: " + str(s[1]))
        print("F1 Score: " + str(s[2]))
