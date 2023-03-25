"""
This file contains the code we used for evaluating issue alignment. It is not based on anything from Nie et al.'s
repository. Running the main method prints precision and recall for all conditions we evaluated (given that all the
files are in the right location).

"""

import json

from evaluation import BirdDistractorDataset
from rsa_eval import KeywordClassifier

# evaluate issue alignment for all captions in a file
def evaluate_captions(path):

    # instantiate the keyword classifier
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

    # return a tuple (precision, recall)
    return precision, recall

if __name__ == '__main__':

    # get scores for all conditions
    s0 = evaluate_captions("./results/random_run_0/S0_gen_captions.json")
    s1 = evaluate_captions("./results/random_run_1/S1_gen_captions.json")
    s1c = evaluate_captions("./results/random_run_2/S1_C_gen_captions.json")
    s1ch = evaluate_captions("./results/random_run_3/S1_CH_wide_gen_captions.json")

    # print scores
    for s, s_name in zip([s0, s1, s1c, s1ch], ["S0", "S1", "S1_C", "S1_CH"]):
        print(s_name)
        print("Precision: " + str(s[0]))
        print("Recall, strict: " + str(s[1]))
