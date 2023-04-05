# Main directory

The code for the full training, sentence generation and evaluation can be found here. 

## Setup
Before running, download the CUB  data and the best training checkpoint using the `rsa-file-setup.sh` file. For the COCO
data, please follow this link: TODO

## Files

Most code has been written by the original authors; however, we have considerably reduced it by deleting functions that 
were not used for our purposes. 

The code we have written can be found in the files `two_issues.py` and `issue_alignment.py`. We have also extended the 
`evaluation_notes.py` file to generate captions for the literal speaker
S0_AVG, which is mentioned in the paper but not implemented in the original repository. In said file, we have included
two alternative ways of calculating S0_AVG: one is more true to the theoretical description of S0_AVG in the original 
paper and the other one yields results that are closer to those reported by Nie et al. The latter is the one that is 
active in the current code. For further discussion, refer to the report.  

### Evaluation

`evaluation_notes.py` contains the code for generating sentences according to the different speaker agents defined in Nie et 
al. The captions are generated using the CUB dataset. 

As it stands, all that is needed to produce captions for all RSA speaker agents is to run the following code:

```shell
python3 evaluation_notes.py --run_time 5
```

The `run_time` argument is used in a loop that goes over the five different speaker agents:

0. Literal speaker
1. Issue-insensitive pragmatic speaker
2. Issue-sensitive pragmatic speaker (S1_C)
3. Issue-sensitive pragmatic speaker with penalization for misleading captions (S1_C+H)
4. Literal speaker with the average of the similar images (S0_AVG)

Alternatively, experiments can be run individually by uncommenting the code that is currently commented out (and 
commenting out the current code as indicated in the file). Once that is done, one can select the experiment one wishes 
to run as follows:

```shell
python3 evaluation_notes.py --exp_num <experiment_number>
```

where `<experiment_number>` refers to the number corresponding to the desired speaker agent as explained above.

### RSA eval

The `rsa_eval.py` file contains code that is used in `issue_alignment.py` and `two_issues.py` for determining whether 
an issue has been resolved. 

One can use the method `classify_parts_aspects` by passing a body part, the aspect 
used for defining the issue, and the caption as parameters; a size for the context window is optional (default: three 
words). This method will locate where the given body part is mentioned in the caption. Then, it will try to find whether
there are any words that would resolve the issue within a context window of words preceding the body part.

### Issue alignment

`issue_alignment.py` contains the code used to evaluate the issue alignment as described in the report. We included two different methods of score averaging: in one case, all image-issue pairs are weighted equally, in the other, scores are calculated for each image separately and averaged afterwards. Running `issue_alignment.py` calculates issue alignment scores (precision, recall, and F-score) for both and for all speaker agents (S0, S1, S1_C, S1_C+H, and S0_AVG) and prints them to the screen.

### Main

The code in `main.py` can be used to train and/or evaluate the models GVE (with CUB data) and LRCN (with MSCOCO data) 
and SentenceClassifier (with CUB captions). Evaluation does not refer to the RSA extension of the models, but rather to 
some automatic measures (e.g. CIDEr, Vedantam et al. (2015)).

As a default, the code is run for the LRCN model with COCO as a dataset. To run it with the GVE model, run the following code:

```shell
python3 main.py --model gve --dataset cub
```

For the sentence classifier, run:

```shell
python3 main.py --model sc --dataset cub
```

### RSA

### Two issues

`two_issues.py` contains the code used to generate and evaluate captions for two issues. The code is modified from `evaluation_notes.py` and `issue_alignment.py`. Running `two_issues.py` generates captions for all test images and all resolvable issue pairs in the CUB dataset and calculates issue alignment scores under different conditions. For a discussion of the general approach, the individual conditions, and the results, see the report.

# References
- Nie, Allen, Reuben Cohn-Gordon, and Christopher Potts. "Pragmatic issue-sensitive image captioning." arXiv preprint arXiv:2004.14451 (2020).
- Vedantam, R., Lawrence Zitnick, C., & Parikh, D. (2015). Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4566-4575).Vedantam et al. (2015): CIDEr: Consensus-based image description evaluation.