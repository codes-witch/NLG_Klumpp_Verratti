# Main directory

The code for the full training, sentence generation and evaluation can be found here. 

## Setup

The following installation guide has been taken from the Salaniz repo.

1. Clone the repository
```shell
git clone https://github.com/codes-witch/NLG_Klumpp_Verratti.git
cd NLG_Klumpp_Verratti
```
2. Create conda environment
```shell
conda env create -f environment.yml
```
3. Activate environment
```shell
conda activate gve-lrcn
```

4. Download pre-trained model and data (Note: For the COCO data, please follow this link: https://cocodataset.org/#download )

TODO CHECK WHAT NEEDS TO BE DOWNLOADED

```bash
sh rsa-file-setup.sh 
```

5. Install other packages

```bash
pip install -r requirements.txt
```


## Files

Most code has been written by the original authors; however, we have considerably reduced it by deleting functions that 
were not used for our purposes. The original code is hosted in [this repository](https://github.com/windweller/Pragmatic-ISIC/).
The authors also mention that the CUB captioning model is modified from https://github.com/salaniz/pytorch-gve-lrcn.

The code we have written can be found in the files `two_issues.py` and `issue_alignment.py`. We have also extended the 
`caption_generation.py` file to generate captions for the literal speaker
S0_AVG, which is mentioned in the paper but not implemented in the original repository. In said file, we have included
two alternative ways of calculating S0_AVG: one is more true to the theoretical description of S0_AVG in the original 
paper and the other one yields results that are closer to those reported by Nie et al. The latter is the one that is 
active in the current code. For further discussion, refer to the report.  

### Caption generation

`caption_generation.py` contains the code for generating sentences according to the different speaker agents defined in Nie et 
al. The captions are generated using the CUB dataset. 

As it stands, all that is needed to produce captions for all RSA speaker agents is to run the following code:

```shell
python3 caption_generation.py --run_time 5
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
python3 caption_generation.py --exp_num <experiment_number>
```

where `<experiment_number>` refers to the number corresponding to the desired speaker agent as explained above.

Note that this file was named `evaluation.py` in the original repository.


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

`rsa.py` contains three classes: BirdDistractorDataset, IncRSA and RSA.

`BirdDistractorDataset` allows us to manage the dataset with CUB images. It loads all the data related to class labels, 
attributes, images, batches and issues and prepares it to be accessed. 

`IncrRSA` contains the code for computing the captions for the sematic and pragmatic speaker. S1_C and S1_H are calculated with the 
function `greedy_pragmatic_speaker_free` by passing different parameters for rationality and entropy penalty as the arguments.

The RSA logprobs ("pragmatic array" in the comments) used for `greedy_pragmatic_speaker` are obtained from the function `compute_pragmatic_speaker_w_similarity`
in the class `RSA` in the same file. 
``

### Two issues

`two_issues.py` contains the code used to generate and evaluate captions for two issues. The code is modified from 
`caption_generation.py` and `issue_alignment.py`. Running `two_issues.py` generates captions for all test images and all resolvable issue pairs in the CUB dataset and calculates issue alignment scores under different conditions. For a discussion of the general approach, the individual conditions, and the results, see the report.

# References
- Nie, Allen, Reuben Cohn-Gordon, and Christopher Potts. "Pragmatic issue-sensitive image captioning." arXiv preprint arXiv:2004.14451 (2020).
- Vedantam, R., Lawrence Zitnick, C., & Parikh, D. (2015). Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4566-4575).Vedantam et al. (2015): CIDEr: Consensus-based image description evaluation.