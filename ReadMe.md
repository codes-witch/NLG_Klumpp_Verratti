# Main directory

The code for the full training, sentence generation and evaluation can be found here. 

## Setup
Before running, download the CUB  data and the best training checkpoint using the `rsa-file-setup.sh` file. For the COCO
data, please follow this link: TODO

## Files

Most code has been written by the original authors. The code we have written can be found in the files `two_issues.py` 
and `issue_alignment.py`. We have also extended the `evaluation_notes.py` file to generate captions for the literal speaker
S0_AVG, which is mentioned in the paper but not implemented in the original repository. In said file, we have included
two alternative ways of calculating S0_AVG: one is more true to the theoretical description of S0_AVG in the original 
paper and the other one yields results that are closer to those reported by Nie et al. The latter is the one that 
remains uncommented. For further discussion, refer to the report.  

### Evaluation

`evaluation_notes.py` contains the code for generating sentences according to the different speaker agents defined in Nie et 
al. 

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
an issue has been resolved. It does so by first locating where certain body parts are mentioned in a caption

### Issue alignment

### Main

### RSA