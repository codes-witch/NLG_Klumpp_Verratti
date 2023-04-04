# Main directory

TODO intro

## Files

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

### Issue alignment

### Main

### RSA