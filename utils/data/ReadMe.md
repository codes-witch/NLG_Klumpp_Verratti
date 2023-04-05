# Utils/data

This folder contains code to load and use the CUB and COCO datasets. This code is needed for both `main.py` 
and `evaluation.py`. All code in this subdirectory was written by the original authors. All documentation is 
ours unless explicitly stated otherwise (preceded by "Original comment(s)").

## Files

### Data Preparation

The `DataPreparation` class is used to set up the dataset and data loader. It is used whenever image data is used,
i.e. for both training and caption generation. A vision model for transformation of the data can be applied too.

### Coco Dataset

Class representing the MSCOCO dataset. The dataset includes images, captions, and the vocabulary. The 
corresponding data files must be in the specified locations. The MSCOCO dataset is neither included here nor
in the original repository and must be obtained separately.

The class contains methods to set parameters, access and return image embeddings and captions, and to 
automatically evaluate generated captions against the gold standard (using metrics such as CIDEr (see
Vedantam et al. (2015): CIDEr: Consensus-based Image Description Evaluation).

### Cub Dataset

Class representing the CUB bird dataset. It inherites from `CocoDataset`, so to use the CUB data, both 
`cub_dataset.py` and `coco_dataset.py` have to be present. Additionally, the CUB data files have to be in the
specified locations. The CUB data can be obtained from Nie et al.'s original repository.
