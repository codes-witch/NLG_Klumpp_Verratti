def get_split_str(train, test=False, dataset=None):
    """
    Get the name of the data split
    """
    # if we are training, the split is "train"
    if train:
        return 'train'
    # the coco dataset does not have a test set
    if test and dataset != 'coco':
        return 'test'
    # if it's not "train" or "test", set to validation.
    return 'val'
