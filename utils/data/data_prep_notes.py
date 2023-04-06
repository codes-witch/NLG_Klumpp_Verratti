# Original Comment: Python packages
import os

# Original Comment: Third party packages
import torch
import torchvision.transforms as transforms

# Original Comment: Local packages
from .coco_dataset_notes import CocoDataset
from .cub_dataset_notes import CubDataset
from ..transform import get_transform

class DataPreparation:
    
    """
    Class used to access a specified dataset
    """
    
    def __init__(self, dataset_name='coco', data_path='./data'):
        # specify which dataset to use (default is COCO)
        if dataset_name == 'coco':
            self.DatasetClass = CocoDataset
        elif dataset_name == 'cub':
            self.DatasetClass = CubDataset
        self.data_path = os.path.join(data_path, self.DatasetClass.dataset_prefix)

    def get_dataset(self, split='train', vision_model=None, vocab=None,
            tokens=None):
        """
        This gets the data, using either coco_dataset.py or cub_dataset.py
        """

        #  get transform, which is dependent on the vision model and whether we are in training
        transform = get_transform(vision_model, split)
        # get the dataset itself and apply transform
        dataset = self.DatasetClass(root=self.data_path,
                                    split=split,
                                    vocab=vocab,
                                    tokenized_captions=tokens,
                                    transform=transform)
        self.dataset = dataset
        return self.dataset

    def get_loader(self, dataset, batch_size=128, num_workers=4):
        """
        Get a data loader for the chosen dataset

        """
        
        # the dataset used for this method must correspond to the one specified for the DataPreparation instance
        assert isinstance(dataset, self.DatasetClass)
        
        # shuffling is done in training only
        if dataset.split == 'train':
            shuffle = True
        else:
            shuffle = False

        # get the data loader from torch.utils
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  collate_fn=dataset.collate_fn)
        return data_loader


    def get_dataset_and_loader(self, split='train', vision_model=None,
            vocab=None, tokens=None, batch_size=128, num_workers=4):
        """
        Get dataset and data loader, using get_dataset() and get_loader() (Details see above)
        """

        dataset = self.get_dataset(split, vision_model, vocab, tokens)
        loader = self.get_loader(dataset, batch_size, num_workers)
        return dataset, loader
