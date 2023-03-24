# Python packages
import os

# Third party packages
import torch
import torchvision.transforms as transforms

# Local packages
from .coco_dataset import CocoDataset
from .cub_dataset import CubDataset
from utils.transform import get_transform

class DataPreparation:
    def __init__(self, dataset_name='coco', data_path='./data'):
        if dataset_name == 'coco':
            self.DatasetClass = CocoDataset
        elif dataset_name == 'cub':
            self.DatasetClass = CubDataset
        self.data_path = os.path.join(data_path, self.DatasetClass.dataset_prefix)

    # NOTE: This gets the data, using either coco_dataset.py or cub_dataset.py
    def get_dataset(self, split='train', vision_model=None, vocab=None,
            tokens=None):

        # NOTE TODO: what do you think the next line (get_transform) does?
        transform = get_transform(vision_model, split)
        dataset = self.DatasetClass(root=self.data_path,
                                    split=split,
                                    vocab=vocab,
                                    tokenized_captions=tokens,
                                    transform=transform)
        self.dataset = dataset
        return self.dataset

    def get_loader(self, dataset, batch_size=128, num_workers=4):
        assert isinstance(dataset, self.DatasetClass)

        # NOTE and TODO: The data is only shuffled for training (why?)
        if dataset.split == 'train':
            shuffle = True
        else:
            shuffle = False

        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  collate_fn=dataset.collate_fn)
        return data_loader


    def get_dataset_and_loader(self, split='train', vision_model=None,
            vocab=None, tokens=None, batch_size=128, num_workers=4):

        # NOTE: This is called in main to get dataset and loader, and
        # it calls get_dataset and get_loader itself. Details see there.

        dataset = self.get_dataset(split, vision_model, vocab, tokens)
        loader = self.get_loader(dataset, batch_size, num_workers)
        return dataset, loader

