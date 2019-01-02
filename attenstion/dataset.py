import os
import torch
import sys
import argparse

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class MyDataset(Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __getitem__(self, index):
        get_source = self.source[index]
        get_target = self.target[index]
        return [get_source, get_target]

    def __len__(self):
        return len(self.source)

    def collater(self, items):
        source_items = [item[0] for item in items]
        target_items = [item[1] for item in items]
        source_padding = pad_sequence(source_items, batch_first=True)
        target_padding = pad_sequence(target_items, batch_first=True)
        return [source_padding, target_padding]

class EvaluateDataset(Dataset):
    def __init__(self, source):
        self.source = source

    def __getitem__(self, index):
        get_source = self.source[index]
        return [get_source]

    def __len__(self):
        return len(self.source)

    def collater(self, items):
        source_items = [item[0] for item in items]
        source_padding = pad_sequence(source_items, batch_first=True)
        return [source_padding]
