import os
import torch
import sys
import argparse

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

PADDING = 0
UNK = 1
BOS = 2
EOS = 3

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

class EvalDataset(Dataset):
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


class Word_Data():
    def __init__(self, source_path, target_path, source_file, target_file):
        self.source_file = source_file
        self.target_file = target_file
        self.source_dict = {"[UNK]": UNK, "[BOS]": BOS, "[EOS]": EOS}
        self.target_dict = {"[UNK]": UNK, "[BOS]": BOS, "[EOS]": EOS}

        self.source_path = source_path
        self.target_path = target_path

    def getVocabSize(self, flag):
        # flag -> 1 source | flag -> 0 target
        if flag:
            self.pushVocab(self.source_dict ,self.source_file)
            return len(self.source_dict) + 1
        else:
            self.pushVocab(self.target_dict ,self.target_file)
            return len(self.target_dict) + 1

    def save(self, source_file , target_file):
        self.pushVocab(self.source_dict, self.source_file)
        self.pushVocab(self.target_dict, self.target_file)
        self.SaveTensorData(self.source_dict, self.source_path, source_file, 1)
        self.SaveTensorData(self.target_dict, self.target_path, target_file, 0)

    def pushVocab(self, langauge_dict, file_name):
        with open(file_name) as f:
            for count, vocab in enumerate(f):
                langauge_dict[vocab.strip()] = len(langauge_dict) + 1

    def SaveTensorData(self, langauge_dict, path, file_name, source_flag):
        tensor_data = self.GetTensorData(langauge_dict, path, source_flag)
        torch.save(tensor_data, file_name)

    def GetTensorData(self, langauge_dict, file_path, source_flag):
        with open(file_path) as f:
            tensor_data = [ self.ConvertTensor(langauge_dict, doc, source_flag) for i, doc in enumerate(f)]
        return tensor_data

    def ConvertTensor(self, langauge_dict, doc, source_flag):
        doc = self.replaceWord(doc)
        words = self.DocToWord(doc)
        if source_flag:
            words = ["[BOS]"] + words + ["[EOS]"]
        words_id = self.SentenceToDictID(langauge_dict, words)
        return words_id

    def replaceWord(self, doc):
        doc = doc.replace("<t>", "")
        doc = doc.replace("</t>", "")
        return doc

    def DocToWord(self, strs):
        return strs.strip().split(' ')

    def SentenceToDictID(self, langauge_dict, sentence):
        slist = []
        for word in sentence:
            if word in langauge_dict:
                slist.append(langauge_dict[word])
            else:
                slist.append(UNK)
        return torch.tensor(slist)
