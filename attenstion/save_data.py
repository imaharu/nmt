import os
import torch
import sys
import argparse

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

PADDING = 0
UNK = 1
START_DECODING = 2
STOP_DECODING = 3

class MyData():
    def __init__(self, data_path, file_name):
        self.data_path = data_path
        self.file_name = file_name
        self.dict = {"[PAD]": PADDING ,"[UNK]": UNK, "[START]": START_DECODING, "[STOP]": STOP_DECODING}

    def pushVocab(self, file_name):
        with open(file_name) as f:
            for count, vocab in enumerate(f):
                self.dict[vocab.strip()] = len(self.dict) + 1

    def LoadTensorData(self, vocab_file):
        tensor_data = self.GetTensorData(self.dict, self.data_path, self.flag)
        return tensor_data

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
