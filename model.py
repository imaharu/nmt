# hyperparameter
from define_hyperparameter import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *

class HierachicalEncoderDecoder(nn.Module):
    def __init__(self, source_size, output_size, hidden_size):
        super(HierachicalEncoderDecoder, self).__init__()
        self.encoder = Encoder(source_size, hidden_size)

class Encoder(nn.Module):
    def __init__(self, source_size, hidden_size):
        super(Encoder, self).__init__()
        self.sentence_encoder = SentenceEncoder(source_size, hidden_size)
        self.doc_encoder = DocEncoder(source_size, hidden_size)

class SentenceEncoder(nn.Module):
    def __init__(self, source_size, hidden_size):
        super(SentenceEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.source_size = source_size
        self.embed_source = nn.Embedding(source_size, hidden_size, padding_idx=0)
        self.drop_source = nn.Dropout(p=0.2)
        self.lstm_source = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, sentence_words, hx, cx):
        source_k = self.embed_source(sentence_words)
        source_k = self.drop_source(source_k)
        hx, cx = self.lstm_source(source_k, (hx, cx) )
        return hx, cx

    def initHidden(self):
        hx = torch.zeros(batch_size, self.hidden_size).cuda()
        cx = torch.zeros(batch_size, self.hidden_size).cuda()
        return hx, cx

class DocEncoder(nn.Module):
    def __init__(self, source_size, hidden_size):
        super(DocEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.source_size = source_size
        self.embed_source_doc = nn.Embedding(source_size, hidden_size, padding_idx=0)
        self.drop_source_doc = nn.Dropout(p=0.2)
        self.lstm_source_doc = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, sentence_words, hx, cx):
        source_k = self.embed_source_doc(sentence_words)
        source_k = self.drop_source_doc(source_k)
        hx, cx = self.lstm_source_doc(source_k, (hx, cx) )
        return hx, cx
    
    def initHidden(self):
        hx = torch.zeros(batch_size, self.hidden_size).cuda()
        cx = torch.zeros(batch_size, self.hidden_size).cuda()
        return hx, cx