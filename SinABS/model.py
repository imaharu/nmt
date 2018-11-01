from define_sin import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *

class HierachicalEncoderDecoder(nn.Module):
    def __init__(self, source_size, target_size, hidden_size):
        super(HierachicalEncoderDecoder, self).__init__()
        self.encoder = Encoder(source_size, hidden_size)
        self.decoder = Decoder(target_size, hidden_size)

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
        # Maybe not need
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

    def forward(self, sentence_hx, hx, cx):
        sentence_hx = self.drop_source_doc(sentence_hx)
        hx, cx = self.lstm_source_doc(sentence_hx, (hx, cx) )
        return hx, cx

    def initHidden(self):
        hx = torch.zeros(batch_size, self.hidden_size).cuda()
        cx = torch.zeros(batch_size, self.hidden_size).cuda()
        return hx, cx

class Decoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(Decoder, self).__init__()
        self.sentence_decoder = SentenceDecoder(target_size, hidden_size)
        self.doc_decoder = DocDecoder(target_size, hidden_size)

class SentenceDecoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(SentenceDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.embed_target = nn.Embedding(target_size, hidden_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=0.2)
        self.lstm_target = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, target_words, hx, cx):
        target_k = self.embed_target(target_words)
        target_k = self.drop_target(target_k)
        hx, cx = self.lstm_target(target_k, (hx, cx) )
        return hx, cx

class DocDecoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(DocDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.target_size = target_size
        # Maybe not need
        self.embed_target_doc = nn.Embedding(target_size, hidden_size, padding_idx=0)
        self.drop_target_doc = nn.Dropout(p=0.2)
        self.lstm_target_doc = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, sentence_hx, hx, cx):
        sentence_hx = self.drop_target_doc(sentence_hx)
        hx, cx = self.lstm_target_doc(sentence_hx, (hx, cx) )
        return hx, cx
