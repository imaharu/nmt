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
        self.w_encoder = WordEncoder(source_size, hidden_size)
        self.s_encoder = SentenceEncoder(source_size, hidden_size)

class WordEncoder(nn.Module):
    def __init__(self, source_size, hidden_size):
        super(WordEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.source_size = source_size
        self.embed_source = nn.Embedding(source_size, hidden_size, padding_idx=0)
        self.drop_source = nn.Dropout(p=0.2)
        self.lstm_source = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, sentence_words, w_hx, w_cx):
        source_k = self.embed_source(sentence_words)
        source_k = self.drop_source(source_k)
        w_hx, w_cx = self.lstm_source(source_k, (w_hx, w_cx) )
        return w_hx, w_cx

    def initHidden(self):
        hx = torch.zeros(batch_size, self.hidden_size).cuda(device=device)
        cx = torch.zeros(batch_size, self.hidden_size).cuda(device=device)
        return hx, cx

class SentenceEncoder(nn.Module):
    def __init__(self, source_size, hidden_size):
        super(SentenceEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.source_size = source_size
        self.drop_source_s = nn.Dropout(p=0.2)
        self.lstm_source_s = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, s_hx, w_hx, w_cx):
        w_hx = self.drop_source_s(s_hx)
        s_hx, s_cx = self.lstm_source_s(s_hx, (w_hx, w_cx) )
        return s_hx, s_cx

    def initHidden(self):
        hx = torch.zeros(batch_size, self.hidden_size).cuda(device=device)
        cx = torch.zeros(batch_size, self.hidden_size).cuda(device=device)
        return hx, cx

class Decoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(Decoder, self).__init__()
        self.w_decoder = WordDecoder(target_size, hidden_size)
        self.s_decoder = SentenceDecoder(target_size, hidden_size)

class WordDecoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(WordDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.embed_target = nn.Embedding(target_size, hidden_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=0.2)
        self.lstm_target = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, target_words, w_hx, w_cx):
        target_k = self.embed_target(target_words)
        target_k = self.drop_target(target_k)
        w_hx, cx = self.lstm_target(target_k, (w_hx, w_cx) )
        return w_hx, w_cx

class SentenceDecoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(SentenceDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.drop_target_doc = nn.Dropout(p=0.2)
        self.lstm_target_doc = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, w_hx, s_hx, s_cx):
        w_hx = self.drop_target_doc(w_hx)
        s_hx, s_cx = self.lstm_target_doc(w_hx, (s_hx, s_cx) )
        return s_hx, s_cx

#    def forward(self, s_hx, w_hx, w_cx):
#        s_hx = self.drop_target_doc(s_hx)
#        s_hx, s_cx = self.lstm_target_doc(s_hx, (w_hx, w_cx) )
#        return s_hx, s_cx
