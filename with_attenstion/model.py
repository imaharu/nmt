from define_variable import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import *

class EncoderDecoder(nn.Module):
    def __init__(self, source_size, output_size, hidden_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(source_size, hidden_size)
        self.decoder = Decoder(output_size, hidden_size)

class Encoder(nn.Module):
    def __init__(self, source_size, hidden_size):
        super(Encoder, self).__init__()
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

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.embed_target = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=0.2)
        self.lstm_target = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.attention_linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, target_words, hx, cx):
        target_k = self.embed_target(target_words)
        target_k = self.drop_target(target_k)
        hx, cx = self.lstm_target(target_k, (hx, cx) )
        return hx, cx

    def attention(self, ht, list_hs, list_source_mask, inf):
        dot = (ht * list_hs).sum(-1, keepdim=True)
        dot = torch.where(list_source_mask == 0, inf, dot)
        a_t = F.softmax( dot, 0 )
        d = (a_t * list_hs).sum(0)
        concat  = torch.cat((d, ht), 1)
        ht_new = F.tanh(self.attention_linear(concat))
        return ht_new
