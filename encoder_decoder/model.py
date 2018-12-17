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
        self.embed_source = nn.Embedding(source_size, hidden_size, padding_idx=0)
        self.drop_source = nn.Dropout(p=0.2)
        self.lstm1 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm3 = nn.LSTMCell(hidden_size, hidden_size)

    def create_mask(self ,sentence_words):
        return torch.cat( [ sentence_words.unsqueeze(-1) ] * hidden_size, 1)

    def multi_layer(self, source_k, mask, hx, cx):
        layer_num = 3
        for i in range(1, layer_num + 1):
            b_hx , b_cx = hx, cx
            hx, cx = eval("self.lstm" + str(i))(source_k, (hx, cx) )
            hx = torch.where(mask == 0, b_hx, hx)
            cx = torch.where(mask == 0, b_cx, cx)
        return hx, cx

    def forward(self, sentence_words, hx, cx):
        source_k = self.embed_source(sentence_words)
        source_k = self.drop_source(source_k)

        mask = self.create_mask(sentence_words)
        hx, cx = self.multi_layer(source_k, mask ,hx, cx)
        return hx, cx

    def initHx(self):
        hx = torch.zeros(batch_size, self.hidden_size).cuda()
        return hx

    def initCx(self):
        cx = torch.zeros(batch_size, self.hidden_size).cuda()
        return cx

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.embed_target = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=0.2)
        self.lstm1 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm3 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def create_mask(self ,sentence_words):
        return torch.cat( [ sentence_words.unsqueeze(-1) ] * hidden_size, 1)

    def multi_layer(self, target_k, mask, hx, cx):
        layer_num = 3
        for i in range(1, layer_num + 1):
            b_hx , b_cx = hx, cx
            hx, cx = eval("self.lstm" + str(i))(target_k, (hx, cx) )
            hx = torch.where(mask == 0, b_hx, hx)
            cx = torch.where(mask == 0, b_cx, cx)
        return hx, cx

    def forward(self, target_words, hx, cx):
        target_k = self.embed_target(target_words)
        target_k = self.drop_target(target_k)
        mask = self.create_mask(target_words)
        hx, cx = self.multi_layer(target_k, mask ,hx, cx)
        return hx, cx
