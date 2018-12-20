from define_variable import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import *

class EncoderDecoder(nn.Module):
    def __init__(self, source_size, output_size, embde_size, hidden_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(source_size, embde_size, hidden_size)
        self.decoder = Decoder(output_size, embde_size, hidden_size)

class Encoder(nn.Module):
    def __init__(self, source_size, embde_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_source = nn.Embedding(source_size, embde_size, padding_idx=0)
        self.drop_source = nn.Dropout(p=args.dropout)
        self.lstm = nn.ModuleList([ nn.LSTMCell(hidden_size, hidden_size) for i in range(args.layer_num)])

    def create_mask(self ,sentence_words):
        return torch.cat( [ sentence_words.unsqueeze(-1) ] * args.hidden_size, 1)

    def multi_layer(self, source_k, mask, lhx, lcx):
        for i, lstm in enumerate(self.lstm):
            b_hx , b_cx = lhx[i], lcx[i]
            if i == 0:
                lhx[i], lcx[i] = lstm(source_k, (lhx[i], lcx[i]) )
            else:
                lhx[i], lcx[i] = lstm(lhx[i - 1], (lhx[i], lcx[i]) )
            lhx[i] = torch.where(mask == 0, b_hx, lhx[i])
            lcx[i] = torch.where(mask == 0, b_cx, lcx[i])
        return lhx, lcx

    def forward(self, sentence_words, lhx, lcx):
        source_k = self.embed_source(sentence_words)
        source_k = self.drop_source(source_k)
        mask = self.create_mask(sentence_words)
        lhx, lcx = self.multi_layer(source_k, mask ,lhx, lcx)
        return lhx, lcx

    def init(self):
        init = torch.zeros(args.batch_size, self.hidden_size).cuda()
        return init

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.embed_target = nn.Embedding(output_size, embed_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=args.dropout)
        self.lstm = nn.ModuleList([ nn.LSTMCell(hidden_size, hidden_size) for i in range(args.layer_num)])
        self.linear = nn.Linear(hidden_size, output_size)
        self.attention_linear = nn.Linear(hidden_size * 2, hidden_size)

    def create_mask(self ,sentence_words):
        return torch.cat( [ sentence_words.unsqueeze(-1) ] * args.hidden_size, 1)

    def multi_layer(self, target_k, mask, lhx, lcx):
        for i, lstm in enumerate(self.lstm):
            b_hx , b_cx = lhx[i], lcx[i]
            if i == 0:
                lhx[i], lcx[i] = lstm(target_k, (lhx[i], lcx[i]) )
            else:
                lhx[i], lcx[i] = lstm(lhx[i - 1], (lhx[i], lcx[i]) )
            lhx[i] = torch.where(mask == 0, b_hx, lhx[i])
            lcx[i] = torch.where(mask == 0, b_cx, lcx[i])
        return lhx, lcx

    def forward(self, target_words, lhx, lcx):
        target_k = self.embed_target(target_words)
        target_k = self.drop_target(target_k)
        mask = self.create_mask(target_words)
        lhx, lcx = self.multi_layer(target_k, mask ,lhx, lcx)
        return lhx, lcx

    def attention(self, hx, list_hx, list_source_mask, inf):
        dot = (hx * list_hx).sum(-1, keepdim=True)
        dot = torch.where(list_source_mask == 0, inf, dot)
        a_t = F.softmax( dot, 0 )
        d = (a_t * list_hx).sum(0)
        concat  = torch.cat((d, hx), 1)
        hx_new = F.tanh(self.attention_linear(concat))
        return hx_new
