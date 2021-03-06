from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class Decoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(Decoder, self).__init__()
        self.embed_target = nn.Embedding(target_size, embed_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=0.2)
        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, target_words, hx, cx):
        embed = self.embed_target(target_words)
        embed = self.drop_target(embed)
        w_hx, w_cx = self.lstm(embed, (hx, cx) )
        return hx, cx

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, decoder_hx, encoder_outputs , encoder_feature , mask_tensor):
        t_k, b, n = list(encoder_outputs.size())
        dec_feature = self.W_s(decoder_hx)
        dec_feature = dec_feature.unsqueeze(0).expand(t_k, b, n)
        att_features = encoder_feature + dec_feature
        e = torch.tanh(att_features)
        scores = self.v(e)
        attn_dist = torch.softmax(scores, dim=0) * mask_tensor
        content_vector = (attn_dist * encoder_outputs).sum(0)
        concat = torch.cat((content_vector, decoder_hx), 1)
        hx_attention = torch.tanh(self.linear(concat))
        return hx_attention
