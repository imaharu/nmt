from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class Decoder(nn.Module):
    def __init__(self, target_size, hidden_size, opts):
        super(Decoder, self).__init__()
        self.opts = opts
        self.attention = Attention(opts)
        self.embed = nn.Embedding(target_size, embed_size, padding_idx=0)
        self.drop = nn.Dropout(p=0.2)
        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, t_input, hx, cx, encoder_outputs, encoder_features, coverage_vector, mask_tensor):
        embed = self.embed(t_input)
        embed = self.drop(embed)
#        if torch.nonzero(t_input.eq(0)).size(0):
#            before_hx, before_cx = hx, cx
#            mask = torch.cat( [ t_input.unsqueeze(-1) ] * hidden_size, 1)
#            hx, cx = self.lstm(embed, (hx, cx) )
#            hx = torch.where(mask == 0, before_hx, hx)
#            cx = torch.where(mask == 0, before_cx, cx)
#        else:
#            hx, cx = self.lstm(embed, (hx, cx) )
        hx, cx = self.lstm(embed, (hx, cx) )
        final_dist, align_weight, next_coverage_vector = self.attention(
                hx, encoder_outputs, encoder_features , coverage_vector, mask_tensor)
        return final_dist, hx, cx, align_weight, next_coverage_vector

class Attention(nn.Module):
    def __init__(self, opts):
        super(Attention, self).__init__()
        self.opts = opts
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.coverage = Coverage()

    def forward(self, decoder_hx, encoder_outputs, encoder_features, coverage_vector, mask_tensor):
        t_k, b, n = list(encoder_outputs.size())
        dec_feature = self.W_s(decoder_hx)
        dec_feature = dec_feature.unsqueeze(0).expand(t_k, b, n)
        att_features = encoder_features + dec_feature
        if self.opts["coverage_vector"]:
            att_features = self.coverage.getFeature(coverage_vector, att_features)

        e = torch.tanh(att_features)
        scores = self.v(e)
        align_weight = torch.softmax(scores, dim=0) * mask_tensor

        if self.opts["coverage_vector"]:
            next_coverage_vector = self.coverage.getNextCoverage(coverage_vector, align_weight)
        else:
            next_coverage_vector = coverage_vector

        content_vector = (align_weight * encoder_outputs).sum(0)
        concat = torch.cat((content_vector, decoder_hx), 1)
        final_dist = torch.tanh(self.linear(concat))
        return final_dist, align_weight, next_coverage_vector

class Coverage(nn.Module):
    def __init__(self):
        super(Coverage, self).__init__()
        self.W_c = nn.Linear(1, hidden_size)

    def getFeature(self, coverage_vector, att_features):
        coverage_input = coverage_vector.view(-1, 1)
        coverage_features = self.W_c(coverage_input).unsqueeze(-1)
        coverage_features = coverage_features.view(-1, att_features.size(1), hidden_size)
        att_features += coverage_features
        return att_features

    def getNextCoverage(self, coverage_vector, align_weight):
        next_coverage_vector = coverage_vector + align_weight
        return next_coverage_vector
