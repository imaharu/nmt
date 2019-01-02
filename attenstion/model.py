from define_variable import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import *

def create_mask(source_sentence_words):
    return torch.cat( [ source_sentence_words.unsqueeze(-1) ] * hidden_size, 1)

class EncoderDecoder(nn.Module):
    def __init__(self, source_size, output_size, hidden_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(source_size, hidden_size)
        self.decoder = Decoder(output_size, hidden_size)

    def forward(self, source, target):
        def init(source_len):
            hx = torch.zeros(source_len, hidden_size).cuda(device=source.device)
            cx = torch.zeros(source_len, hidden_size).cuda(device=source.device)
            return hx, cx

        source_len = len(source)
        hx, cx = init(source_len)

        source = source.t()
        target = target.t()
        loss = 0

        hx_list = []
        lmasks = []
        for words in source:
            hx , cx = self.encoder(words, hx, cx)
            hx_list.append(hx)
            masks = torch.cat( [ words.unsqueeze(-1) ] , 1)
            lmasks.append( torch.unsqueeze(masks, 0) )

        hx_list = torch.stack(hx_list, 0)
        lmasks = torch.cat(lmasks)

        inf = torch.full((len(source), source_len), float("-inf")).cuda(device=source.device)
        inf = torch.unsqueeze(inf, -1)

        lines_t_last = target[1:]
        lines_f_last = target[:(len(source) - 1)]

        for words_f, word_t in zip(lines_f_last, lines_t_last):
            hx , dw_cx = self.decoder(words_f, hx, cx)
            new_hx = self.decoder.attention(
                        hx, hx_list, lmasks, inf)
            loss += F.cross_entropy(
               self.decoder.linear(new_hx),
                   word_t , ignore_index=0)

        loss = torch.tensor(loss, requires_grad=True).unsqueeze(0).cuda(device=source.device)
        return loss

    def eval(self, source, target):
        return 1

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

    def attention(self, decoder_hx, ew_hx_list, ew_mask, inf):
        attention_weights = (decoder_hx * ew_hx_list).sum(-1, keepdim=True)
        masked_score = torch.where(ew_mask == 0, inf, attention_weights)
        align_weight = F.softmax(masked_score, 0)
        content_vector = (align_weight * ew_hx_list).sum(0)
        concat = torch.cat((content_vector, decoder_hx), 1)
        hx_attention = torch.tanh(self.attention_linear(concat))
        return hx_attention
