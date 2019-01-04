from define_variable import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import *

def create_mask(source_sentence_words):
    return torch.cat( [ source_sentence_words.unsqueeze(-1) ] * hidden_size, 1)

def map_tuple(func, tup):
    return tuple(map(func, tup))

class EncoderDecoder(nn.Module):
    def __init__(self, source_size, target_size, hidden_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(source_size, hidden_size)
        self.decoder = Decoder(target_size, hidden_size)
        self.attention = Attention(hidden_size)

    def forward(self, source=None, target=None, train=False, phase=0):
        if train:
            loss = 0
            source = source.t()
            target = target.t()

            hx_list , hx_cx = self.encoder(source)

            # attenstion mask for inf
            mask_tensor = source.eq(PADDING).unsqueeze(-1)

            lines_t_last = target[1:]
            lines_f_last = target[:(len(source) - 1)]
            hx_cx = map_tuple(lambda x: x.squeeze(0), hx_cx)
            for words_f, word_t in zip(lines_f_last, lines_t_last):
                hx , cx = self.decoder(words_f, hx_cx)
                hx_new = self.attention(hx, hx_list, mask_tensor)
                loss += F.cross_entropy(
                    self.decoder.linear(hx_new), word_t , ignore_index=0)
            return loss

        elif phase == 1:
            source = source.t()
            hx_list , hx_cx = self.encoder(source)

            mask_tensor = source.eq(PADDING).unsqueeze(-1)
            hx_cx = map_tuple(lambda x: x.squeeze(0), hx_cx)

            word_id = torch.tensor( [ target_dict["[START]"] ] ).cuda()
            result = []
            loop = 0
            while True:
                hx , cx = self.decoder(word_id, hx_cx)
                hx_new = self.attention(hx, hx_list, mask_tensor)
                word_id = torch.tensor([ torch.argmax(F.softmax(self.decoder.linear(hx_new), dim=1).data[0]) ]).cuda()
                loop += 1
                if loop >= 50 or int(word_id) == target_dict['[STOP]']:
                    break
                result.append(word_id)
            return result

class Encoder(nn.Module):
    def __init__(self, source_size, hidden_size):
        super(Encoder, self).__init__()
        self.embed_source = nn.Embedding(source_size, hidden_size, padding_idx=0)
        self.drop_source = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, sentences):
        embed = self.embed_source(sentences)
        embed = self.drop_source(embed)
        return self.lstm(embed)

class Decoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(Decoder, self).__init__()
        self.embed_target = nn.Embedding(target_size, hidden_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=0.2)
        self.lstm_target = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, target_words, hx_cx):
        target_k = self.embed_target(target_words)
        target_k = self.drop_target(target_k)
        hx, cx = self.lstm_target(target_k, hx_cx )
        return hx, cx

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, decoder_hx, ew_hx_list, mask_tensor):
        attention_weights = (decoder_hx * ew_hx_list).sum(-1, keepdim=True)
        masked_score = attention_weights.masked_fill_(mask_tensor, float('-inf'))
        align_weight = F.softmax(masked_score, 0)
        content_vector = (align_weight * ew_hx_list).sum(0)
        concat = torch.cat((content_vector, decoder_hx), 1)
        hx_attention = torch.tanh(self.linear(concat))
        return hx_attention
