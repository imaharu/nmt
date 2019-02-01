from define import *
from encoder import *
from decoder import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class EncoderDecoder(nn.Module):
    def __init__(self, source_size, target_size, hidden_size):
        super(EncoderDecoder, self).__init__()
        opts = { "bidirectional": True }
        self.encoder = Encoder(source_size, hidden_size, opts)
        self.decoder = Decoder(target_size, hidden_size)
        self.attention = Attention(hidden_size)

    def forward(self, source=None, target=None, train=False, generate=False):
        encoder_outputs, encoder_feature, hx = self.encoder(source)
        mask_tensor = source.t().gt(PADDING).unsqueeze(-1).float().cuda()
        if train:
            loss = 0
            target = target.t()
            for words_f, words_t in zip(target[:-1],  target[1:]):
                hx = self.decoder(words_f, hx)
                final_dist = self.attention(hx, encoder_outputs, encoder_feature , mask_tensor)
                loss += F.cross_entropy(
                   self.decoder.linear(final_dist), words_t , ignore_index=0)
            return loss

        elif generate:
            encoder_outputs, encoder_features, hx = self.encoder(source)
            mask_tensor = source.t().gt(PADDING).unsqueeze(-1).float().cuda()
            word_id = torch.tensor( [ target_dict["[START]"] ] ).cuda()
            sentence = []
            loop = 0
            while True:
                hx = self.decoder(word_id, hx)
                final_dist = self.attention(hx, encoder_outputs, encoder_feature , mask_tensor)

                word_id = torch.tensor([ torch.argmax(
                        F.softmax(self.decoder.linear(final_dist), dim=1).data[0]) ]).cuda()
                loop += 1
                if loop >= 50 or int(word_id) == target_dict['[STOP]']:
                    break
                sentence.append(word_id)
            return sentence
