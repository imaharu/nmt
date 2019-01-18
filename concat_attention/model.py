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

    def forward(self, source=None, target=None, train=False, phase=0):
        encoder_outputs , encoder_feature , hx, cx = self.encoder(source)
        mask_tensor = source.t().gt(PADDING).unsqueeze(-1).float().cuda()
        if train:
            loss = 0
            target = target.t()
            for words_f, words_t in zip(target[:-1],  target[1:]):
                hx, cx = self.decoder(words_f, hx, cx)
                hx_new = self.attention(hx, encoder_outputs, encoder_feature , mask_tensor)
                loss += F.cross_entropy(
                   self.decoder.linear(hx_new), words_t , ignore_index=0)
            return loss

        elif phase == 1:
            word_id = torch.tensor( [ target_dict["[START]"] ] ).cuda()
            result = []
            loop = 0
            while True:
                hx , cx = self.decoder(word_id, hx, cx)
                hx_new = self.attention(hx, encoder_outputs, encoder_feature , mask_tensor)
                word_id = torch.tensor([ torch.argmax(F.softmax(self.decoder.linear(hx_new), dim=1).data[0]) ]).cuda()
                loop += 1
                if loop >= 50 or int(word_id) == target_dict['[STOP]']:
                    break
                result.append(word_id)
            return result
