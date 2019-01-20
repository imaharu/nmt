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
            k = 3 # beam_size
            self.encoder_outputs = encoder_outputs.expand(-1, k, hidden_size)
            self.encoder_feature = encoder_feature.expand(-1, k, hidden_size)
            self.mask_tensor = mask_tensor
            hx = hx.expand(k, hidden_size)
            cx = cx.expand(k, hidden_size)
            word_ids = torch.tensor( [ target_dict["[START]"]] * k).cuda()
            top_k_scores = torch.zeros(k, 1).cuda()
            top_k_sentences = []
            step = 0
            top_sentences = self.beam_search(top_k_scores, top_k_sentences, word_ids, hx, cx, k, step)
            print("pass")
            exit()
            return result

    def getscores(self, top_k_words, hx, cx):
        hx, cx = self.decoder(top_k_words, hx, cx)
        hx_new = self.attention(hx, self.encoder_outputs, self.encoder_feature, self.mask_tensor)
        scores = F.log_softmax(self.decoder.linear(hx_new), dim=1)
        return hx, cx, scores

    def beam_search(self, top_k_scores, top_k_sentences, top_k_words, hx, cx, k, step):
        if step == 2:
            return top_k_sentences
        hx, cx, scores = self.getscores(top_k_words, hx, cx)
        # scoreの合計値が高い上位k個を取るため
        scores = top_k_scores.expand_as(scores) + scores
        if step == 0:
            top_k_scores, top_k_words = scores[0].topk(k)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k)
        # 以前のword_id
        prev_words = top_k_words / len(target_dict)
        top_k_words = top_k_words % len(target_dict)
        top_k_scores = top_k_scores.unsqueeze(-1)
        return self.beam_search(top_k_scores, top_k_sentences, top_k_words, hx, cx, k, step + 1)
