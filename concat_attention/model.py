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
        encoder_outputs, encoder_feature, hx, cx = self.encoder(source)
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
            self.mask_tensor = mask_tensor
            hx = hx.expand(k, hidden_size)
            cx = cx.expand(k, hidden_size)
            max_step = 50
            finish_sentences, finish_scores = self.beam_search(hx, cx, k, max_step, encoder_outputs, encoder_feature)
            new_completed_scores = list()
            for seq, score in zip(finish_sentences, finish_scores):
                length_penalty = 0.6
                ln = len(seq) - 1
                lp = ((5 + ln) ** length_penalty) / ((5 + 1) ** length_penalty)
                new_completed_scores.append(score/lp)
            index = new_completed_scores.index(max(new_completed_scores))
            best_summary = finish_sentences[index][1:]
            best_summary = self.eos_truncate(best_summary, target_dict["[STOP]"])
            return best_summary

    def eos_truncate(self, labels, eos_label):
        if (labels == eos_label).nonzero().size(0):
            labels = labels.narrow(0, 0, (labels == eos_label).nonzero().item())
        return labels

    def getscores(self, top_k_words, hx, cx):
        hx, cx = self.decoder(top_k_words, hx, cx)
        hx_new = self.attention(hx, self.encoder_outputs, self.encoder_feature, self.mask_tensor)
        scores = F.log_softmax(self.decoder.linear(hx_new), dim=1)
        return hx, cx, scores

    def beam_search(self, hx, cx, k, max_step, encoder_outputs, encoder_feature):
        self.encoder_outputs = encoder_outputs.expand(-1, k, hidden_size)
        self.encoder_feature = encoder_feature.expand(-1, k, hidden_size)
        words = torch.tensor( [ target_dict["[START]"]] * k).cuda()
        top_k_scores = torch.zeros(k, 1).cuda()
        finish_sentences = []
        finish_scores = []
        seqs = words.unsqueeze(1)
        for step in range(max_step):
            hx, cx, scores = self.getscores(words, hx, cx)
            # scoreの合計値が高い上位k個を取るため
            scores = top_k_scores.expand_as(scores) + scores
            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(k)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k)
            next_word_indexs = top_k_words % len(target_dict)
            prev_word_indexs = top_k_words / len(target_dict)
            seqs = torch.cat((seqs[prev_word_indexs], next_word_indexs.unsqueeze(1)), dim=1)

            incomplete_indexs = [index for index, next_word in enumerate(next_word_indexs) if next_word != target_dict['[STOP]']]

            ## eos ##
            if torch.nonzero(next_word_indexs.eq(target_dict['[STOP]'])).size(0):
                for score in top_k_scores.masked_select(next_word_indexs.eq(target_dict['[STOP]'])).unsqueeze(1):
                    finish_scores.append(score.item())
                for finish_sentence in seqs[next_word_indexs.eq(target_dict['[STOP]'])].unsqueeze(1):
                    finish_sentences.append(finish_sentence.squeeze())
            ### 終了条件 ###
            if step == max_step or len(finish_scores) == k:
                break
            seqs = seqs[incomplete_indexs]
            hx = hx[prev_word_indexs[incomplete_indexs]]
            cx = cx[prev_word_indexs[incomplete_indexs]]
            self.encoder_outputs = encoder_outputs.expand(-1, k - len(finish_scores), hidden_size)
            self.encoder_feature = encoder_feature.expand(-1, k - len(finish_scores), hidden_size)
            top_k_scores = top_k_scores[incomplete_indexs].unsqueeze(1)
            words = next_word_indexs[prev_word_indexs[incomplete_indexs]]
        if len(finish_scores) == 0:
            for score in top_k_scores.unsqueeze(1):
                finish_scores.append(score.item())
            for finish_sentence in seqs.unsqueeze(1):
                finish_sentences.append(finish_sentence.squeeze())
        return finish_sentences, finish_scores
