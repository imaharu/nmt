from define_variable import *
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
        if train:
            loss = 0
            target = target.t()
            encoder_outputs , hx, cx = self.encoder(source)
            mask_tensor = source.t().eq(PADDING).unsqueeze(-1)
            for words_f, words_t in zip(target[:-1] , target[1:]):
                hx, cx = self.decoder(words_f, hx, cx)
                hx_new = self.attention(hx, encoder_outputs, mask_tensor)
                loss += F.cross_entropy(
                    self.decoder.linear(hx_new), words_t , ignore_index=0)
            return loss

        elif phase == 1:
            encoder_outputs , hx, cx = self.encoder(source)
            mask_tensor = source.t().eq(PADDING).unsqueeze(-1)
            word_id = torch.tensor( [ target_dict["[START]"] ] ).cuda()
            result = []
            loop = 0
            while True:
                hx , cx = self.decoder(word_id, hx, cx)
                hx_new = self.attention(hx, encoder_outputs, mask_tensor)
                word_id = torch.tensor([ torch.argmax(F.softmax(self.decoder.linear(hx_new), dim=1).data[0]) ]).cuda()
                loop += 1
                if loop >= 50 or int(word_id) == target_dict['[STOP]']:
                    break
                result.append(word_id)
            return result

class Encoder(nn.Module):
    def __init__(self, source_size, hidden_size, opts):
        super(Encoder, self).__init__()
        self.opts = opts
        self.embed = nn.Embedding(source_size, hidden_size, padding_idx=0)
        self.drop = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=self.opts["bidirectional"])

    def forward(self, sentences):
        '''
            return
                encoder_ouput, hx, cx
            option
                bidirectional
        '''
        b = sentences.size(0)
        input_lengths = sentences.ne(0).sum(-1)
        embed = self.embed(sentences)
        embed = self.drop(embed)
        sequence = rnn.pack_padded_sequence(embed, input_lengths, batch_first=True)

        packed_output, (hx, cx) = self.lstm(sequence)
        output, _ = rnn.pad_packed_sequence(
            packed_output
        )
        if self.opts["bidirectional"]:
            output = output[:, :, :hidden_size] + output[:, :, hidden_size:]
            hx = hx.view(-1, 2 , b, hidden_size).sum(1)
            cx = cx.view(-1, 2 , b, hidden_size).sum(1)
        hx = hx.view(b, -1)
        cx = cx.view(b, -1)
        return output, hx, cx

class Decoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(target_size, hidden_size, padding_idx=0)
        self.drop = nn.Dropout(p=0.2)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, target_words, hx, cx):
        embed = self.embed(target_words)
        embed = self.drop(embed)
        if torch.nonzero(target_words.eq(0)).size(0):
            before_hx, before_cx = hx, cx
            mask = torch.cat( [ target_words.unsqueeze(-1) ] * hidden_size, 1)
            hx, cx = self.lstm(embed, (hx, cx) )
            hx = torch.where(mask == 0, before_hx, hx)
            cx = torch.where(mask == 0, before_cx, cx)
        else:
            hx, cx = self.lstm(embed, (hx, cx) )
        return hx, cx

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, decoder_hx, hx_list, mask_tensor):
        attention_weights = (decoder_hx * hx_list).sum(-1, keepdim=True)
        masked_score = attention_weights.masked_fill_(mask_tensor, float('-inf'))
        align_weight = F.softmax(masked_score, 0)
        content_vector = (align_weight * hx_list).sum(0)
        concat = torch.cat((content_vector, decoder_hx), 1)
        hx_attention = torch.tanh(self.linear(concat))
        return hx_attention
