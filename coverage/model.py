from define_variable import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

def create_mask(source_sentence_words):
    return torch.cat( [ source_sentence_words.unsqueeze(-1) ] * hidden_size, 1)

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
            encoder_outputs , encoder_feature , hx, cx = self.encoder(source)

            mask_tensor = source.t().eq(PADDING).unsqueeze(-1).float().cuda()
            target = target.t()
            for words_f, words_t in zip(target[:-1],  target[1:]):
                hx, cx = self.decoder(words_f, hx, cx)
                hx_new = self.attention(hx, encoder_outputs, encoder_feature , mask_tensor)
                loss += F.cross_entropy(
                    self.decoder.linear(hx_new), words_t , ignore_index=0)
            return loss

        elif phase == 1:
            encoder_outputs , encoder_feature , hx, cx = self.encoder(source)
            mask_tensor = source.t().eq(PADDING).unsqueeze(-1).float().cuda()
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

class Encoder(nn.Module):
    def __init__(self, source_size, hidden_size, opts):
        super(Encoder, self).__init__()
        self.opts = opts
        self.embed_source = nn.Embedding(source_size, hidden_size, padding_idx=0)
        self.drop_source = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=self.opts["bidirectional"])
        self.W_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, sentences):
        '''
            return
                encoder_ouput, hx, cx
            option
                bidirectional
        '''
        input_lengths = torch.tensor(
            [seq.size(-1) for seq in sentences])
        embed = self.embed_source(sentences)
        embed = self.drop_source(embed)
        sequence = rnn.pack_padded_sequence(embed, input_lengths, batch_first=True)

        packed_output, (hx, cx) = self.lstm(sequence)
        encoder_outputs, _ = rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        if self.opts["bidirectional"]:
            encoder_outputs = encoder_outputs[:, :, :hidden_size] + encoder_outputs[:, :, hidden_size:]
            hx = hx.view(-1, 2 , batch_size, hidden_size).sum(1)
            cx = cx.view(-1, 2 , batch_size, hidden_size).sum(1)
        encoder_outputs = encoder_outputs.contiguous()
        encoder_feature = encoder_outputs.view(-1, hidden_size)
        encoder_feature = self.W_h(encoder_feature)
        hx = hx.view(batch_size, -1)
        cx = cx.view(batch_size, -1)
        return encoder_outputs, encoder_feature, hx, cx

class Decoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(Decoder, self).__init__()
        self.embed_target = nn.Embedding(target_size, hidden_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=0.2)
        self.lstm_target = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, target_words, hx, cx):
        embed = self.embed_target(target_words)
        embed = self.drop_target(embed)
        hx, cx = self.lstm_target(embed, (hx, cx) )
        return hx, cx

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, decoder_hx, encoder_outputs , encoder_feature , mask_tensor):
        b, t_k, n = list(encoder_outputs.size())
        dec_feature = self.W_s(decoder_hx)
        dec_feature_expanded = dec_feature.unsqueeze(1).expand(b, t_k, n).contiguous()
        dec_feature_expanded = dec_feature_expanded.view(-1, n)
        att_features = encoder_feature + dec_feature_expanded
        e = torch.tanh(att_features)
        scores = self.v(e)
        scores = scores.view(-1, t_k)
        mask_tensor = mask_tensor.squeeze().view(b, -1)
        attn_dist = F.softmax(scores, dim=1) * mask_tensor
        attn_dist = attn_dist.unsqueeze(-1)

        encoder_outputs = encoder_outputs.view(-1, b, n)
        attn_dist = attn_dist.view(-1, b, 1)
        content_vector = (attn_dist * encoder_outputs).sum(0)
        concat = torch.cat((content_vector, decoder_hx), 1)
        hx_attention = torch.tanh(self.linear(concat))
        return hx_attention
