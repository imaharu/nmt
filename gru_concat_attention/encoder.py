from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class Encoder(nn.Module):
    def __init__(self, source_size, hidden_size, opts):
        super(Encoder, self).__init__()
        self.opts = opts
        self.embed_source = nn.Embedding(source_size, embed_size, padding_idx=0)
        self.drop = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)#, bidirectional=self.opts["bidirectional"])
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=self.opts["bidirectional"])
        self.W_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, sentences):
        b = sentences.size(0)
        input_lengths = torch.tensor(
            [seq.size(-1) for seq in sentences])
        embed = self.embed_source(sentences)
        embed = self.drop(embed)
        sequence = rnn.pack_padded_sequence(embed, input_lengths, batch_first=True)

        packed_output, hx = self.gru(sequence)
        encoder_outputs, _ = rnn.pad_packed_sequence(
            packed_output
        )
        if self.opts["bidirectional"]:
            encoder_outputs = encoder_outputs[:, :, :hidden_size] + encoder_outputs[:, :, hidden_size:]
            hx = hx.view(-1, 2 , b, hidden_size).sum(1)
        encoder_feature = self.W_h(encoder_outputs)
        hx = hx.view(b, -1)
        return encoder_outputs, encoder_feature, hx
