import torch
from torch import tensor as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
import time
import numpy as np
from get_data import *
import torch.optim as optim

train_num, hidden_size= 20000, 256
test_num = 1000

input_vocab , input_lines, input_lines_number = {}, {}, {}
target_vocab ,target_lines ,target_lines_number = {}, {}, {}
output_input_lines = {}
translate_words = {}

get_train_data_input(train_num, input_vocab, input_lines_number, input_lines)
ev = len(input_vocab) + 1

get_train_data_target(train_num, target_vocab, target_lines_number, target_lines, translate_words)
jv = len(target_vocab) + 1

get_test_data_target(test_num, output_input_lines)

class EncoderDecoder(nn.Module):
    def __init__(self, source_size, output_size, hidden_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(source_size, hidden_size)
        self.decoder = Decoder(output_size, hidden_size)

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
        # source_k = self.drop_source(source_k)
        hx, cx = self.lstm_source(source_k, (hx, cx) )
        return hx, cx
    
    def initHidden(self):
        hx = torch.zeros(1, self.hidden_size).cuda()
        cx = torch.zeros(1, self.hidden_size).cuda()
        return hx, cx

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.embed_target = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=0.2)
        self.lstm_target = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, target_words, hx, cx):
        target_k = self.embed_target(target_words)
        # target_k = self.drop_target(target_k)
        hx, cx = self.lstm_target(target_k, (hx, cx) )
        return hx, cx

def output(encoder, decoder, output_input_line):
    result = []
    loop = 0
    hx, cx = encoder.initHidden()

    for i in range(len(output_input_line)):
        ## 辞書にある場合は
        if input_vocab.get(output_input_line[i]):
            word_id = torch.tensor([input_vocab[output_input_line[i]]]).cuda()
        else:
            word_id = torch.tensor([ input_vocab["<unk>"] ]).cuda()
        hx, cx = encoder(word_id, hx, cx)
        word_id = torch.tensor( [ target_vocab["<bos>"] ] ).cuda()

    while(int(word_id) != target_vocab['<eos>']):
        if loop >= 50:
            break
        hx, cx = decoder(word_id, hx, cx)

        word_id = torch.tensor([ torch.argmax(F.softmax(decoder.linear(hx), dim=1).data[0]) ]).cuda()
        loop += 1
        if int(word_id) != target_vocab['<eos>'] and int(word_id) != 0:
            result.append(translate_words[int(word_id)])
    return result

if __name__ == '__main__':
    device = torch.device('cuda:0')

    model = EncoderDecoder(ev, jv, hidden_size).to(device)
    model.load_state_dict(torch.load("model-20.model"))
    model.eval()
    optimizer = torch.optim.Adam( model.parameters(), weight_decay=0.002)

    result_file_ja = '/home/ochi/src/data/blue/model.txt'
    result_file = open(result_file_ja, 'w', encoding="utf-8")

    for i in range(len(output_input_lines)):
        output_input_line = output_input_lines[i].split()
        result = output(model.encoder, model.decoder, output_input_line)
        print("出力データ ", ' '.join(result).strip())
        if i == (len(output_input_lines) - 1):
            result_file.write(' '.join(result).strip())
        else:
            result_file.write(' '.join(result).strip() + '\n')
    result_file.close