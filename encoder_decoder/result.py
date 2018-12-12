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
import os

train_num, hidden_size= 20000, 256
test_num = 100

input_vocab , input_lines, input_lines_number = {}, {}, {}
target_vocab ,target_lines ,target_lines_number = {}, {}, {}
output_input_lines = {}
translate_words = {}

get_train_data_input(train_num, input_vocab, input_lines_number, input_lines)
ev = len(input_vocab) + 1

get_train_data_target(train_num, target_vocab, target_lines_number, target_lines, translate_words)
jv = len(target_vocab) + 1

get_test_data_target(test_num, output_input_lines)

class Encoder_Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Encoder_Decoder, self).__init__()
        self.embed_input = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.embed_target = nn.Embedding(output_size, hidden_size, padding_idx=0)

        self.lstm_input = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm_target = nn.LSTMCell(hidden_size, hidden_size)

        self.linear = nn.Linear(hidden_size, output_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, output_input_line):
        result = []
        hx = torch.zeros(1, self.hidden_size).cuda()
        cx = torch.zeros(1, self.hidden_size).cuda()

        for i in range(len(output_input_line)):
            ## 辞書にある場合は
            if input_vocab.get(output_input_line[i]):
                word_id = torch.tensor([input_vocab[output_input_line[i]]]).cuda()
            else:
                word_id = torch.tensor([ input_vocab["<unk>"] ]).cuda()
            input_k = self.embed_input(word_id)
            hx, cx = self.lstm_input(input_k, (hx, cx) )
        loop = 0
        word_id = torch.tensor( [ target_vocab["<bos>"] ] ).cuda()

        while(int(word_id) != target_vocab['<eos>']):
            if loop >= 50:
                break
            target_k = self.embed_target(word_id)
            hx, cx = self.lstm_target(target_k, (hx, cx) )
            word_id = torch.tensor([ torch.argmax(F.softmax(self.linear(hx), dim=1).data[0]) ]).cuda()
            loop += 1
            if int(word_id) != target_vocab['<eos>'] and int(word_id) != 0:
                result.append(translate_words[int(word_id)])
        return result

model = Encoder_Decoder(ev, jv, hidden_size)
model.load_state_dict(torch.load("last-15.model"))

device = torch.device('cuda:0')
model = model.to(device)

result_file_ja = "text"
result_file = open(result_file_ja, 'w', encoding="utf-8")

## 出力結果を得る
for i in range(len(output_input_lines)):
    output_input_line = output_input_lines[i].split()
    result = model(output_input_line)
    print("出力データ ", ' '.join(result).strip())
    if i == (len(output_input_lines) - 1):
        result_file.write(' '.join(result).strip())
    else:
        result_file.write(' '.join(result).strip() + '\n')
result_file.close
