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

train_num, padding_num, hidden_size, batch_size = 200, 50, 128, 10

input_vocab , input_lines, input_lines_number = {}, {}, {}
target_vocab ,target_lines ,target_lines_number = {}, {}, {}
translate_words = {}

get_train_data_input(train_num, input_vocab, input_lines_number, input_lines)
ev = len(input_vocab)
get_train_data_target(train_num, target_vocab, target_lines_number, target_lines, translate_words)
jv = len(target_vocab)

class Encoder_Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Encoder_Decoder, self).__init__()
        self.embed_input = nn.Embedding(input_size, hidden_size, padding_idx=-1)
        self.embed_target = nn.Embedding(output_size, hidden_size, padding_idx=-1)

        self.lstm1 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, output_size)

        self.input_size = input_size
        self.hidden_size =  hidden_size
        self.output_size = output_size

    def forward(self, input_lines ,target_lines):
        global all_loss
        all_loss = 0
        out = torch.zeros(batch_size, self.hidden_size).cuda()
        hidden = torch.zeros(batch_size, self.hidden_size).cuda()
        for input_sentence_words in input_lines:
            input_k = self.embed_input(input_sentence_words)
            out, hidden = self.lstm1(input_k, (out, hidden))
            print("out", out)
            print("hidden", hidden)
        print("------------------------------")
        return all_loss

model = Encoder_Decoder(ev, jv, hidden_size)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
device = torch.device('cuda:0')
model = model.to(device)

start = time.time()
for epoch in range(1):
    print("epoch",epoch)
    indexes = torch.randperm(train_num)
    for i in range(0, train_num, batch_size):

        batch_input_lines = [ input_lines_number[int(index)] for index in indexes[i:i+batch_size]]
        for batch_input_line in batch_input_lines:
            batch_input_line.append(input_vocab['<eos>'])
        batch_input_paddings = Padding(batch_input_lines, padding_num)
        Transposed_input = batch_input_paddings.t().cuda()

        batch_target_lines = [ target_lines_number[int(index)] for index in indexes[i:i+batch_size]]
        for batch_target_line in batch_target_lines:
            batch_target_line.append(target_vocab['<eos>'])
        batch_target_paddings = Padding(batch_target_lines, padding_num)
        Transposed_target = batch_target_paddings.t().cuda()

        model.zero_grad()
        optimizer.zero_grad()
        loss = model(Transposed_input, Transposed_target)
        # loss.backward()
        # optimizer.step()

    # outfile = "gpu_batch_mt-" + str(epoch) + ".model"
    # serializers.save_npz(outfile, model)
    elapsed_time = time.time() - start
    print("時間:",elapsed_time / 60.0, "分")