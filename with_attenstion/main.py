import torch
from torch import tensor as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
import time
from get_data import *
import torch.optim as optim
import math

train_num, padding_num, hidden_size, batch_size = 5, 50, 256, 5

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
        self.embed_input = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.embed_target = nn.Embedding(output_size, hidden_size, padding_idx=0)

        self.lstm1 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, output_size)
        self.target_linear1 = nn.Linear(hidden_size + hidden_size, output_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def each_score(self, ht, hs):
        bmm = torch.bmm(torch.unsqueeze(ht.t(), 0), torch.unsqueeze(hs, 0))
        bmm_squeeze = torch.squeeze(bmm, 0)
        return torch.exp(bmm_squeeze)

    def all_score(self, ht, list_hs):
        all_score = 0
        for hs in  list_hs:
            all_score += self.each_score(ht, hs)
        return all_score

    def a_t(self, ht, hs ,list_hs):
        each_score = self.each_score(ht, hs)
        all_score = self.all_score(ht, list_hs)
        return each_score / all_score

    def forward(self, input_lines ,target_lines):
        global all_loss

        hs = torch.zeros(batch_size, self.hidden_size).cuda()
        cx = torch.zeros(batch_size, self.hidden_size).cuda()
        ct = 0
        list_hs = []
        for input_sentence_words in input_lines:
            input_k = self.embed_input(input_sentence_words)
            hs, cx = self.lstm1(input_k, (hs, cx) )
            ct += hs
            list_hs.append(hs)

        target_lines_not_last = target_lines[:(padding_num-1)]
        target_lines_next = target_lines[1:]
        for i ,(target_sentence_words , target_sentence_words_next) in enumerate(zip(target_lines_not_last, target_lines_next)):
            if i == 0:
                accum_loss = F.cross_entropy(self.linear1(hs), target_sentence_words)
                ht = hs
            target_k = self.embed_target(target_sentence_words)
            ht, cx = self.lstm1(target_k, (ht, cx) )
            at = self.a_t(ht, list_hs[i] ,list_hs)
            loss = 0
            break
            # loss = F.cross_entropy(self.linear1(ht), target_sentence_words_next)
            accum_loss += loss

        return accum_loss

model = Encoder_Decoder(ev, jv, hidden_size)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
device = torch.device('cuda:0')
model = model.to(device)

start = time.time()

epoch_num = 1
for epoch in range(epoch_num):
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

        optimizer.zero_grad()
        loss = model(Transposed_input, Transposed_target)
        loss.backward()
        optimizer.step()

    # if (epoch + 1) % 5 == 0:
    #     outfile = "gpu_batch_mt-" + str(epoch) + ".model"
    #     torch.save(model.state_dict(), outfile)
    elapsed_time = time.time() - start
    print("時間:",elapsed_time / 60.0, "分")