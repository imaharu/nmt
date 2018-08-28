import torch
from torch import tensor as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
import time
import numpy as np
from get_train_data import *

input_path = "/home/ochi/src/data/train/train_clean.txt.en"
target_path = "/home/ochi/src/data/train/train_clean.txt.ja"

train_num, padding_num, demb, batch_size = 200, 50, 256, 10

input_vocab , input_lines, input_lines_number = {}, {}, {}
target_vocab ,target_lines ,target_lines_number = {}, {}, {}
translate_words = {}

get_train_data_input(input_path, train_num, input_vocab, input_lines_number, input_lines)
ev = len(input_vocab)
get_train_data_target(target_path, train_num, target_vocab, target_lines_number, target_lines, translate_words)
jv = len(target_vocab)

def Padding(batch_lines):
    for i in range(len(batch_lines)):
        if i == 0:
            batch_padding = F.pad(tt( [ batch_lines[i] ] ) , (0, padding_num - len(batch_lines[i])), mode='constant', value=-1)
        else:
            k = F.pad(tt( [ batch_lines[i] ] ) , (0, padding_num - len(batch_lines[i])), mode='constant', value=-1)
            batch_padding = torch.cat((batch_padding, k), dim=0)
    return batch_padding

for epoch in range(1):
    print("epoch",epoch)
    indexes = torch.randperm(train_num)
    for i in range(0, train_num, batch_size):

        batch_input_lines = [ input_lines_number[int(index)] for index in indexes[i:i+batch_size]]
        for batch_input_line in batch_input_lines:
            batch_input_line.append(input_vocab['<eos>'])
        batch_input_paddings = Padding(batch_input_lines)
        Transposed_input = batch_input_paddings.t()

        batch_target_lines = [ target_lines_number[int(index)] for index in indexes[i:i+batch_size]]
        for batch_target_line in batch_target_lines:
            batch_target_line.append(target_vocab['<eos>'])
        batch_target_paddings = Padding(batch_target_lines)
        Transposed_target = batch_target_paddings.t()

# model = Model(ev, jv, demb)

# use_gpu = torch.cuda.is_available()
# if use_gpu:
#     print('cuda is available!')

# optimizer = optim.SGD(model.parameters(), lr=0.1)

# start = time.time()
# for epoch in range(1):
#     print("epoch",epoch)
#     indexes = torch.randperm(train_num)
#     print(indexes)
#     for i in range(0, train_num, batch_size):
#         batch_input_lines = [ input_lines_number[int(index)] for index in indexes[i:i+batch_size]]
#         for batch_input_line in batch_input_lines:
#             batch_input_line.append(input_vocab['<eos>'])
#         batch_input_paddings = padding(batch_input_lines)
#         input_reStructured = reStructured(batch_input_paddings)

#         batch_target_lines = [ target_lines_number[int(index)] for index in indexes[i:i+batch_size]]
#         for batch_target_line in batch_target_lines:
#             batch_target_line.append(target_vocab['<eos>'])
#         batch_target_paddings = padding(batch_target_lines)
#         target_reStructured = reStructured(batch_target_paddings)
    
#         loss = model(input_reStructured, target_reStructured)
#         print("loss",loss)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

    # outfile = "gpu_batch_mt-" + str(epoch) + ".model"
    # serializers.save_npz(outfile, model)
    # elapsed_time = time.time() - start
    # print("時間:",elapsed_time / 60.0, "分")

# class Model(nn.Module):
#     def __init__(self, input_size, output_size, dumb):
#         super(Model, self).__init__()
#         self.embed_input = nn.Embedding(input_size, dumb, padding_idx=-1),
#         self.embed_target = nn.Embedding(output_size, dumb, padding_idx=-1),

#         self.lstm1 = nn.LSTMCell(dumb, dumb),
#         self.linear1 = nn.Linear(dumb, output_size)

#     def forward(self, input_line ,target_line):
#         global accum_loss
#         for input_sentence_words in input_line:
#             input_k = self.embed_input(input_sentence_words)
#             h = self.lstm1(input_k)

#         target_line_not_last = target_line[:(padding_num-1)]
#         target_line_next = target_line[1:]
#         for i ,(target_sentence_words , target_sentence_words_next) in enumerate(zip(target_line_not_last, target_line_next)):
#             if i == 0:
#                 accum_loss = F.softmax_cross_entropy(self.linear1(h), target_sentence_words)
#             target_k = self.embed_target(target_sentence_words)
#             h = self.lstm1(target_k)
#             loss = F.softmax_cross_entropy(self.linear1(h), target_sentence_words_next)
#             accum_loss += loss

#         return accum_loss