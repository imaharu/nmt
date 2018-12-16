import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import *
import time
import torch.optim as optim
from operator import itemgetter
from define_variable import *

class Encoder_Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Encoder_Decoder, self).__init__()
        self.embed_input = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.drop_input = nn.Dropout(p=0.2)
        self.embed_target = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=0.2)

        self.lstm_input = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm_target = nn.LSTMCell(hidden_size, hidden_size)

        self.linear = nn.Linear(hidden_size, output_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def create_mask(self, input_sentence_words):
        return torch.cat( [ input_sentence_words.unsqueeze(-1) ] * self.hidden_size, 1)

    def forward(self, input_lines, target_lines):
        hx = torch.zeros(batch_size, self.hidden_size).cuda()
        cx = torch.zeros(batch_size, self.hidden_size).cuda()

        for input_sentence_words in input_lines:
            before_hx = hx
            before_cx = cx
            input_k = self.embed_input(input_sentence_words)
            input_k = self.drop_input(input_k)
            hx, cx = self.lstm_input(input_k, (hx, cx) )
            mask = self.create_mask(input_sentence_words)
            hx = torch.where(mask == 0, before_hx, hx)
            cx = torch.where(mask == 0, before_cx, cx)

        max_num =  len(target_lines) # paddingの数
        target_lines_not_last = target_lines[:(max_num-1)]
        target_lines_next = target_lines[1:]
        loss = 0

        for target_sentence_words , target_sentence_words_next in zip(target_lines_not_last, target_lines_next):
            target_k = self.embed_target(target_sentence_words)
            target_k = self.drop_target(target_k)
            hx, cx = self.lstm_target(target_k, (hx, cx) )
            loss += F.cross_entropy(self.linear(hx), target_sentence_words_next, ignore_index=0)
        return loss

model = Encoder_Decoder(ev, jv, hidden_size)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1.0e-6)
device = torch.device('cuda:0')
model = model.to(device)

start = time.time()

for epoch in range(20):
    print("epoch",epoch + 1)
    indexes = torch.randperm(train_num)
    for i in range(0, train_num, batch_size):
        batch_input_lines = [ input_lines_number[int(index)] for index in indexes[i:i+batch_size]]
        batch_input_paddings = Padding(batch_input_lines)
        Transposed_input = batch_input_paddings.t().cuda()

        batch_target_lines = [ target_lines_number[int(index)] for index in indexes[i:i+batch_size]]
        if (epoch + 1) == 1:
            for batch_target_line in batch_target_lines:
                batch_target_line.insert(0, target_vocab['<bos>'])
                batch_target_line.append(target_vocab['<eos>'])

        batch_target_paddings = Padding(batch_target_lines)
        Transposed_target = batch_target_paddings.t().cuda()

        optimizer.zero_grad()
        loss = model(Transposed_input, Transposed_target)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        outfile = "before-" + str(epoch + 1) + ".model"
        torch.save(model.state_dict(), outfile)
    elapsed_time = time.time() - start
    print("時間:",elapsed_time / 60.0, "分")
