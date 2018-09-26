import torch
from torch import tensor as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
import time
from get_data import *
import torch.optim as optim

# train_num, hidden_size, batch_size = 20000, 256, 50
train_num, hidden_size, batch_size = 5, 256, 5

input_vocab , input_lines, input_lines_number = {}, {}, {}
target_vocab ,target_lines ,target_lines_number = {}, {}, {}
translate_words = {}

# padddingで0を入れるから
get_train_data_input(train_num, input_vocab, input_lines_number, input_lines)
ev = len(input_vocab) + 1

get_train_data_target(train_num, target_vocab, target_lines_number, target_lines, translate_words)
jv = len(target_vocab) + 1

class Encoder_Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Encoder_Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embed_input = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.drop_input = nn.Dropout(p=0.2)
        self.lstm_input = nn.LSTMCell(hidden_size, hidden_size)

        self.embed_target = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=0.2)
        self.lstm_target = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

        self.attention_linear = nn.Linear(hidden_size * 2, hidden_size)

    def create_mask(self, input_sentence_words):
        return torch.cat( [ input_sentence_words.unsqueeze(-1) ] * 256, 1)

    def forward(self, input_lines ,target_lines):
        hs = torch.zeros(batch_size, self.hidden_size).cuda()
        cs = torch.zeros(batch_size, self.hidden_size).cuda()
        list_hs = []
        for input_sentence_words in input_lines:
            before_hs = hs
            before_cs = cs

            input_k = self.embed_input(input_sentence_words)
            input_k = self.drop_input(input_k)

            hs, cx = self.lstm_input(input_k, (hs, cs) )

            mask = self.create_mask(input_sentence_words)
            hs = torch.where(mask == 0, before_hs, hs)
            cs = torch.where(mask == 0, before_cs, cs)

            list_hs.append(hs)
        list_hs = torch.stack(list_hs, 0)

        max_num =  len(target_lines) # paddingの数
        target_lines_not_last = target_lines[:(max_num-1)]
        target_lines_next = target_lines[1:]
        loss = 0

        ht = hs
        for target_sentence_words , target_sentence_words_next in zip(target_lines_not_last, target_lines_next):
            target_k = self.embed_target(target_sentence_words)
            target_k = self.drop_target(target_k)
            ht, cx = self.lstm_target(target_k, (ht, cx) )

            a_t = F.softmax( (ht * list_hs).sum(-1,keepdim=True), 0 )
            d = (a_t * list_hs).sum(0)
            concat  = torch.cat((d, ht), 1)
            ht_new = F.tanh(self.attention_linear(concat))
            
            loss += F.cross_entropy(self.linear(ht_new), target_sentence_words_next, ignore_index=0)
        return loss

model = Encoder_Decoder(ev, jv, hidden_size)
optimizer = torch.optim.Adam(model.parameters())
device = torch.device('cuda:0')
model = model.to(device)

start = time.time()

epoch_num = 1
for epoch in range(epoch_num):
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

    # if (epoch + 1) % 5 == 0:
    #     outfile = "attention_mt-" + str(epoch + 1) + ".model"
    #     torch.save(model.state_dict(), outfile)
    # elapsed_time = time.time() - start
    # print("時間:",elapsed_time / 60.0, "分")