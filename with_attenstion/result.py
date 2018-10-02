import torch
from torch import tensor as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
import time
from get_data import *
import torch.optim as optim

train_num, hidden_size = 20000, 256
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

    def forward(self, output_input_line):
        result = []

        hs = torch.zeros(1, self.hidden_size).cuda()
        cs = torch.zeros(1, self.hidden_size).cuda()
        list_hs = []

        for i in range(len(output_input_line)):
            ## 辞書にある場合は        
            if input_vocab.get(output_input_line[i]):
                word_id = torch.tensor([input_vocab[output_input_line[i]]]).cuda()
            else:
                ## TODO:ない場合は<unk>にしたい -> 学習時点から
                word_id = torch.tensor([ input_vocab["<unk>"] ]).cuda()

            input_k = self.embed_input(word_id)
            hs, cs = self.lstm_input(input_k, (hs, cs) )
            list_hs.append(hs)
    
        list_hs = torch.stack(list_hs, 0)

        ht = hs
        ct = cs
        loop = 0
        word_id = torch.tensor( [ target_vocab["<bos>"] ] ).cuda()

        while(int(word_id) != target_vocab['<eos>']) and (loop <= 50):
            if loop >= 50:
                break
            target_k = self.embed_target(word_id)
            ht, ct = self.lstm_target(target_k, (ht, ct) )

            dot = (ht * list_hs).sum(-1, keepdim=True)
            a_t = F.softmax( dot, 0 )
            d = (a_t * list_hs).sum(0)
            concat  = torch.cat((d, ht), 1)
            ht_new = F.tanh(self.attention_linear(concat))

            word_id = torch.tensor([ torch.argmax(F.softmax(self.linear(ht_new), dim=1).data[0]) ]).cuda()
            loop += 1
            if int(word_id) != target_vocab['<eos>'] and int(word_id) != 0:
                result.append(translate_words[int(word_id)])
        return result

model = Encoder_Decoder(ev, jv, hidden_size)
model.load_state_dict(torch.load("attention_mt-15.model"))

optimizer = torch.optim.Adam(model.parameters())
device = torch.device('cuda:0')
model = model.to(device)

result_file_ja = '/home/ochi/src/data/blue/attention_result.txt'
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