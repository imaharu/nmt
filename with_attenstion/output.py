import torch
from torch import tensor as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
import time
from get_data import *
import torch.optim as optim

train_num, padding_num, hidden_size, batch_size = 20000, 50, 256, 50
test_num = 1000

input_vocab , input_lines, input_lines_number = {}, {}, {}
target_vocab ,target_lines ,target_lines_number = {}, {}, {}
output_input_lines = {}
translate_words = {}

get_train_data_input(train_num, input_vocab, input_lines_number, input_lines)
ev = len(input_vocab)

get_train_data_target(train_num, target_vocab, target_lines_number, target_lines, translate_words)
jv = len(target_vocab)

get_test_data_target(test_num, output_input_lines)

class Encoder_Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Encoder_Decoder, self).__init__()
        self.embed_input = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.embed_target = nn.Embedding(output_size, hidden_size, padding_idx=0)

        self.lstm_input = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm_target = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.target_linear = nn.Linear(hidden_size * 2, hidden_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

model = Encoder_Decoder(ev, jv, hidden_size)
model.load_state_dict(torch.load("attention_mt-5.model"))

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
device = torch.device('cuda:0')
model = model.to(device)

def each_score(ht, hs):
    return torch.exp((ht * hs).sum(-1,keepdim=True))

def all_score(ht, list_hs):
    all_score = 0
    for hs in  list_hs:
        all_score += each_score(ht, hs)
    return all_score

def a_t(ht, hs ,list_hs):
    each_score_ = 0
    each_score_ = each_score(ht, hs)
    all_score_ = 0
    all_score_ = all_score(ht, list_hs)
    return each_score_ / all_score_

def d(a_t, list_hs):
    d = 0
    for h in list_hs:
        d += a_t * h
    return d

def output(model, output_input_line):
    result = []

    hs = torch.zeros(1, model.hidden_size).cuda()
    cx = torch.zeros(1, model.hidden_size).cuda()
    list_hs = []

    for i in range(len(output_input_line)):
        ## 辞書にある場合は        
        if input_vocab.get(output_input_line[i]):
            wid = torch.tensor([input_vocab[output_input_line[i]]]).cuda()
        else:
            ## TODO:ない場合は<unk>にしたい -> 学習時点から
            wid = torch.tensor([ input_vocab[","] ]).cuda()
        # print(wid)
        input_k = model.embed_input(wid)
        hs, cx = model.lstm_input(input_k, (hs, cx) )
        list_hs.append(hs)

    loop = 0
    wid = torch.tensor([ torch.argmax(F.softmax(model.linear(hs), dim=1).data[0]) ]).cuda()
    i = 0
    while(int(wid) != target_vocab['<eos>']) and (loop <= 50):
        target_k = model.embed_target(wid)
        if i == 0:
            ht, cx = model.lstm_target(target_k, (hs, cx) )
            i += 1
        else:
            ht, cx = model.lstm_target(target_k, (ht, cx) )
        ## TODO:dimの検討
        a_ = 0
        print(loop)
        print(list_hs[loop])
        a_ = a_t(ht, list_hs[loop] ,list_hs)
        d_ = 0
        d_ = d(a_, list_hs)
        concat_ = 0
        concat_  = torch.cat((d_, ht), 1)
        ht_new = F.tanh(model.target_linear(concat_))
        wid = torch.tensor([ torch.argmax(F.softmax(model.linear(ht_new), dim=1).data[0]) ]).cuda()
        loop +=1
        if int(wid) != target_vocab['<eos>']:
            result.append(translate_words[int(wid)])
    return result

result_file_ja = '/home/ochi/src/data/blue/torch_result.txt'
result_file = open(result_file_ja, 'w', encoding="utf-8")

## 出力結果を得る
for i in range(len(output_input_lines)):
    output_input_line = output_input_lines[i].split()
    print(output_input_line)
    result = output(model, output_input_line)
    print("出力データ {}", ' '.join(result).strip())
    # if i == (len(output_input_lines) - 1):
    #     result_file.write(' '.join(result).strip())
    # else:
    #     result_file.write(' '.join(result).strip() + '\n')
result_file.close

# blue_correct_ja = open("/home/ochi/src/data/blue/torch_correct_ja.txt", 'w', encoding="utf-8")
# path_test_ja = "/home/ochi/src/data/test/test_clean.txt.ja"

# with open(path_test_ja,'r',encoding='utf-8') as f:
#     lines = f.read().strip().split('\n')
#     i = 0
#     for line in lines:
#         i += 1
#         if i == test_num:
#             blue_correct_ja.write(line.strip())
#             break
#         else:
#             blue_correct_ja.write(line.strip() + '\n')

# blue_correct_ja.close