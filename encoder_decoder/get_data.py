def get_train_data_input(train_num, input_vocab, input_lines_number, input_lines):
    with open("/home/ochi/src/data/train/train_clean.txt.en",'r',encoding='utf-8') as f:
        lines_en = f.read().strip().split('\n')
        i = 0
        for line in lines_en:
            if i == train_num:
                break
            for input_word in line.split():
                if input_word not in input_vocab:
                    input_vocab[input_word] = len(input_vocab)
            input_lines_number[i] = [input_vocab[word] for word in line.split()]
            input_lines[i] = line
            i += 1
        input_vocab['<eos>'] = len(input_vocab)

def get_train_data_target(train_num, target_vocab, target_lines_number, target_lines, translate_words):
    with open("/home/ochi/src/data/train/train_clean.txt.ja",'r',encoding='utf-8') as f:
        lines_ja = f.read().strip().split('\n')
        i = 0
        for line in lines_ja:
            if i == train_num:
                break
            for target_word in line.split():
                if target_word not in target_vocab:
                    id = len(target_vocab)
                    target_vocab[target_word] = len(target_vocab)
                    translate_words[id] = target_word
            target_lines_number[i] = [target_vocab[word] for word in line.split()]
            target_lines[i] = line
            i += 1

        id = len(target_vocab)
        target_vocab['<eos>'] = id
        translate_words[id] = "<eos>"

# Only Pytorch
import torch.nn.functional as F
from torch import tensor as tt
import torch

def Padding(batch_lines, padding_num):
    for i in range(len(batch_lines)):
        if i == 0:
            batch_padding = F.pad(tt( [ batch_lines[i] ] ) , (0, padding_num - len(batch_lines[i])), mode='constant', value=-1)
        else:
            k = F.pad(tt( [ batch_lines[i] ] ) , (0, padding_num - len(batch_lines[i])), mode='constant', value=-1)
            batch_padding = torch.cat((batch_padding, k), dim=0)
    return batch_padding