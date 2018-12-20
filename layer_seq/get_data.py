def get_train_data_input(train_en_path, train_num, input_vocab, input_lines_number, input_lines):
    with open(train_en_path,'r',encoding='utf-8') as f:
        lines_en = f.read().strip().split('\n')
        i = 0
        for line in lines_en:
            if i == train_num:
                break
            for input_word in line.split():
                if input_word not in input_vocab:
                    input_vocab[input_word] = len(input_vocab) + 1
            input_lines_number[i] = [input_vocab[word] for word in line.split()]
            input_lines[i] = line
            i += 1
        input_vocab['<unk>'] = len(input_vocab) + 1
def get_train_data_target(train_ja_path, train_num, target_vocab, target_lines_number, target_lines, translate_words):
    with open(train_ja_path,'r',encoding='utf-8') as f:
        lines_ja = f.read().strip().split('\n')
        i = 0
        for line in lines_ja:
            if i == train_num:
                break
            for target_word in line.split():
                if target_word not in target_vocab:
                    id = len(target_vocab) + 1
                    target_vocab[target_word] = len(target_vocab) + 1
                    translate_words[id] = target_word
            target_lines_number[i] = [target_vocab[word] for word in line.split()]
            target_lines[i] = line
            i += 1

        id = len(target_vocab)
        target_vocab['<bos>'] = len(target_vocab) + 1
        target_vocab['<eos>'] = len(target_vocab) + 1
        translate_words[id + 1] = "<bos>"
        translate_words[id + 2] = "<eos>"

def get_test_data_target(test_num, test_input_lines):
    with open("../train_data/test.en",'r',encoding='utf-8') as f:
        lines_en = f.read().strip().split('\n')
        i = 0
        for line in lines_en:
            if i == test_num:
                break
            test_input_lines[i] = line
            i += 1

# Only Pytorch
import torch.nn.functional as F
from torch import tensor as tt
import torch

def Padding(batch_lines):
    max_num =  max([*map(lambda x: len(x), batch_lines)])
    for i in range(len(batch_lines)):
        if i == 0:
            batch_padding = F.pad(tt( [ batch_lines[i] ] ) , (0, max_num - len(batch_lines[i])), mode='constant', value=0)
        else:
            k = F.pad(tt( [ batch_lines[i] ] ) , (0, max_num - len(batch_lines[i])), mode='constant', value=0)
            batch_padding = torch.cat((batch_padding, k), dim=0)
    return batch_padding
