import torch
from torch import tensor as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
from model import *
from define_variable import *
from dataset import *
from evaluate_util import *
from collections import OrderedDict

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = EncoderDecoder(source_size, target_size, hidden_size).to(device)
    state_dict = torch.load('trained_model/010320.model')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    data_set = EvaluateDataset(test_source)
    eval_iter = DataLoader(data_set, batch_size=1, collate_fn=data_set.collater)

    Evaluate = Evaluate(target_dict)
    pred_file = open("pred.txt", 'w', encoding="utf-8")
    for iters in eval_iter:
        pred = model(source=iters.cuda(), phase=1)
        sentence = Evaluate.TranslateSentence(pred)
        sentence = ' '.join(sentence)
        pred_file.write(sentence + '\n')
    pred_file.close
