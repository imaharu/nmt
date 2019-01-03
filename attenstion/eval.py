import torch
from torch import tensor as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
from model import *
from define_variable import *
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

    data_set = EvaluateDataset(source_data)
    train_iter = DataLoader(data_set, batch_size=1, collate_fn=data_set.collater)
