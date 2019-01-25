import torch
from torch import tensor as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
from model import *
from define import *
from dataset import *
from evaluate_util import *

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = EncoderDecoder(source_size, target_size, hidden_size).to(device)
    state_dict = torch.load('trained_model/' + str(args.model_path))
    model.load_state_dict(state_dict)
    model.eval()
    data_set = EvaluateDataset(generate_source)
    eval_iter = DataLoader(data_set, batch_size=1, collate_fn=data_set.collater)

    Evaluate = Evaluate(target_dict)
    generate_file = open(str(args.result_path), 'w', encoding="utf-8")
    for iters in eval_iter:
        pred = model(source=iters.cuda(), phase=1)
        sentence = Evaluate.TranslateSentence(pred)
        sentence = ' '.join(sentence)
        generate_file.write(sentence + '\n')
    generate_file.close
