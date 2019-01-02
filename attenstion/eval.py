import torch
from torch import tensor as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
from model import *
from define_variable import *

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = EncoderDecoder(source_size, target_size, hidden_size).to(device)
    print(model)
    exit()
    model.load_state_dict(torch.load("trained_model/" + args.model_path))
    model.eval()

    device = torch.device('cuda:0')

    data_set = MyDataset(source_data, target_data)
    train_iter = DataLoader(data_set, batch_size=batch_size, collate_fn=data_set.collater)

    model = EncoderDecoder(source_size, target_size, hidden_size)

    result_file = open(args.result_path, 'w', encoding="utf-8")

    for i in range(len(output_input_lines)):
        output_input_line = output_input_lines[i].split()
        result = output(model.encoder, model.decoder, output_input_line)
        print("出力データ ", ' '.join(result).strip())
        if i == (len(output_input_lines) - 1):
            result_file.write(' '.join(result).strip())
        else:
            result_file.write(' '.join(result).strip() + '\n')
    result_file.close
