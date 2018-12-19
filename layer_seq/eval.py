import torch
from torch import tensor as tt
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
import time
import torch.optim as optim
from operator import itemgetter
from define_variable import *
from model import *

def output(encoder, decoder, output_input_line):
    result = []
    loop = 0
    lhx_layer = []
    lcx_layer = []
    for i in range(layer_num):
        lhx_layer.append(encoder.init())
        lcx_layer.append(encoder.init())
    torch.stack(lhx_layer, 0)

    for i in range(len(output_input_line)):
        ## 辞書にある場合は
        if input_vocab.get(output_input_line[i]):
            word_id = torch.tensor([input_vocab[output_input_line[i]]]).cuda()
        else:
            word_id = torch.tensor([ input_vocab["<unk>"] ]).cuda()
        lhx_layer, lcx_layer = encoder(word_id, lhx_layer, lcx_layer)
    word_id = torch.tensor( [ target_vocab["<bos>"] ] ).cuda()

    while(int(word_id) != target_vocab['<eos>']):
        if loop >= 50:
            break
        lhx_layer, lcx_layer = decoder(word_id, lhx_layer, lcx_layer)

        word_id = torch.tensor([ torch.argmax(decoder.linear(lhx_layer[layer_num - 1]), dim=1) ]).cuda()
        loop += 1
        if int(word_id) != target_vocab['<eos>'] and int(word_id) != 0:
            print(translate_words[int(word_id)])
            result.append(translate_words[int(word_id)])
    return result

if __name__ == '__main__':
    device = torch.device('cuda:0')

    model = EncoderDecoder(ev, jv, hidden_size).to(device)
    model.load_state_dict(torch.load("layer-10.model"))
    model.eval()

    result_file_ja = "layer.txt"
    result_file = open(result_file_ja, 'w', encoding="utf-8")

    for i in range(len(output_input_lines)):
        output_input_line = output_input_lines[i].split()
        result = output(model.encoder, model.decoder, output_input_line)
        print("出力データ ", ' '.join(result).strip())
        if i == (len(output_input_lines) - 1):
            result_file.write(' '.join(result).strip())
        else:
            result_file.write(' '.join(result).strip() + '\n')
    result_file.close
