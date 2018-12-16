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
    hx, cx = encoder.initHidden()

    for i in range(len(output_input_line)):
        ## 辞書にある場合は
        if input_vocab.get(output_input_line[i]):
            word_id = torch.tensor([input_vocab[output_input_line[i]]]).cuda()
        else:
            word_id = torch.tensor([ input_vocab["<unk>"] ]).cuda()
        hx, cx = encoder(word_id, hx, cx)
    word_id = torch.tensor( [ target_vocab["<bos>"] ] ).cuda()

    while(int(word_id) != target_vocab['<eos>']):
        if loop >= 50:
            break
        hx, cx = decoder(word_id, hx, cx)

        word_id = torch.tensor([ torch.argmax(decoder.linear(hx), dim=1) ]).cuda()
        loop += 1
        if int(word_id) != target_vocab['<eos>'] and int(word_id) != 0:
            result.append(translate_words[int(word_id)])
    return result

if __name__ == '__main__':
    device = torch.device('cuda:0')

    model = EncoderDecoder(ev, jv, hidden_size).to(device)
    model.load_state_dict(torch.load("model-15.model"))
    model.eval()

    result_file_ja = "text"
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
