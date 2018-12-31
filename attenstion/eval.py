import torch
from torch import tensor as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
from get_data import *
from model import *
from define_variable import *

def output(encoder, decoder, output_input_line):
    result = []
    loop = 0
    hs, cs = encoder.initHidden()
    list_hs = []

    for i in range(len(output_input_line)):
        ## 辞書にある場合は
        if input_vocab.get(output_input_line[i]):
            word_id = torch.tensor([input_vocab[output_input_line[i]]]).cuda()
        else:
            word_id = torch.tensor([ input_vocab["<unk>"] ]).cuda()
        hs, cs = encoder(word_id, hs, cs)
        list_hs.append(hs)

    list_hs = torch.stack(list_hs, 0)
    ht = hs
    ct = cs
    word_id = torch.tensor( [ target_vocab["<bos>"] ] ).cuda()

    while(int(word_id) != target_vocab['<eos>']):
        if loop >= 50:
            break
        ht, ct = decoder(word_id, ht, ct)
        dot = (ht * list_hs).sum(-1, keepdim=True)
        a_t = F.softmax( dot, 0 )
        d = (a_t * list_hs).sum(0)
        concat  = torch.cat((d, ht), 1)
        ht_new = F.tanh(decoder.attention_linear(concat))

        word_id = torch.tensor([ torch.argmax(F.softmax(decoder.linear(ht_new), dim=1).data[0]) ]).cuda()
        loop += 1
        if int(word_id) != target_vocab['<eos>'] and int(word_id) != 0:
            result.append(translate_words[int(word_id)])
    return result

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = EncoderDecoder(ev, jv, hidden_size).to(device)
    model.load_state_dict(torch.load("attention-100000-20.model"))
    model.eval()

    result_file = open("result100000", 'w', encoding="utf-8")

    for i in range(len(output_input_lines)):
        output_input_line = output_input_lines[i].split()
        result = output(model.encoder, model.decoder, output_input_line)
        print("出力データ ", ' '.join(result).strip())
        if i == (len(output_input_lines) - 1):
            result_file.write(' '.join(result).strip())
        else:
            result_file.write(' '.join(result).strip() + '\n')
    result_file.close
