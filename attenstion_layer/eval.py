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

def f_eval(encoder, decoder, source_lines):
    result = []
    loop = 0
    list_hx = []
    list_source_mask = []

    lhx_layer = []
    lcx_layer = []
    for i in range(args.layer_num):
        lhx_layer.append(encoder.init())
        lcx_layer.append(encoder.init())
    torch.stack(lhx_layer, 0)
    torch.stack(lcx_layer, 0)
    for word_id in source_lines:
        word_id = torch.tensor( [ word_id ]).cuda()
        lhx_layer, lcx_layer = encoder(word_id, lhx_layer, lcx_layer)
        list_hx.append(lhx_layer[args.layer_num - 1])
        masks = torch.cat( [ word_id.unsqueeze(-1) ] , 1)
        list_source_mask.append( torch.unsqueeze(masks, 0))
    list_hx = torch.stack(list_hx, 0)
    list_source_mask = torch.cat(list_source_mask)

    inf = torch.full((len(source_lines), args.batch_size), float('-inf')).cuda()
    inf = torch.unsqueeze(inf, -1)

    word_id = torch.tensor( [ target_vocab["<bos>"] ] ).cuda()

    while(int(word_id) != target_vocab['<eos>']):
        if loop >= 50:
            break
        lhx_layer, lcx_layer = decoder(word_id, lhx_layer, lcx_layer)
        ht_new = decoder.attention(lhx_layer[args.layer_num - 1], list_hx, list_source_mask, inf)
        word_id = torch.tensor([ torch.argmax(decoder.linear(ht_new), dim=1) ]).cuda()
        loop += 1
        if int(word_id) != target_vocab['<eos>'] and int(word_id) != 0:
            result.append(translate_words[int(word_id)])
    return result

if __name__ == '__main__':
    device = torch.device('cuda:0')

    model = EncoderDecoder(ev, jv, args.embed_size ,args.hidden_size).to(device)
    model.load_state_dict(torch.load("trained_model/" + args.model_path))
    model.eval()
    result_file_ja = "files_evaled/" + args.result_path + ".txt"
    result_file = open(result_file_ja, 'w', encoding="utf-8")
    for i in range(len(output_input_lines)):
        source_line = output_input_lines[i].split()
        source_line = [  input_vocab[word] if word in input_vocab else input_vocab["<unk>"] for word in source_line ]
        result = f_eval(model.encoder, model.decoder, source_line)
        print("出力データ ", ' '.join(result).strip())

        if i == (len(output_input_lines) - 1):
            result_file.write(' '.join(result).strip())
        else:
            result_file.write(' '.join(result).strip() + '\n')

    result_file.close
