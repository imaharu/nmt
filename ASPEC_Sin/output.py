import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *

# model
from model import *

# my function
from create_sin_dict import *

# hyperparameter
from define_sin import *

# Other
import time

def result(encoder, decoder, source_doc):
    loop_s = 0
    loop_w = 0
    ew_hx, ew_cx = encoder.w_encoder.initHidden()
    es_hx, es_cx = encoder.s_encoder.initHidden()
    max_dsn =  max([*map(lambda x: len(x), source_doc )])

    for i in range(0, max_dsn):
        ew_hx, ew_cx = es_hx, es_cx
        line = torch.tensor([ x[i] for x in source_doc ]).t().cuda(device=device)
        for word in line:
            ew_hx , ew_cx = encoder.w_encoder(word, ew_hx, ew_cx)
        es_hx, es_cx = encoder.s_encoder(ew_hx, es_hx, es_cx)
    ds_hx, ds_cx = es_hx, es_cx
    word_id = torch.tensor( [ target_vocab["<bos>"] ]).cuda(device=device)

    result_d = []
    flag = 0
    while(1):
        loop_w = 0
        result_s = []
        dw_hx, dw_cx = ds_hx, ds_cx
        while(1):
            dw_hx, dw_cx = decoder.w_decoder(word_id, dw_hx, dw_cx)
            word_id = torch.tensor([ torch.argmax(decoder.w_decoder.linear(dw_hx), dim=1).data[0]]).cuda(device=device)
            word = translate_vocab[int(word_id)]
            if (int(word_id) == target_vocab["<teos>"]):
                break
            if (int(word_id) == target_vocab["<eod>"]):
                flag = 1
            result_s.append(word)
            loop_w += 1
            if loop_w == 50:
                break
        ds_hx, ds_cx = decoder.s_decoder(dw_hx, ds_hx, ds_cx)

        if loop_s == 3:
            break
        elif flag == 1:
            break

        result_d.append("".join(result_s))
        loop_s += 1
    return result_d

if __name__ == '__main__':
    translate_vocab = {v:k for k,v in target_vocab.items()}
    model = HierachicalEncoderDecoder(source_size, target_size, hidden_size).to(device)
    model.load_state_dict(torch.load("work-20.model"))
    model.eval()
    optimizer = torch.optim.Adam( model.parameters(), weight_decay=0.002)
    for doc_num in range(1):
        source_doc = [ [ get_source_doc(test_file, doc_num + 1, source_vocab) ] ]
        source_doc = [  [ s + [ source_vocab["<seos>"] ] for s in t_d ] for t_d in source_doc ]
        for source in source_doc:
            source.append([ source_vocab["<bod>"] ])
        result_d = result(model.encoder, model.decoder, source_doc)
        print(result_d)
