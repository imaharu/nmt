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
    loop_d = 0
    ew_hx, ew_cx = encoder.w_encoder.initHidden()
    es_hx, es_cx = encoder.s_encoder.initHidden()
    max_dsn =  max([*map(lambda x: len(x), source_doc )])
    for i in range(0, max_dsn):
        line = torch.tensor([ x[i] for x in source_doc ]).t().cuda(device=device)
        for word in line:
            ew_hx , ew_cx = encoder.w_encoder(word, ew_hx, ew_cx)
        es_hx , es_cx = encoder.s_encoder(ew_hx, es_hx, es_cx)

    ds_hx, ds_cx = es_hx, es_cx
    #word_id = torch.tensor( [ english_vocab["<bod>"] ]).cuda(device=device)
    word_id = 0
    result_d = ""
    print("--------------------")
    while(int(word_id) != english_vocab["<eod>"] ):
        loop_w = 0
        result_s = []
        dw_hx, dw_cx = ds_hx, ds_cx
        if loop_s >= 10:
            break
        while(1):
            if loop_w >= 5:
                word_id = torch.tensor( [ english_vocab["<teos>"] ]).cuda(device=device)
                dw_hx, dw_cx = decoder.w_decoder(word_id, dw_hx, dw_cx)
                break
            #dw_hx, dw_cx = decoder.w_decoder(word_id, dw_hx, dw_cx)
            word_id = torch.tensor([ torch.argmax(decoder.w_decoder.linear(dw_hx), dim=1).data[0]]).cuda(device=device)
            dw_hx, dw_cx = decoder.w_decoder(word_id, dw_hx, dw_cx)
            word = [k for k, v in english_vocab.items() if v == word_id ]

            if (int(word_id) == english_vocab["<teos>"]):
                break
            result_s.append(word)
            loop_w += 1
        loop_s += 1
        print("s: ", result_s)
        ds_hx, ds_cx = decoder.s_decoder(dw_hx, ds_hx, ds_cx)
    return result_d

if __name__ == '__main__':
    model = HierachicalEncoderDecoder(source_size, target_size, hidden_size).to(device)
    model.load_state_dict(torch.load("19990-8.model"))
    model.eval()
    optimizer = torch.optim.Adam( model.parameters(), weight_decay=0.002)
    for i in range(len(english_paths)):
        source_doc = [ get_source_doc(english_paths[i], english_vocab) ]
        source_doc = [ [ s + [ english_vocab["<teos>"] ] for s in t_d ] for t_d in source_doc]
        for source in source_doc:
            source.append([ english_vocab["<bod>"] ])
        result_doc = result(model.encoder, model.decoder, source_doc)
