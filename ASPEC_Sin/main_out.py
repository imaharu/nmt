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
    ew_hx, ew_cx = encoder.w_encoder.initHidden()
    max_dsn =  max([*map(lambda x: len(x), source_doc )])
    for i in range(0, max_dsn):
        line = torch.tensor([ x[i] for x in source_doc ]).t().cuda(device=device)
        for word in line:
            ew_hx , ew_cx = encoder.w_encoder(word, ew_hx, ew_cx)

    dw_hx, dw_cx = ew_hx, ew_cx
    word_id = torch.tensor( [ target_vocab["<bos>"] ]).cuda(device=device)
    result_s = []
    while(1):
        dw_hx, dw_cx = decoder.w_decoder(word_id, dw_hx, dw_cx)
        word_id = torch.tensor([ torch.argmax(decoder.w_decoder.linear(dw_hx), dim=1).data[0]]).cuda(device=device)
        word = translate_vocab[int(word_id)]

        if (int(word_id) == target_vocab["<teos>"]):
            break
        result_s.append(word)
        loop_s += 1
        if loop_s == 50:
            break
    return result_s

if __name__ == '__main__':
    translate_vocab = {v:k for k,v in target_vocab.items()}
    model = HierachicalEncoderDecoder(source_size, target_size, hidden_size).to(device)
    model.load_state_dict(torch.load("bos-10.model"))
    model.eval()
    optimizer = torch.optim.Adam( model.parameters(), weight_decay=0.002)
    for doc_num in range(5):
        source_docs = [ [ get_source_doc(test_file, doc_num + 1, source_vocab) ] ]
        source_docs = [  [ s + [ source_vocab["<seos>"] ] for s in t_d ] for t_d in source_docs ]
        result_s = result(model.encoder, model.decoder, source_docs)
        print(result_s)
