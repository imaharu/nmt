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
    es_hx_list = []
    es_mask = []
    ew_hx, ew_cx = encoder.w_encoder.initHidden()
    es_hx, es_cx = encoder.s_encoder.initHidden()
    max_dsn =  max([*map(lambda x: len(x), source_doc )])

    for i in range(0, max_dsn):
        ew_hx, ew_cx = es_hx, es_cx
        line = torch.tensor([ x[i] for x in source_doc ]).t().cuda(device=device)
        for word in line:
            ew_hx , ew_cx = encoder.w_encoder(word, ew_hx, ew_cx)
        es_hx, es_cx = encoder.s_encoder(ew_hx, es_hx, es_cx)
        es_hx_list.append(es_hx)
    ds_hx, ds_cx = es_hx, es_cx

    es_hx_list = torch.stack(es_hx_list, 0)
    result_d = []
    flag = 0
    while(1):
        loop_w = 0
        result_s = []
        dw_hx, dw_cx = ds_hx, ds_cx
        word_id = torch.tensor( [ english_vocab["<bos>"] ]).cuda(device=device)
        while(1):
            dw_hx, dw_cx = decoder.w_decoder(word_id, dw_hx, dw_cx)
            word_id = torch.tensor([ torch.argmax(decoder.w_decoder.linear(dw_hx), dim=1).data[0]]).cuda(device=device)
            word = translate_vocab[int(word_id)]
            if (int(word_id) == english_vocab["<teos>"]):
                break
            if (int(word_id) == english_vocab["<eod>"]):
                flag = 1
                break
            result_s.append(word)
            loop_w += 1
            if loop_w == 50:
                break
        ds_hx, ds_cx = decoder.s_decoder(dw_hx, ds_hx, ds_cx)
        dot = (ds_hx * es_hx_list).sum(-1, keepdim=True)
        a_t = F.softmax(dot, 0)
        d = (a_t * es_hx_list).sum(0)
        concat = torch.cat((d, ds_hx), 1)
        ds_hx = F.tanh(decoder.s_decoder.attention_linear(concat))
        if loop_s == 5:
            break
        if flag == 1:
            break
        result_d.append(" ".join(result_s))
        loop_s += 1
    return result_d

if __name__ == '__main__':
    translate_vocab = {v:k for k,v in english_vocab.items()}
    model = HierachicalEncoderDecoder(source_size, target_size, hidden_size).to(device)
    model.load_state_dict(torch.load("models/19500-30.model"))
    model.eval()
    optimizer = torch.optim.Adam( model.parameters(), weight_decay=0.002)
    test_path = os.environ["cnn_unk"] + "/test"
    result_paths = sorted(glob.glob(test_path + "/*.story"))[0:5]
    for doc_num in range(2):
        source_doc = [ get_source_doc(result_paths[doc_num], english_vocab) ]
        source_doc = [  [ s + [ english_vocab["<seos>"] ] for s in t_d ] for t_d in source_doc ]
        for source in source_doc:
            source.append([ english_vocab["<eod>"] ])
        result_d = result(model.encoder, model.decoder, source_doc)
        print(result_d)
