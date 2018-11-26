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

def create_mask(words):
    return torch.cat( [ words.unsqueeze(-1) ] * hidden_size, 1)

def train(encoder, decoder, source_doc, target_doc):
    loss = 0
    ew_hx, ew_cx = encoder.w_encoder.initHidden()
    max_dsn =  max([*map(lambda x: len(x), source_docs )])
    max_dtn =  max([*map(lambda x: len(x), target_docs )])
    for i in range(0, max_dsn):
        lines = torch.tensor([ x[i]  for x in source_doc ]).t().cuda(device=device)
        for words in lines:
            before_ew_hx , before_ew_cx = ew_hx , ew_cx
            ew_hx , ew_cx = encoder.w_encoder(words, ew_hx, ew_cx)
            w_mask = create_mask(words)
            ew_hx = torch.where(w_mask == 0, before_ew_hx, ew_hx)
            ew_cx = torch.where(w_mask == 0, before_ew_cx, ew_cx)

    dw_hx, dw_cx = ew_hx, ew_cx
    for i in range(0, max_dtn):
        lines = torch.tensor([ x[i]  for x in target_doc ]).t().cuda(device=device)
        # t -> true, f -> false
        lines_t_last = lines[1:]
        lines_f_last = lines[:(len(lines) - 1)]
        for words_f, words_t in zip(lines_f_last, lines_t_last):
            before_dw_hx, before_dw_cx = dw_hx, dw_cx
            dw_hx , dw_cx = decoder.w_decoder(words_f, dw_hx, dw_cx)
            w_mask = create_mask(words_f)
            dw_hx = torch.where(w_mask == 0, before_dw_hx, dw_hx)
            dw_cx = torch.where(w_mask == 0, before_dw_cx, dw_cx)
            loss += F.cross_entropy(decoder.w_decoder.linear(dw_hx), words_t , ignore_index=0)
    return loss

if __name__ == '__main__':
    start = time.time()
    model = HierachicalEncoderDecoder(source_size, target_size, hidden_size).to(device)
    model.train()
    optimizer = torch.optim.Adam( model.parameters(), weight_decay=0.002)

    for epoch in range(10):
        target_docs = []
        source_docs = []
        print("epoch",epoch + 1)
        indexes = torch.randperm(train_doc_num)
        for i in range(0, train_doc_num, batch_size):
            source_docs = [ [ get_source_doc(sfn, doc_num + 1, source_vocab) ] for doc_num in indexes[i:i+batch_size]]
            target_docs = [ [ get_target_doc(tfn, doc_num + 1, target_vocab) ] for doc_num in indexes[i:i+batch_size]]
            # source_docs
            max_doc_sentence_num =  max([*map(lambda x: len(x), source_docs )])
            source_docs = [  [ s + [ source_vocab["<seos>"] ] for s in t_d ] for t_d in source_docs ]
            source_wpadding = word_padding(source_docs, max_doc_sentence_num)

            max_doc_target_num =  max([*map(lambda x: len(x), target_docs )])
            # add <teos> to target_docs

            target_docs = [ [ [target_vocab["<bos>"] ] + s + [ target_vocab["<teos>"] ] for s in t_d ] for t_d in target_docs]
            #target_docs = [ [ s + [ target_vocab["<teos>"] ] for s in t_d ] for t_d in target_docs]

            target_wpadding = word_padding(target_docs, max_doc_target_num)

            optimizer.zero_grad()
            loss = train(model.encoder, model.decoder, source_wpadding,target_wpadding)
            loss.backward()
            optimizer.step()

        if (epoch + 1)  % 10 == 0:
            outfile = "bos-" + str(epoch + 1) + ".model"
            torch.save(model.state_dict(), outfile)
        elapsed_time = time.time() - start
        print("時間:",elapsed_time / 60.0, "分")
