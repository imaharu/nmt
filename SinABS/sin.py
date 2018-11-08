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

def train(encoder, decoder, source_doc, target_doc):
    s_hx, s_cx = encoder.s_encoder.initHidden()
    d_hx, d_cx = encoder.d_encoder.initHidden()
    max_dsn =  max([*map(lambda x: len(x), source_docs )])
    max_dtn =  max([*map(lambda x: len(x), target_docs )])
    for i in range(0, max_dsn):
        lines = torch.tensor([ x[i]  for x in source_doc ]).t().cuda()
        #    s_hx , s_cx = encoder.s_encoder(source_w)
    return 1

if __name__ == '__main__':
    start = time.time()
    device = torch.device('cuda:0')
    model = HierachicalEncoderDecoder(source_size, target_size, hidden_size).to(device)
    model.train()
    optimizer = torch.optim.Adam( model.parameters(), weight_decay=0.002)
    for epoch in range(1):
        target_docs = []
        source_docs = []
        print("epoch",epoch + 1)
        indexes = torch.randperm(train_doc_num)
        for i in range(0, train_doc_num, batch_size):
            source_docs = [ get_source_doc(english_paths[doc_num], english_vocab) for doc_num in indexes[i:i+batch_size]]
            target_docs = [ get_target_doc(english_paths[doc_num], english_vocab) for doc_num in indexes[i:i+batch_size]]
            # source_docs
            max_doc_sentence_num =  max([*map(lambda x: len(x), source_docs )])
            source_spadding = sentence_padding(source_docs, max_doc_sentence_num)
            source_wpadding = word_padding(source_spadding, max_doc_sentence_num)
            max_doc_target_num =  max([*map(lambda x: len(x), target_docs )])
            # add <teos> to target_docs
            target_docs = [ [ s + [ english_vocab["<teos>"] ] for s in t_d ] for t_d in target_docs]
            target_spadding = sentence_padding(target_docs, max_doc_target_num)
            target_wpadding = word_padding(target_spadding, max_doc_target_num)
            for target in target_wpadding:
                target.insert(0, [ english_vocab["<bod>"] ])
                target.append([english_vocab["<eod>"]])
            train(model.encoder, model.decoder, source_wpadding,target_wpadding)
