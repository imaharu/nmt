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

def train(encoder, decoder, source_doc):
    sentence_hx, sentece_cx = encoder.sentence_encoder.initHidden()
    doc_hx, doc_cx = encoder.doc_encoder.initHidden()

    # for source_sentence in source_doc:

    return 1

if __name__ == '__main__':
    start = time.time()
    #device = torch.device('cuda:0')
    #model = HierachicalEncoderDecoder(source_size, target_size, hidden_size).to(device)
    #model.train()
    #optimizer = torch.optim.Adam( model.parameters(), weight_decay=0.002)
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
            # targets_docs
            max_doc_target_num =  max([*map(lambda x: len(x), target_docs )])
            target_spadding = sentence_padding(target_docs, max_doc_target_num)
            target_wpadding = word_padding(target_spadding, max_doc_target_num)
            # for batch_targets_doc in batch_targets_docs:
            #     batch_targets_doc.insert(0, english_vocab['<bod>'])
            #     batch_targets_doc.append(english_vocab['<teos>'])
            #     batch_targets_doc.append(english_vocab['<eod>'])
            # train(model.encoder, batch_sources_docs, batch_sources_docs)
