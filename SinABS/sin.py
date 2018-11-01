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
    device = torch.device('cuda:0')
    model = HierachicalEncoderDecoder(source_size, target_size, hidden_size).to(device)
    model.train()
    optimizer = torch.optim.Adam( model.parameters(), weight_decay=0.002)
    for epoch in range(1):
        batch_targets_docs = []
        batch_sources_docs = []
        print("epoch",epoch + 1)
        indexes = torch.randperm(train_doc_num)
        for i in range(0, train_doc_num, batch_size):
            print(111)
            # targets_docs
            batch_sources_docs = [ get_source_doc(english_paths[doc_num] , english_vocab, 2) for doc_num in indexes[i:i+batch_size] ]
            max_doc_word_num = max([*map(lambda x: max(x), [ [ *map(lambda x: len(x), sentence) ]for sentence in batch_sources_docs])])
            max_doc_sentence_num =  max([*map(lambda x: len(x), batch_sources_docs)])
            source_padding = sentence_padding(batch_sources_docs, max_doc_sentence_num)
            # targets_docs
            # batch_targets_docs = [ [ english_vocab[i] for i in get_target_doc(english_paths[doc_num]).split() ] for doc_num in indexes[i:i+batch_size] ]
            # for batch_targets_doc in batch_targets_docs:
            #     batch_targets_doc.insert(0, english_vocab['<bod>'])
            #     batch_targets_doc.append(english_vocab['<teos>'])
            #     batch_targets_doc.append(english_vocab['<eod>'])
            
            # train(model.encoder, batch_sources_docs, batch_sources_docs)
