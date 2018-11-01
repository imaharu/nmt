import glob
import linecache

def get_dict(language_files, vocab, limit):
    vocab['<unk>'] = len(vocab) + 1
    vocab['<teos>'] = len(vocab) + 1
    vocab['<bod>'] = len(vocab) + 1
    vocab['<eod>'] = len(vocab) + 1
    i = 4
    for file_path in language_files:
        with open(file_path) as f:
            if i == (limit + 4):
                break
            doc = f.read().split()
            for word in doc:
                if word not in vocab:
                    vocab[word] = len(vocab) + 1
        i += 1

def get_source_doc(file_name, vocab_dict ,line_num):
    with open(file_name) as lines:
        next(lines)
        doc_source = []
        doc_source = [ [ vocab_dict[word] for word in line.split()  ] for line in lines ]
        doc_source = [x for x in doc_source if x]
        doc_source
    return doc_source


import torch
def sentence_padding(batch_docs, max_doc_sentence_num):
    for batch_doc in batch_docs:
        if len(batch_doc) != max_doc_sentence_num:
            for i in range(max_doc_sentence_num - len(batch_doc)):
                batch_doc.append([0])
    return batch_docs
