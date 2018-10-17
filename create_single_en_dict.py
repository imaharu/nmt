import glob
import linecache

def get_summary(file_path, line_num):
    return linecache.getline(file_path, line_num, module_globals=None).replace('\n','')

def get_dict(language_files, vocab, limit):
    vocab['<unk>'] = len(vocab) + 1
    vocab['<seos>'] = len(vocab) + 1
    vocab['<tbos>'] = len(vocab) + 1
    vocab['<teos>'] = len(vocab) + 1
    vocab['<bod>'] = len(vocab) + 1
    vocab['<eod>'] = len(vocab) + 1
    i = 6
    for file_path in language_files:
        with open(file_path) as f:
            if i == (limit + 6):
                break
            doc = f.read().split()
            for word in doc:
                if word not in vocab:
                    vocab[word] = len(vocab) + 1
        i += 1

def get_source_doc(file_name, vocab_dict ,line_num):
    with open(file_name) as lines:
        next(lines)
        doc_target = []
        doc_target = [ [ vocab_dict[word] for word in line.split()  ] for line in lines ]
        doc_target = [x for x in doc_target if x]
        doc_target
    return doc_target

def get_target_doc(file_name):
    return get_summary(file_name, 1)

import torch
def sentence_padding(batch_docs, max_doc_sentence_num):
    for batch_doc in batch_docs:
        # for sources_sentence in batch_sources_doc:
        #     for i in range(max_doc_word_num - len(sources_sentence)):
        #         sources_sentence.append(0)
        if len(batch_doc) != max_doc_sentence_num:
            for i in range(max_doc_sentence_num - len(batch_doc)):
                batch_doc.append([0])
    return batch_docs