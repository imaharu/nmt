import glob
import linecache
from cnn_data import *

def count_dict(language_files, c_vocab):
    for filename in language_files:
        story_lines = [ line.split() for line in separate_source_data(filename) ]
        highlights_lines = [ line.split() for line in separate_target_data(filename) ]
        for lines in story_lines:
            for word in lines:
                if word in c_vocab:
                    c_vocab[word] += 1
                else:
                    c_vocab[word] = 1
    print(c_vocab)
    return 1

def get_dict(language_files, vocab):
    vocab['<unk>'] = len(vocab) + 1
    vocab['<teos>'] = len(vocab) + 1
    vocab['<bod>'] = len(vocab) + 1
    vocab['<eod>'] = len(vocab) + 1
    i = 0
    for filename in language_files:
        story_lines = [ line.split() for line in separate_source_data(filename) ]
        highlights_lines = [ line.split() for line in separate_target_data(filename) ]
        for lines in story_lines:
            for word in lines:
                if word not in vocab:
                    vocab[word] = len(vocab) + 1

        for lines in highlights_lines:
            for word in lines:
                if word not in vocab:
                    vocab[word] = len(vocab) + 1
        i +=  1

def get_source_doc(filename, vocab_dict):
    story = separate_source_data(filename)
    doc_source = [ [ vocab_dict[word] for word in line.split() ] for line in story ]
    return doc_source

def get_target_doc(filename, vocab_dict):
    highlight = separate_target_data(filename)
    doc_target = [ [ vocab_dict[word] for word in line.split() ] for line in highlight ]
    return doc_target

def sentence_padding(docs, max_ds_num):
    for doc in docs:
        if len(doc) < max_ds_num:
            padding_list = [[0]] * (max_ds_num - len(doc))
            for i in range(0, (max_ds_num - len(doc))):
                doc.extend([[0]])
    return docs


def word_padding(docs, max_ds_num):
    for i in range(0, max_ds_num):
        max_word_num = max([*map(lambda x: len(x), [ sentence[i] for sentence in docs ] ) ])
        for j in range(0, len(docs)):
            sl = len(docs[j][i])
            if sl < max_word_num:
                padding_word = [0] * (max_word_num - sl)
                docs[j][i].extend(padding_word)
    return docs
