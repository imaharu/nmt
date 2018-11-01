import glob
import linecache
from cnn_data import *

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
