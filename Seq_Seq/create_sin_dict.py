import glob
import linecache

def get_dict(file_name, vocab):
    vocab['<unk>'] = len(vocab) + 1
    vocab['<teos>'] = len(vocab) + 1
    vocab['<bod>'] = len(vocab) + 1
    vocab['<eod>'] = len(vocab) + 1

    with open(file_name) as f:
        docs = f.read().strip().split("\n")
        for doc in docs:
            for word in doc.split():
                if word not in vocab:
                    vocab[word] = len(vocab) + 1
    return vocab

def get_source_doc(filename, ln, vocab_dict):
    source_doc = linecache.getline(filename, int(ln))
    doc_source = [  vocab_dict[word] if word in vocab_dict else vocab_dict["<unk>"] for word in source_doc.split()   ]
    return doc_source


def get_target_doc(filename, ln, vocab_dict):
    target_doc = linecache.getline(filename, int(ln))
    doc_target = [  vocab_dict[word] if word in vocab_dict else vocab_dict["<unk>"] for word in target_doc.split()   ]
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
