# my function
from create_sin_dict import *

# Other
import os
import glob
import sys
import argparse
from collections import Counter

en_c_vocab = Counter()

parser = argparse.ArgumentParser()
parser.add_argument("-md", "--max_doc", help="max_doc")
parser.add_argument("-cn", "--co_num", help="co_num")
args = parser.parse_args()

if args.max_doc:
    max_doc = int(args.max_doc)
if args.co_num:
    co_num = int(args.co_num)
data_path = os.environ["aspec_data"]

def get_unk_dict(file_name, c_vocab):
    with open(file_name) as f:
        docs = f.read().strip().split('\n')
        for doc in docs:
            for word in doc.split():
                if word in c_vocab:
                    c_vocab[word] += 1
                else:
                    c_vocab[word] = 1
    return c_vocab.most_common()[:co_num-1:-1]

def unk_file(file_name, unk_dict, f_n):
    with open(file_name) as f:
        docs = f.read().strip().split('\n')
        docs_c = []
        for doc in docs:
            concat_line = []
            for word in doc.split():
                if word in unk_dict:
                    concat_line.append("<unk>")
                else:
                    concat_line.append(word)
            docs_c.append(" ".join(concat_line))
        with open("aspec_unk/" + f_n, "w") as w_f:
            w_f.write("\n".join(docs_c))

ja_unk_dict = dict(get_unk_dict(str(data_path + "/train20000.ja"), en_c_vocab))
en_unk_dict = dict(get_unk_dict(str(data_path + "/train20000.en"), en_c_vocab))
#unk_file(str(data_path + "/train20000.en"), en_unk_dict, "train.en")
unk_file(str(data_path + "/train20000.ja"), en_unk_dict, "train.ja")
unk_file(str(data_path + "/test1000.en"), en_unk_dict, "test.en")
