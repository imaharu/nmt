# my function
from create_sin_dict import *

# Other
import os
import glob
import sys
from collections import Counter

argvs = sys.argv
c_vocab = Counter()
train_doc_num = int(argvs[1]) # defalut 20000
co_num = int(argvs[2]) # defalut 30000
data_path = os.environ["cnn_data"]
#data_path = "/home/ochi/Lab/SinABS/test"

english_paths = sorted(glob.glob(data_path + "/*.story"))[0:train_doc_num]
def get_unk_dict(language_files, c_vocab):
    for filename in language_files:
        story_lines = [ line.split() for line in separate_source_data(filename) ]
        highlights_lines = [ line.split() for line in separate_target_data(filename) ]
        # story
        for lines in story_lines:
            for word in lines:
                if word in c_vocab:
                    c_vocab[word] += 1
                else:
                    c_vocab[word] = 1
        # highlights
        for lines in highlights_lines:
            for word in lines:
                if word in c_vocab:
                    c_vocab[word] += 1
                else:
                    c_vocab[word] = 1
    return c_vocab.most_common()[:co_num-1:-1]

def unk_file(language_files, unk_dict):
    i = 1 
    for filename in language_files:
        with open(filename) as f:
            doc = []
            for line in f:
                concat_line = []
                for word in line.split():
                    if word in unk_dict:
                        concat_line.append("<unk>")
                    else:
                        concat_line.append(word)
                doc.append(" ".join(concat_line))
        with open("cnn_unk/" + str(i).zfill(5) + ".story",  "w" ) as w_f:
            w_f.write("\n".join(doc))
        i += 1
        print(i)

unk_dict = dict(get_unk_dict(english_paths, c_vocab))
unk_file(english_paths, unk_dict)
