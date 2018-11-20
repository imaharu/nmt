# my function
from create_sin_dict import *

# Other
import os
import glob
import torch
import pickle
english_vocab = {}

train_doc_num = 19990
batch_size = 10
hidden_size = 512

data_path = os.environ["cnn_unk"]
#english_paths = sorted(glob.glob(data_path + "/*.story"))[train_doc_num:train_doc_num+10]
english_paths = sorted(glob.glob(data_path + "/*.story"))[0:train_doc_num]

#get_dict(english_paths, english_vocab)
with open('cnn.dump', 'rb') as f:
    english_vocab = pickle.load(f)

source_size = len(english_vocab) + 1
target_size = len(english_vocab) + 1

device = torch.device('cuda:1')
