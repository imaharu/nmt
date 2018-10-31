# my function
from create_sin_dict import *

# Other
import os
import glob

english_vocab = {}

train_doc_num = 2
batch_size =  2
hidden_size = 256

data_path = os.environ["cnn_data"]

english_paths = sorted(glob.glob(data_path + "/*.story"))[0:train_doc_num]

get_dict(english_paths, english_vocab, train_doc_num)

english_vocab = get_dict(english_paths, english_vocab, train_doc_num)
get_cnn_source_doc(english_paths[0],  english_vocab)

# source_size = len(english_vocab) + 1
# target_size = len(english_vocab) + 1
