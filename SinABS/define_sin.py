# my function
from create_sin_dict import *

# Other
import os
import glob

english_vocab = {}

train_doc_num = 20000
batch_size = 5
hidden_size = 1000

data_path = os.environ["cnn_data"]

english_paths = sorted(glob.glob(data_path + "/*.story"))[0:train_doc_num]

get_dict(english_paths, english_vocab, train_doc_num)

source_size = len(english_vocab) + 1
target_size = len(english_vocab) + 1
