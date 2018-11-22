from create_sin_dict import *

# Other
import os
import glob
import pickle

english_vocab = {}

train_doc_num = 19990

data_path = os.environ["cnn_unk"]

english_paths = sorted(glob.glob(data_path + "/*.story"))[0:train_doc_num]

get_dict(english_paths, english_vocab)

with open('cnn.dump', 'wb') as f:
    pickle.dump(english_vocab, f)
