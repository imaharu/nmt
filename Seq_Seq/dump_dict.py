from create_sin_dict import *

# Other
import os
import glob
import pickle

source_vocab = {}
target_vocab = {}

#data_path = os.environ["aspec_unk"]
data_path = "/home/ochi/Lab/Seq_Seq/train_data"

source_vocab = get_source_dict(str(data_path) + "/train.en", source_vocab)
target_vocab = get_target_dict(str(data_path) + "/train.ja", target_vocab)
with open('en.dump', 'wb') as f:
    pickle.dump(source_vocab, f)

with open('ja.dump', 'wb') as f:
    pickle.dump(target_vocab, f)
