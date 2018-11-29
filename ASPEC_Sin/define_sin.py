# my function
from create_sin_dict import *

# Other
import os
import glob
import torch
import pickle
import argparse

##### args #####
parser = argparse.ArgumentParser()
parser.add_argument("-tdc", "--train_doc_num", help="train_doc_num")
parser.add_argument("-bs", "--batch_size", help="batch_size")
parser.add_argument("-hs", "--hidden_size", help="hidden_size")
parser.add_argument("-t", "--train", help="train")
parser.add_argument("-e", "--eval", help="eval")
parser.add_argument("-n", "--new", help="new")
parser.add_argument("-de", "--device", help="device")
args = parser.parse_args()
##### end #####

source_vocab = {}
target_vocab = {}
if args.train_doc_num:
    train_doc_num = int(args.train_doc_num)

if args.batch_size:
    batch_size = int(args.batch_size)

if args.hidden_size:
    hidden_size = int(args.hidden_size)

if args.device:
    device = torch.device('cuda:'+ str(args.device))
else:
    device = torch.device("cuda:0")

#data_path = os.environ["aspec_unk"]
data_path = "/home/ochi/Lab/Seq_Seq/train_data"

if not args.new:
    with open('en.dump', 'rb') as f:
        source_vocab = pickle.load(f)
    with open('ja.dump', 'rb') as f:
        target_vocab = pickle.load(f)
else:
    source_vocab = get_source_dict(str(data_path) + "/train.en", source_vocab)
    target_vocab = get_target_dict(str(data_path) + "/train.ja", target_vocab)

source_size = len(source_vocab) + 1
target_size = len(target_vocab) + 1
sfn = str(data_path)  + "/train.en"
tfn = str(data_path)  + "/train.ja"
test_file = str(data_path) + "/test.en"
