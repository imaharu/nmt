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

english_vocab = {}

if args.train_doc_num:
    train_doc_num = int(args.train_doc_num)
else:
    train_doc_num = 6

if args.batch_size:
    batch_size = int(args.batch_size)
else:
    batch_size = 3

if args.hidden_size:
    hidden_size = int(args.hidden_size)
else:
    hidden_size = 4

if args.device:
    device = torch.device('cuda:'+ str(args.device))
else:
    device = torch.device("cuda:0")

data_path = os.environ["cnn_unk"] + "/train"
english_paths = sorted(glob.glob(data_path + "/*.story"))[0:train_doc_num]

if not args.new:
    with open('cnn.dump', 'rb') as f:
        english_vocab = pickle.load(f)
else:
    get_dict(english_paths, english_vocab)

source_size = len(english_vocab) + 1
target_size = len(english_vocab) + 1

