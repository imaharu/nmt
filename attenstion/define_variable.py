import os
import glob
import torch
import pickle
import argparse
from preprocessing import *

# Set logger
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

##### args #####
parser = argparse.ArgumentParser(description='Sequence to Sequence Model by using Pytorch')
''' mode '''
parser.add_argument('--mode', type=int, default=0,
                    help='0: debug / 1: train / 2: eval')
'''train details'''
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')

'''train_num embed hidden batch'''
parser.add_argument('--embed_size', type=int, default=256,
                    help='size of embed size for word representation')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--batch_size', '-b', type=int, default=50,
                    help='Number of batchsize')

parser.add_argument('--result_path', '-r' ,type=str)
parser.add_argument('--model_path', '-m' , type=str)
parser.add_argument('--save_path', '-s' , type=str, default="train")

parser.add_argument('--is_short_data', '-d' , type=int, default=1,
                    help='short: vocab20000, long: vocab100000')
args = parser.parse_args()

##### end #####

if args.is_short_data:
    train_en = "../train_data/train.en"
    train_ja = "../train_data/train.ja"
    source_vocab = "vocab/source_20000vocab"
    target_vocab = "vocab/target_20000vocab"
else:
    train_en = "../train_data/100000train.en"
    train_ja = "../train_data/100000train.ja"
    source_vocab = "vocab/source_vocab"
    target_vocab = "vocab/target_vocab"

val_en = "../train_data/val.en"
val_ja = "../train_data/val.ja"

pre_data = Preprocess()
source_dict = pre_data.getVocab(source_vocab)
target_dict = pre_data.getVocab(target_vocab)

source_size = len(source_dict)
target_size = len(target_dict)

if args.mode == 0:
    train_source = pre_data.load(train_en , 0, source_dict)
    train_target = pre_data.load(train_ja , 1, target_dict)
    val_source = pre_data.load(val_en, 0, source_dict)
    train_source = train_source[:6]
    train_target = train_target[:6]
    val_source = val_source[:3]
    hidden_size = 2
    embed_size = 4
    batch_size = 3
    epoch = 2

elif args.mode == 1:
    train_source = pre_data.load(train_en , 0, source_dict)
    train_target = pre_data.load(train_ja , 1, target_dict)
    val_source = pre_data.load(val_en, 0, source_dict)
    hidden_size = args.hidden_size
    embed_size = args.embed_size
    batch_size = args.batch_size
    epoch = args.epoch

else:
    batch_size = 1
    hidden_size = args.hidden_size
    test_en = "../train_data/eval.en"
    test_source = pre_data.load(test_en , 0, source_dict)
