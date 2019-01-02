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
parser.add_argument('--save_path', '-s' , type=str)

parser.add_argument('--is_short_data', '-d' , type=int, default=1,
                    help='short: vocab20000, long: vocab100000')
args = parser.parse_args()

##### end #####

if args.is_short_data:
    train_en = "../train_data/train.en"
    train_ja = "../train_data/train.ja"
    source_vocab = "vocab/source20000_vocab"
    target_vocab = "vocab/target20000_vocab"
else:
    train_en = "../train_data/100000train.en"
    train_ja = "../train_data/100000train.ja"
    source_vocab = "vocab/source_vocab"
    target_vocab = "vocab/target_vocab"

word_data = Word_Data(train_en, train_ja, source_vocab, target_vocab)

exit()

if args.mode == 0:
    train_doc_num = 6
    hidden_size = 4
    embed_size = 4
    batch_size = 2
    epoch = 2
    source_data = torch.load(args.load_source_file)
    target_data = torch.load(args.load_target_file)

elif args.mode == 1:
    train_doc_num = args.train_num
    hidden_size = args.hidden_size
    embed_size = args.embed_size
    batch_size = args.batch_size
    epoch = args.epoch
    source_data = torch.load(args.load_source_file)
    target_data = torch.load(args.load_target_file)
else:
    batch_size = 1
    hidden_size = args.hidden_size

source_size = word_data.getVocabSize(1)
target_size = word_data.getVocabSize(0)


