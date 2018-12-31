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
'''model param'''
parser.add_argument('--weightdecay', type=float, default=1.0e-6,
                    help='Weight Decay norm')
parser.add_argument('--gradclip', type=float, default=5.0,
                    help='Gradient norm threshold to clip')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Set dropout ratio in training')
'''train details'''
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')

'''train_num embed hidden batch'''
parser.add_argument('--train_doc_num','-t', type=int,
                    help='train num')
parser.add_argument('--embed_size', type=int, default=256,
                    help='size of embed size for word representation')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--batch_size', '-b', type=int, default=50,
                    help='Number of batchsize')
parser.add_argument('--train_or_generate', '--tg', type=int, default=0, help='train is 0 : generete is 1')
parser.add_argument('--test_size',type=int, default=1000, help='test_size')

parser.add_argument('--result_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--debug', type=int, default=0)

parser.add_argument('--load_source_file', type=str, default="data/source.pt",
                    help='load source file')
parser.add_argument('--load_target_file', type=str, default="data/target.pt",
                    help='load target file')
parser.add_argument('--create_data', type=int, default=0,
                    help='if over 0 like 1 it will create data')
args = parser.parse_args()

##### end #####

word_data = Word_Data("vocab/source_vocab", "vocab/target_vocab")

if args.create_data:
    source_save_path = "data/source.pt"
    target_save_path = "data/target.pt"
    word_data.save(source_save_path, target_save_path)
    exit()

if args.debug:
    train_doc_num = 6
    hidden_size = 4
    embed_size = 4
    batch_size = 2
    epoch = 2

else:
    train_doc_num = args.train_doc_num
    hidden_size = args.hidden_size
    embed_size = args.embed_size
    batch_size = args.batch_size
    epoch = args.epoch

if args.train_or_generate == 1:
    get_test_data_target(args.test_size, output_input_lines)

source_size = word_data.getVocabSize(1)
target_size = word_data.getVocabSize(0)

if train_doc_num is None:
    source_data = torch.load(args.load_source_file)
    target_data = torch.load(args.load_target_file)
    train_doc_num = len(source_data)
else:
    source_data = torch.load(args.load_source_file)[0:train_doc_num]
    target_data = torch.load(args.load_target_file)[0:train_doc_num]

logger.debug("訓練文書数: " +  str(train_doc_num))
logger.debug("hidden_size: " + str(hidden_size))
logger.debug("embed_size: " +  str(embed_size))
logger.debug("epoch : " + str(epoch))
logger.debug("batch size : " +  str(batch_size))
