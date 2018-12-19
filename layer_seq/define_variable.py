from get_data import *
import argparse

parser = argparse.ArgumentParser(description='Sequence to Sequence Model by using Pytorch')
'''model param'''
parser.add_argument('--weightdecay', type=float, default=1.0e-6,
                    help='Weight Decay norm')
parser.add_argument('--gradclip', type=float, default=5.0,
                    help='Gradient norm threshold to clip')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Set dropout ratio in training')
'''train details'''
parser.add_argument('--epoch', '-e', type=int, default=10,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU')
parser.add_argument('--model', '-m', type=str, default='best_model.pth',
                    help='Model file name to serialize')
parser.add_argument('--data_path', '-p', type=str, default='/home/nakamoto/workspace/NMT/where_seq2seq/data',
                    help='Directory to input the train dataset')
parser.add_argument('--train_file', '-f', type=str, default='train.txt')
parser.add_argument('--out', '-o', type=str, default='result/seq2seq/20000/',
                    help='Directory to output the result')

parser.add_argument('--evaluate', type=int, default=-1,
                    help='Interval to output eval loss')
parser.add_argument('--generate', help='generate only', action='store_true')

'''train_num embed hidden batch'''
parser.add_argument('--train_size','-t', type=int, default=20000,
                    help='train num')
parser.add_argument('--embed_size', type=int, default=256,
                    help='size of embed size for word representation')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--batch_size', '-b', type=int, default=50,
                    help='Number of batchsize')
parser.add_argument('--layer_num', '-l', type=int, default=2,
                    help='Layer num')
parser.add_argument('--train_or_generate', '--tg', type=int, default=0, help='train is 0 : generete is 1')
parser.set_defaults(generate=False)
args = parser.parse_args()

input_vocab , input_lines, input_lines_number = {}, {}, {}
target_vocab ,target_lines ,target_lines_number = {}, {}, {}
output_input_lines = {}
translate_words = {}

# paddingで0を入れるから
get_train_data_input(args.train_size, input_vocab, input_lines_number, input_lines)
ev = len(input_vocab) + 1

get_train_data_target(args.train_size, target_vocab, target_lines_number, target_lines, translate_words)
jv = len(target_vocab) + 1

if args.train_or_generate == 1:
    get_test_data_target(test_num, output_input_lines)
