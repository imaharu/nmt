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
parser.add_argument('--test_size','--ts', type=int, default=1000, help='test_size')

parser.add_argument('--result_path', '-p', type=str, default='$HOME/')
parser.add_argument('--model_path', '-m', type=str, default='$HOME/')
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
    get_test_data_target(args.test_size, output_input_lines)
