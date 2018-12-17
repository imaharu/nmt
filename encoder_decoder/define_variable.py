from get_data import *
train_num, hidden_size, batch_size = 20000, 256, 50
#train_num, hidden_size, batch_size = 100, 4, 2
train_num, hidden_size, batch_size = 20000, 256, 1


input_vocab , input_lines, input_lines_number = {}, {}, {}
target_vocab ,target_lines ,target_lines_number = {}, {}, {}
output_input_lines = {}
translate_words = {}

# paddingで0を入れるから
get_train_data_input(train_num, input_vocab, input_lines_number, input_lines)
ev = len(input_vocab) + 1

get_train_data_target(train_num, target_vocab, target_lines_number, target_lines, translate_words)
jv = len(target_vocab) + 1

test_num = 1000

get_test_data_target(test_num, output_input_lines)
