def get_train_data_input(input_path, train_num, input_vocab, input_lines_number, input_lines):
    with open(input_path,'r',encoding='utf-8') as f:
        lines_en = f.read().strip().split('\n')
        i = 0
        for line in lines_en:
            if i == train_num:
                break
            for input_word in line.split():
                if input_word not in input_vocab:
                    input_vocab[input_word] = len(input_vocab)
            input_lines_number[i] = [input_vocab[word] for word in line.split()]
            input_lines[i] = line
            i += 1
        input_vocab['<eos>'] = len(input_vocab)

def get_train_data_target(target_path, train_num, target_vocab, target_lines_number, target_lines, translate_words):
    with open(target_path,'r',encoding='utf-8') as f:
        lines_ja = f.read().strip().split('\n')
        i = 0
        for line in lines_ja:
            if i == train_num:
                break
            for target_word in line.split():
                if target_word not in target_vocab:
                    id = len(target_vocab)
                    target_vocab[target_word] = len(target_vocab)
                    translate_words[id] = target_word
            target_lines_number[i] = [target_vocab[word] for word in line.split()]
            target_lines[i] = line
            i += 1

        id = len(target_vocab)
        target_vocab['<eos>'] = id
        translate_words[id] = "<eos>"