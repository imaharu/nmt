def create_dict_word_num():
    dicts = {}
    with open("train.ja",'r',encoding='utf-8') as lines:
        for line in lines:
            for word in line.split():
                if dicts.get(word):
                    dicts[word] += 1
                else:
                    dicts[word] = 1
    return dicts

dicts = create_dict_word_num()

train_unk = open('train_unk.ja', 'w', encoding="utf-8")
with open("train.ja",'r',encoding='utf-8') as lines:
    for line in lines:
        new_line = []
        for word in line.split():
            if dicts[word] == 1:
                new_line.append("<unk>")
            else:
                new_line.append(word)
        train_unk.write(' '.join(new_line)  + '\n')
train_unk.closed