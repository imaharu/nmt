path_train = "/home/ochi/src/ASPEC/ASPEC-JE/train/train-1.txt"
path_test = "/home/ochi/src/ASPEC/ASPEC-JE/test/test.txt"

f_ja_train = open('/home/ochi/src/data/basic_ja_train.txt', 'w', encoding="utf-8")
f_en_train = open('/home/ochi/src/data/basic_en_train.txt', 'w', encoding="utf-8")
with open(path_train,'r',encoding='utf-8') as f:
    lines_je = f.read().strip().split('\n')
    pairs = [[words for i,words in enumerate(line.split('|||')) if i > 2] for k,line in enumerate(lines_je)]
    i = 0
    for pair in pairs:
        if not pair is pairs[-1]:
            f_ja_train.write(pair[0].strip() + '\n')
            f_en_train.write(pair[1].strip() + '\n')
        else:
            f_ja_train.write(pair[0].strip())
            f_en_train.write(pair[1].strip())

f_ja_train.closed
f_en_train.closed

f_ja_test = open('/home/ochi/src/data/basic_ja_test.txt', 'w', encoding="utf-8")
f_en_test = open('/home/ochi/src/data/basic_en_test.txt', 'w', encoding="utf-8")
with open(path_train,'r',encoding='utf-8') as f:
    lines_je = f.read().strip().split('\n')
    pairs = [[words for i,words in enumerate(line.split('|||')) if i > 2] for k,line in enumerate(lines_je)]
    
    for pair in pairs:
        if not pair is pairs[-1]:
            f_ja_test.write(pair[0].strip() + '\n')
            f_en_test.write(pair[1].strip() + '\n')
        else:
            f_ja_test.write(pair[0].strip())
            f_en_test.write(pair[1].strip())

f_ja_test.closed
f_en_test.closed