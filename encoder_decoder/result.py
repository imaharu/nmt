import chainer
from chainer import serializers
from chainer import cuda, Variable, optimizers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

path_train_en = "/home/ochi/src/data/train/train_clean.txt.en"
path_train_ja = "/home/ochi/src/data/train/train_clean.txt.ja"

path_test_en = "/home/ochi/src/data/test/test_clean.txt.en"

train_num = 20000
test_num = 1000

input_vocab = {}
target_vocab = {}

translate_words = {}

with open(path_train_en,'r',encoding='utf-8') as f:
    lines_en = f.read().strip().split('\n')
    i = 0
    for line in lines_en:
        if i == train_num:
            break
        for input_word in line.split():
            if input_word not in input_vocab:
                input_vocab[input_word] = len(input_vocab)
        i += 1
    input_vocab['<eos>'] = len(input_vocab)
    ev = len(input_vocab)

with open(path_train_ja,'r',encoding='utf-8') as f:
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
        i += 1

    id = len(target_vocab)
    target_vocab['<eos>'] = id
    translate_words[id] = "<eos>"
    jv = len(target_vocab)


test_input_lines = {}
with open(path_test_en,'r',encoding='utf-8') as f:
    lines_en = f.read().strip().split('\n')
    i = 0
    for line in lines_en:
        if i == test_num:
            break
        test_input_lines[i] = line
        i += 1

class MyMT(chainer.Chain):
    def __init__(self, ev, jv, k):
        super(MyMT, self).__init__( 
            embed_input = L.EmbedID(ev,k),
            embed_target = L.EmbedID(jv,k),
            lstm1 = L.LSTM(k,k),
            linear1 = L.Linear(k, jv),
        )

demb = 64
model = MyMT(ev, jv, demb)
serializers.load_npz("mt-13.model", model)

gpu_device = 0
cuda.get_device(gpu_device).use()
model.to_gpu()
xp = cuda.cupy
optimizer = optimizers.Adam()
optimizer.setup(model)

def mt(model, test_input_line):
    ans_ja = []
    model.lstm1.reset_state()
    for i in range(len(test_input_line)):    
        ## 辞書にある場合は
        if input_vocab.get(test_input_line[i]):
            wid = input_vocab[test_input_line[i]]
        else:
            wid = input_vocab[","]

        with chainer.using_config('train', False):
            input_k= model.embed_input(Variable(xp.array([wid], dtype=xp.int32)))
        h = model.lstm1(input_k)
    with chainer.using_config('train', False):
        last_input_k = model.embed_input(Variable(xp.array([input_vocab["<eos>"]],dtype=xp.int32)))

    h = model.lstm1(last_input_k)
    wid = xp.argmax(F.softmax(model.linear1(h)).data[0])
    
    loop = 0

    while(wid != target_vocab['<eos>']) and (loop <= 50):
        with chainer.using_config('train', False):
            target_k = model.embed_target(Variable(xp.array([wid],dtype=xp.int32)))
        h = model.lstm1(target_k)
        wid = xp.argmax(F.softmax(model.linear1(h)).data[0])
        wid = int(cuda.to_cpu(wid))
        loop +=1
        if wid != target_vocab['<eos>']:
            ans_ja.append(translate_words[wid])
    print("出力データ |   ", ' '.join(ans_ja))
    print("--------------------------------")
    return ans_ja

result_file_ja = '/home/ochi/src/data/blue/result_ja.txt'
result_file = open(result_file_ja, 'w', encoding="utf-8")

for i in range(len(test_input_lines)):
    print(i)
    print("入力データ |   ", test_input_lines[i])
    test_input_line = test_input_lines[i].split()
    result = mt(model, test_input_line)

    if i == (len(test_input_lines) - 1):
        result_file.write(' '.join(result).strip())
    else:
        result_file.write(' '.join(result).strip() + '\n')
result_file.close