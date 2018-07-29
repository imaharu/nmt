import chainer
from chainer import cuda, Variable, optimizers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import MeCab
import re
import time
path_train = "/home/ochi/src/ASPEC/ASPEC-JE/train/train-1.txt"
path_test = "/home/ochi/src/ASPEC/ASPEC-JE/test/test.txt"
train_num = 1
test_num = 10
cleaning_num = 80

def mecab(s):
    m = MeCab.Tagger ("-Owakati")
    s = re.sub(r"\n", "",s)
    return m.parse(s)

def lowercasing(s):
    return s.lower()

def cleaning(target_words,input_words):
    # Trueはどちらかがcleaning_num文字以上
    if len(target_words.split()) >= cleaning_num or len(input_words.split()) >= cleaning_num :
        return True
    else:
        return False

input_vocab = {}
input_lines = {}

target_vocab = {}
target_lines = {}

translate_words = {}

accum_loss = 0
with open(path_train,'r',encoding='utf-8') as f:
    lines_je = f.read().strip().split('\n')
    pairs = [[words for i,words in enumerate(line.split('|||')) if i > 2] for k,line in enumerate(lines_je) if k < train_num]

    for i in range(train_num):

        input_words = lowercasing(pairs[i][1])
        target_words = mecab(pairs[i][0])
        
        if cleaning(target_words,input_words):
            continue

        for input_word in input_words.split():
            if input_word not in input_vocab:
                id = len(input_vocab)
                input_vocab[input_word] = id
                translate_words[id] = input_word

        for target_word in target_words.split():
            if target_word not in target_vocab:
                target_vocab[target_word] = len(target_vocab)

        input_before = pairs[i][1]
        print("input_before", input_before)
        input_lines[i] = lowercasing(pairs[i][1])
        target_lines[i] = mecab(pairs[i][0])

    input_vocab['<eos>'] = len(input_vocab)
    ev = len(input_vocab)
    
    id = len(target_vocab)
    target_vocab['<eos>'] = id
    translate_words[id] = "<eos>"
    jv = len(target_vocab)


class MyMT(chainer.Chain):
    def __init__(self, jv, ev, k):
        super(MyMT, self).__init__( 
            embed_input = L.EmbedID(ev,k),
            embed_target = L.EmbedID(jv,k),
            lstm1 = L.LSTM(k,k),
            linear1 = L.Linear(k, jv),
        )

    def __call__(self, input_line ,target_line):
        global accum_loss
        for i in range(len(input_line)):
            wid = input_vocab[input_line[i]]
            input_k = self.embed_input(Variable(xp.array([wid], dtype=xp.int32)))
            h = self.lstm1(input_k)
        last_input_k = self.embed_input(Variable(xp.array([input_vocab["<eos>"]],dtype=xp.int32)))
        tx = Variable(xp.array([input_vocab[input_line[0]]], dtype=xp.int32))
        
        h = self.lstm1(last_input_k)

        accum_loss = F.softmax_cross_entropy(self.linear1(h), tx)
        
        for i in range(len(target_line)):
            wid = target_vocab[target_line[i]]
            target_k = self.embed_target(Variable(xp.array([wid], dtype=xp.int32)))
            next_wid = target_vocab["<eos>"] if (i == len(target_line) -1) else target_vocab[target_line[i+1]]
            tx = Variable(xp.array([next_wid], dtype=xp.int32))
            h = self.lstm1(target_k)
            
            loss = F.softmax_cross_entropy(self.linear1(h),tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

demb = 64
batch_size = 128
model = MyMT(ev, jv, demb)

gpu_device = 0
cuda.get_device(gpu_device).use()
model.to_gpu()
xp = cuda.cupy
optimizer = optimizers.Adam()
optimizer.setup(model)

start = time.time()

for epoch in range(1):
    print("epoch",epoch)
    for i in range(train_num):
        input_line = input_lines[i].split()
        target_line = target_lines[i].split()
        model.lstm1.reset_state()
        model.cleargrads()
        loss = model(input_line, target_line)
        # print("before_loss",loss)
        loss.backward()
        # print("after_loss",loss)
        # print("model.linear1.linear1",model.linear1.linear1)
        # print("model.linear1.linear1.grad",model.linear1.linear1.grad)
        # print("model.linear1.b",model.linear1.b)
        # print("model.linear1.b.grad",model.linear1.b.grad)
        loss.unchain_backward()
        optimizer.update()