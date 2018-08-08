import chainer
from chainer import serializers
from chainer import cuda, Variable, optimizers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import MeCab
import re
import time
path_train_en = "/home/ochi/src/data/train/train_clean.txt.en"
path_train_ja = "/home/ochi/src/data/train/train_clean.txt.ja"
train_num = 20000

input_vocab = {}
input_lines = {}

target_vocab = {}
target_lines = {}

translate_words = {}

accum_loss = 0

with open(path_train_en,'r',encoding='utf-8') as f:
    lines_en = f.read().strip().split('\n')
    i = 0
    for line in lines_en:
        if i == train_num:
            break
        for input_word in line.split():
            if input_word not in input_vocab:
                input_vocab[input_word] = len(input_vocab)
        input_lines[i] = line
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

        target_lines[i] = line
        i += 1

    id = len(target_vocab)
    target_vocab['<eos>'] = id
    translate_words[id] = "<eos>"
    jv = len(target_vocab)


class MyMT(chainer.Chain):
    def __init__(self, ev, jv, k):
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
        # print("target_line",target_line)
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

for epoch in range(15):
    print("epoch",epoch)
    for i in range(train_num):
        input_line = input_lines[i].split()
        target_line = target_lines[i].split()
        model.lstm1.reset_state()
        model.cleargrads()
        loss = model(input_line, target_line)
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
    outfile = "mt-" + str(epoch) + ".model"
    serializers.save_npz(outfile, model)
    elapsed_time = time.time() - start
    print("時間:",elapsed_time / 60.0, "分")
