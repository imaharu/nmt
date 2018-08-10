import chainer
from chainer import serializers
from chainer import cuda, Variable, optimizers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import MeCab
import re
import time
import numpy as np

path_train_en = "/home/ochi/src/data/train/train_clean.txt.en"
path_train_ja = "/home/ochi/src/data/train/train_clean.txt.ja"
train_num = 200
padding_num = 60 # コーパス作成際にcleaningを60にしたため

input_vocab = {}
input_lines = {}
input_lines_number = {}

target_vocab = {}
target_lines = {}
target_lines_number = {}

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
        input_lines_number[i] = [input_vocab[word] for word in line.split()]
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
        target_lines_number[i] = [target_vocab[word] for word in line.split()]
        target_lines[i] = line
        i += 1

    id = len(target_vocab)
    target_vocab['<eos>'] = id
    translate_words[id] = "<eos>"
    jv = len(target_vocab)


class MyMT(chainer.Chain):
    def __init__(self, ev, jv, k):
        super(MyMT, self).__init__( 
            embed_input = L.EmbedID(ev,k,ignore_label=-1),
            embed_target = L.EmbedID(jv,k,ignore_label=-1),
            lstm1 = L.LSTM(k,k),
            linear1 = L.Linear(k, jv),
        )

    def __call__(self, input_line ,target_line):
        global accum_loss
        print(input_line[0])

        for sentence_words in input_line:
            input_k = self.embed_input(sentence_words)
            h = self.lstm1(input_k)

        for i in range(len(input_line)):
            wid = input_vocab[input_line[i]]
            input_k = self.embed_input(Variable(np.array([wid], dtype=np.int32)))
            h = self.lstm1(input_k)

        last_input_k = self.embed_input(Variable(np.array([input_vocab["<eos>"]],dtype=np.int32)))
        tx = Variable(np.array([input_vocab[input_line[0]]], dtype=np.int32))
        
        h = self.lstm1(last_input_k)

        accum_loss = F.softmax_cross_entropy(self.linear1(h), tx)

        for i in range(len(target_line)):
            wid = target_vocab[target_line[i]]
            target_k = self.embed_target(Variable(np.array([wid], dtype=np.int32)))
            next_wid = target_vocab["<eos>"] if (i == len(target_line) -1) else target_vocab[target_line[i+1]]
            tx = Variable(np.array([next_wid], dtype=np.int32))
            h = self.lstm1(target_k)
            
            loss = F.softmax_cross_entropy(self.linear1(h),tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

demb = 64
batch_size = 1
model = MyMT(ev, jv, demb)

# gpu_device = 0
# cuda.get_device(gpu_device).use()
# model.to_gpu()
# np = cuda.cupy
# optimizer = optimizers.Adam()
optimizer = optimizers.SGD()
optimizer.setup(model)

def padding(batch_lines):
    for i in range(len(batch_lines)):
        if i == 0:
            a1 = np.array([ batch_lines[i] ], dtype=np.int32)
            batch_padding = F.pad_sequence(a1 ,50 ,-1)
        else:
            a1 = np.array([ batch_lines[i] ], dtype=np.int32)
            k = F.pad_sequence(a1, 50, -1)
            batch_padding = F.concat((batch_padding, k), axis=0)
    return batch_padding

def reStructured(batch_input_paddings):
    for i in range(50): # padding_num　実際は
        for j in range(batch_size):
            if j == 0:
                word_line = np.array([ batch_input_paddings[j][i].data ], dtype=np.int32)
            else:
                a = np.array([ batch_input_paddings[j][i].data ], dtype=np.int32)
                word_line = F.concat((word_line, a), axis=0)
        if i == 0:
            reStructured = np.array([ word_line.data ], dtype=np.int32)
        else:
            a = np.array([ word_line.data ], dtype=np.int32)
            reStructured = F.concat((reStructured, a), axis=0)
    return reStructured

start = time.time()
for epoch in range(1):
    print("epoch",epoch)
    indexes = np.random.permutation(train_num)

    for i in range(0, train_num, batch_size):
        batch_input_lines = [ input_lines_number[int(index)] for index in indexes[i:i+batch_size]]
        batch_input_paddings = padding(batch_input_lines)
        input_reStructured = reStructured(batch_input_paddings)
        batch_target_lines = [ target_lines_number[int(index)] for index in indexes[i:i+batch_size]]
        batch_target_paddings = padding(batch_target_lines)
        target_reStructured = reStructured(batch_target_paddings)

        model.lstm1.reset_state()
        model.cleargrads()
        loss = model(input_reStructured, target_reStructured)
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
        break
    # outfile = "mt-" + str(epoch) + ".model"
    # serializers.save_npz(outfile, model)
    # elapsed_time = time.time() - start
    # print("時間:",elapsed_time / 60.0, "分")
