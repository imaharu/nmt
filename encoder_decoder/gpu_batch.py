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
from chainer import cuda
import cupy
from get_train_data import *

input_path = "/home/ochi/src/data/train/train_clean.txt.en"
target_path = "/home/ochi/src/data/train/train_clean.txt.ja"

train_num, padding_num, demb, batch_size = 20, 50, 256, 10

input_vocab , input_lines, input_lines_number = {}, {}, {}
target_vocab ,target_lines ,target_lines_number = {}, {}, {}

translate_words = {}

get_train_data_input(input_path, train_num, input_vocab, input_lines_number, input_lines)
ev = len(input_vocab)
get_train_data_target(target_path, train_num, target_vocab, target_lines_number, target_lines, translate_words)
jv = len(target_vocab)


class MyMT(chainer.Chain):
    def __init__(self, ev, jv, k):
        super(MyMT, self).__init__(
            embed_input = L.EmbedID(ev, k, ignore_label=-1),
            embed_target = L.EmbedID(jv, k, ignore_label=-1),
            lstm1 = L.LSTM(k,k),
            linear1 = L.Linear(k, jv),
        )

    def __call__(self, input_lines ,target_lines):
        global accum_loss
        for input_sentence_words in input_lines:
            print("input_sentence_words",input_sentence_words)
            print("data",input_sentence_words.data)
            print("type",input_sentence_words.dtype)
            input_k = self.embed_input(input_sentence_words)
            print(input_k)
            h = self.lstm1(input_k)
            print(h)

        target_lines_not_last = target_lines[:(padding_num-1)]
        target_lines_next = target_lines[1:]
        for i ,(target_sentence_words , target_sentence_words_next) in enumerate(zip(target_lines_not_last, target_lines_next)):
            if i == 0:
                accum_loss = F.softmax_cross_entropy(self.linear1(h), target_sentence_words)
            target_k = self.embed_target(target_sentence_words)
            h = self.lstm1(target_k)
            loss = F.softmax_cross_entropy(self.linear1(h), target_sentence_words_next)
            accum_loss += loss

        return accum_loss

model = MyMT(ev, jv, demb)

cuda.get_device(0).use()
model.to_gpu()

optimizer = optimizers.SGD()
optimizer.setup(model)

def padding(batch_lines):
    for i in range(len(batch_lines)):
        if i == 0:
            a1 = cupy.array([ batch_lines[i] ])
            batch_padding = F.pad_sequence(a1 ,padding_num ,-1)
        else:
            a1 = cupy.array([ batch_lines[i] ])
            k = F.pad_sequence(a1, padding_num, -1)
            batch_padding = F.concat((batch_padding, k), axis=0)
    return batch_padding

def reStructured(batch_input_paddings):
    for i in range(padding_num):
        for j in range(batch_size):
            if j == 0:
                word_line = cupy.array([ batch_input_paddings[j][i].data ],dtype=cupy.float32)
            else:
                a = cupy.array([ batch_input_paddings[j][i].data ],dtype=cupy.float32)
                word_line = F.concat((word_line, a), axis=0)
        if i == 0:
            reStructured = cupy.array([ word_line.data ],dtype=cupy.float32)
        else:
            a = cupy.array([ word_line.data ],dtype=cupy.float32)
            reStructured = F.concat((reStructured, a), axis=0)
    return reStructured

start = time.time()
for epoch in range(1):
    print("epoch",epoch)
    indexes = np.random.permutation(train_num)

    for i in range(0, train_num, batch_size):
        batch_input_lines = [ input_lines_number[int(index)] for index in indexes[i:i+batch_size]]
        for batch_input_line in batch_input_lines:
            batch_input_line.append(input_vocab['<eos>'])
        batch_input_paddings = padding(batch_input_lines)
        input_lines = reStructured(batch_input_paddings)

        batch_target_lines = [ target_lines_number[int(index)] for index in indexes[i:i+batch_size]]
        for batch_target_line in batch_target_lines:
            batch_target_line.append(target_vocab['<eos>'])
        batch_target_paddings = padding(batch_target_lines)
        target_lines = reStructured(batch_target_paddings)

        # float32 > int32
        input_lines = F.cast(input_lines, cupy.int32)
        target_lines = F.cast(target_lines, cupy.int32)

        model.lstm1.reset_state()
        model.cleargrads()
        loss = model(input_lines, target_lines)
        print("loss",loss)
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
    # outfile = "gpu_batch_mt-" + str(epoch) + ".model"
    # serializers.save_npz(outfile, model)
    # elapsed_time = time.time() - start
    # print("時間:",elapsed_time / 60.0, "分")