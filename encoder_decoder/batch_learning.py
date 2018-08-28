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
train_num = 20000
padding_num = 50 # コーパス作成際にcleaningを50にしたため

input_vocab = {}
input_lines = {}
input_lines_number = {}

target_vocab = {}
target_lines = {}
target_lines_number = {}

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
        for input_sentence_words in input_line:
            input_k = self.embed_input(input_sentence_words)
            h = self.lstm1(input_k)
        # eoses = F.broadcast_to(np.asarray([input_vocab["<eos>"]]), (1, batch_size))
        # last_input_k = self.embed_input(eoses)
        # h = self.lstm1(last_input_k[0])
        target_line_not_last = target_line[:(padding_num-1)]
        target_line_next = target_line[1:]
        for i ,(target_sentence_words , target_sentence_words_next) in enumerate(zip(target_line_not_last, target_line_next)):
            if i == 0:
                accum_loss = F.softmax_cross_entropy(self.linear1(h), target_sentence_words)
            target_k = self.embed_target(target_sentence_words)
            h = self.lstm1(target_k)
            loss = F.softmax_cross_entropy(self.linear1(h), target_sentence_words_next)
            accum_loss += loss

        return accum_loss

demb = 256
batch_size = 50
model = MyMT(ev, jv, demb)

# gpu_device = 0
# cuda.get_device(gpu_device).use()
# model.to_gpu(gpu_device)
# xp = cuda.cupy

optimizer = optimizers.SGD()
optimizer.setup(model)

def padding(batch_lines):
    for i in range(len(batch_lines)):
        if i == 0:
            a1 = np.array([ batch_lines[i] ], dtype=np.int32)
            batch_padding = F.pad_sequence(a1 ,padding_num ,-1)
        else:
            a1 = np.array([ batch_lines[i] ], dtype=np.int32)
            k = F.pad_sequence(a1, padding_num, -1)
            batch_padding = F.concat((batch_padding, k), axis=0)
    return batch_padding

def reStructured(batch_input_paddings):
    for i in range(padding_num):
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
for epoch in range(20):
    print("epoch",epoch)
    indexes = np.random.permutation(train_num)

    for i in range(0, train_num, batch_size):
        batch_input_lines = [ input_lines_number[int(index)] for index in indexes[i:i+batch_size]]
        for batch_input_line in batch_input_lines:
            batch_input_line.append(input_vocab['<eos>'])
        batch_input_paddings = padding(batch_input_lines)
        input_reStructured = reStructured(batch_input_paddings)

        batch_target_lines = [ target_lines_number[int(index)] for index in indexes[i:i+batch_size]]
        for batch_target_line in batch_target_lines:
            batch_target_line.append(target_vocab['<eos>'])
        batch_target_paddings = padding(batch_target_lines)
        target_reStructured = reStructured(batch_target_paddings)

        model.lstm1.reset_state()
        model.cleargrads()
        loss = model(input_reStructured, target_reStructured)
        print("loss",loss)
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
    outfile = "batch_mt-" + str(epoch) + ".model"
    serializers.save_npz(outfile, model)
    elapsed_time = time.time() - start
    print("時間:",elapsed_time / 60.0, "分")