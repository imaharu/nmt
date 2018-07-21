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
train_num = 1000
test_num = 10
cleaning_num = 80

def mecab(s):
    m = MeCab.Tagger ("-Owakati")
    s = re.sub(r"\n", "",s)
    return m.parse(s)

def lowercasing(s):
    return s.lower()

def cleaning(ja_words,en_words):
    # Trueはどちらかがcleaning_num文字以上
    if len(ja_words.split()) >= cleaning_num or len(en_words.split()) >= cleaning_num :
        return True
    else:
        return False

jvocab = {}
evocab = {}
id2wd = {}
jlines = {}
elines = {}
accum_loss = 0
with open(path_train,'r',encoding='utf-8') as f:
    lines_je = f.read().strip().split('\n')
    pairs = [[words for i,words in enumerate(line.split('|||')) if i > 2] for k,line in enumerate(lines_je) if k < train_num]

    for i in range(train_num):
        ja_words = mecab(pairs[i][0])
        en_words = lowercasing(pairs[i][1])
        
        if cleaning(ja_words,en_words):
            continue
        
        for ja_word in ja_words.split():
            if ja_word not in jvocab:
                jvocab[ja_word] = len(jvocab)

        for en_word in en_words.split():
            if en_word not in evocab:
                id = len(evocab)
                evocab[en_word] = id
                id2wd[id] = en_word
        # listを作る
        jlines[i] = mecab(pairs[i][0])
        elines[i] = lowercasing(pairs[i][1])

    jvocab['<eos>'] = len(jvocab)
    jv = len(jvocab)
    
    id = len(evocab)
    evocab['<eos>'] = id
    id2wd[id] = "<eos>"
    ev = len(evocab)


class MyMT(chainer.Chain):
    def __init__(self, jv, ev, k):
        super(MyMT, self).__init__( 
            embedx = L.EmbedID(jv,k),
            embedy = L.EmbedID(ev,k),
            H = L.LSTM(k,k),
            W = L.Linear(k, ev),
        )

    def __call__(self, jline ,eline):
        global accum_loss
        self.H.reset_state()
        for i in range(len(jline)):
            wid = jvocab[jline[i]]
            x_k = self.embedx(Variable(xp.array([wid], dtype=xp.int32)))
            h = self.H(x_k)
        x_k = self.embedx(Variable(xp.array([jvocab["<eos>"]],dtype=xp.int32)))
        tx = Variable(xp.array([evocab[eline[0]]], dtype=xp.int32))
        
        h = self.H(x_k)
        accum_loss = F.softmax_cross_entropy(self.W(h), tx)
        for i in range(len(eline)):
            wid = evocab[eline[i]]
            x_k = self.embedy(Variable(xp.array([wid], dtype=xp.int32)))
            next_wid = evocab["<eos>"] if (i == len(eline) -1) else evocab[eline[i+1]]
            tx = Variable(xp.array([next_wid], dtype=xp.int32))
            h = self.H(x_k)
            
            loss = F.softmax_cross_entropy(self.W(h),tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

demb = 64
batch_size = 128
model = MyMT(jv, ev, demb)

gpu_device = 0
cuda.get_device(gpu_device).use()
model.to_gpu()
xp = cuda.cupy
optimizer = optimizers.SGD()
optimizer.setup(model)

start = time.time()

for epoch in range(10):
    print("epoch",epoch)
    for i in range(train_num):
        jlnr = jlines[i].split()[::-1]
        elnr = elines[i].split()
        model.H.reset_state()
        model.cleargrads()
        loss = model(jlnr, elnr)
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

jlines = {}
with open(path_test,'r',encoding='utf-8') as f:
    lines_je = f.read().strip().split('\n')
    
    pairs = [[words for i,words in enumerate(line.split('|||')) if i == 2] for k,line in enumerate(lines_je) if k < test_num]

    for i in range(test_num):
        ja_words = mecab(pairs[i][0])
        if len(ja_words.split()) >= cleaning_num:
            continue
        
        for ja_word in ja_words.split():
            if ja_word not in jvocab:
                jvocab[ja_word] = len(jvocab)
        # listを作る
        jlines[i] = mecab(pairs[i][0])


def mt(model, jline):
    ans_en = []
    model.H.reset_state()
    for i in range(len(jline)):
        wid = jvocab[jline[i]]
        with chainer.using_config('train', False):
            x_k = model.embedx(Variable(xp.array([wid], dtype=xp.int32)))
        h = model.H(x_k)
    with chainer.using_config('train', False):
        x_k = model.embedx(Variable(xp.array([jvocab["<eos>"]],dtype=xp.int32)))

    h = model.H(x_k)
    wid = xp.argmax(F.softmax(model.W(h)).data[0])
    
    loop = 0
    print("wid", wid)
    while(wid != evocab['<eos>']) and (loop <= 30):
        with chainer.using_config('train', False):
            x_k = model.embedx(Variable(xp.array([wid],dtype=xp.int32)))
        h = model.H(x_k)
        wid = xp.argmax(F.softmax(model.W(h)).data[0])
        wid = int(cuda.to_cpu(wid))
        loop +=1
        ans_en.append(id2wd[wid])
    print("ans_en", ans_en)
    # if wid != evocab['<eos>']:
    #     print(','.join(ans_en).replace(",<eos>", ""))

for i in range(len(jlines)-1):
    jln = jlines[i].split()
    jlner = jln[::-1]
    print(jlner)
    mt(model, jlner)

elapsed_time = time.time() - start
print("時間:",elapsed_time / 60.0, "分")