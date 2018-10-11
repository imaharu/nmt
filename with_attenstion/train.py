import torch
from torch import tensor as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
import time
from get_data import *
import torch.optim as optim

train_num, hidden_size, batch_size = 20000, 256, 50
# train_num, hidden_size, batch_size = 10, 4, 2

input_vocab , input_lines, input_lines_number = {}, {}, {}
target_vocab ,target_lines ,target_lines_number = {}, {}, {}
translate_words = {}

# paddingで0を入れるから
get_train_data_input(train_num, input_vocab, input_lines_number, input_lines)
ev = len(input_vocab) + 1

get_train_data_target(train_num, target_vocab, target_lines_number, target_lines, translate_words)
jv = len(target_vocab) + 1

class EncoderDecoder(nn.Module):
    def __init__(self, source_size, output_size, hidden_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(source_size, hidden_size)
        self.decoder = Decoder(output_size, hidden_size)

class Encoder(nn.Module):
    def __init__(self, source_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.source_size = source_size
        self.embed_source = nn.Embedding(source_size, hidden_size, padding_idx=0)
        self.drop_source = nn.Dropout(p=0.2)
        self.lstm_source = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, sentence_words, hs, cs):
        source_k = self.embed_source(sentence_words)
        source_k = self.drop_source(source_k)
        hs, cs = self.lstm_source(source_k, (hs, cs) )
        return hs, cs
    
    def initHidden(self):
        hs = torch.zeros(batch_size, self.hidden_size).cuda()
        cs = torch.zeros(batch_size, self.hidden_size).cuda()
        return hs, cs

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.embed_target = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=0.2)
        self.lstm_target = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.attention_linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, target_words, hx, cx):
        target_k = self.embed_target(target_words)
        target_k = self.drop_target(target_k)
        hx, cx = self.lstm_target(target_k, (hx, cx) )
        return hx, cx
    
    def attention(self, ht, list_hs, list_source_mask, inf):
        dot = (ht * list_hs).sum(-1, keepdim=True)
        dot = torch.where(list_source_mask == 0, inf, dot)
        a_t = F.softmax( dot, 0 )
        d = (a_t * list_hs).sum(0)
        concat  = torch.cat((d, ht), 1)
        ht_new = F.tanh(self.attention_linear(concat))
        return ht_new

def create_mask(source_sentence_words):
    return torch.cat( [ source_sentence_words.unsqueeze(-1) ] * hidden_size, 1)

def train(encoder, decoder, source_lines, target_lines):
    loss = 0
    list_hs = []
    list_source_mask = []
    max_num =  len(target_lines) # paddingの数
    target_lines_not_last = target_lines[:(max_num-1)]
    target_lines_next = target_lines[1:]
    
    hs, cs = encoder.initHidden()

    for sentence_words in source_lines:
        before_hs = hs
        before_cs = cs
        hs, cs = encoder(sentence_words, hs, cs)
        mask = create_mask(sentence_words)
        hx = torch.where(mask == 0, before_hs, hs)
        cx = torch.where(mask == 0, before_cs, cs)
        list_hs.append(hs)
        masks = torch.cat( [ sentence_words.unsqueeze(-1) ] , 1)
        list_source_mask.append( torch.unsqueeze(masks, 0))

    list_hs = torch.stack(list_hs, 0)
    list_source_mask = torch.cat(list_source_mask)

    ht = hs
    ct = cs
    inf = torch.full((len(source_lines), batch_size), float('-inf')).cuda()
    inf = torch.unsqueeze(inf, -1)

    for target_sentence_words , target_sentence_words_next in zip(target_lines_not_last, target_lines_next):
        ht, ct = decoder(target_sentence_words, ht, ct)
        ht_new = decoder.attention(ht, list_hs, list_source_mask, inf)
        loss += F.cross_entropy(decoder.linear(ht_new), target_sentence_words_next, ignore_index=0)
    return loss

if __name__ == '__main__':
    start = time.time()
    device = torch.device('cuda:0')
    model = EncoderDecoder(ev, jv, hidden_size).to(device)
    model.train()
    optimizer = torch.optim.Adam( model.parameters(), weight_decay=0.002)

    for epoch in range(20):
        print("epoch",epoch + 1)
        indexes = torch.randperm(train_num)
        for i in range(0, train_num, batch_size):
            batch_input_lines = [ input_lines_number[int(index)] for index in indexes[i:i+batch_size]]
            batch_input_paddings = Padding(batch_input_lines)
            Transposed_input = batch_input_paddings.t().cuda()

            batch_target_lines = [ target_lines_number[int(index)] for index in indexes[i:i+batch_size]]
            if (epoch + 1) == 1:
                for batch_target_line in batch_target_lines:
                    batch_target_line.insert(0, target_vocab['<bos>'])
                    batch_target_line.append(target_vocab['<eos>'])

            batch_target_paddings = Padding(batch_target_lines)
            Transposed_target = batch_target_paddings.t().cuda()

            optimizer.zero_grad()
            loss = train(model.encoder, model.decoder, Transposed_input, Transposed_target)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            outfile = "attention-" + str(epoch + 1) + ".model"
            torch.save(model.state_dict(), outfile)
        elapsed_time = time.time() - start
        print("時間:",elapsed_time / 60.0, "分")