import torch
from torch import tensor as tt
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
import time
import torch.optim as optim
from operator import itemgetter
from define_variable import *
from model import *

def train(encoder, decoder, source_lines, target_lines):
    loss = 0
    max_num =  len(target_lines) # paddingの数
    target_before = target_lines[:(max_num-1)]
    target_next = target_lines[1:]

    lhx_layer = []
    lcx_layer = []
    for i in range(args.layer_num):
        lhx_layer.append(encoder.init())
        lcx_layer.append(encoder.init())
    torch.stack(lhx_layer, 0)
    torch.stack(lcx_layer, 0)
    for sentence_words in source_lines:
        lhx_layer, lcx_layer = encoder(sentence_words, lhx_layer, lcx_layer)

    for target_b , target_n in zip(target_before, target_next):
        lhx_layer, lcx_layer = decoder(target_b, lhx_layer, lcx_layer)
        loss += F.cross_entropy(decoder.linear(lhx_layer[args.layer_num - 1]), target_n, ignore_index=0)
    return loss

if __name__ == '__main__':
    start = time.time()
    device = torch.device('cuda:0')
    model = EncoderDecoder(ev, jv, args.embed_size , args.hidden_size).to(device)
    model.train()
    optimizer = torch.optim.Adam( model.parameters(), weight_decay=args.weightdecay)
    for epoch in range(args.epoch):
        print("epoch",epoch + 1)
        indexes = torch.randperm(args.train_size)
        for i in range(0, args.train_size, args.batch_size):
            batch_input_lines = [ input_lines_number[int(index)] for index in indexes[i:i+args.batch_size]]
            batch_input_paddings = Padding(batch_input_lines)
            Transposed_input = batch_input_paddings.t().cuda()

            batch_target_lines = [ target_lines_number[int(index)] for index in indexes[i:i+args.batch_size]]
            if (epoch + 1) == 1:
                for batch_target_line in batch_target_lines:
                    batch_target_line.insert(0, target_vocab['<bos>'])
                    batch_target_line.append(target_vocab['<eos>'])

            batch_target_paddings = Padding(batch_target_lines)
            Transposed_target = batch_target_paddings.t().cuda()

            optimizer.zero_grad()
            loss = train(model.encoder, model.decoder, Transposed_input, Transposed_target)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % args.epoch == 0:
            outfile = "trained_model/attention_unk_layer-" + str(args.layer_num) + "-epoch" + str(epoch + 1) + ".model"
            torch.save(model.state_dict(), outfile)
        elapsed_time = time.time() - start
        print("時間:",elapsed_time / 60.0, "分")
