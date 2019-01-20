#!/usr/bin/env python
from utils.nltk.nltk.translate.bleu_score import corpus_bleu
from utils.coco_loader_ver2 import make_loader
from utils.coco_homogeneous_data import HomogeneousData
from collections import defaultdict 
from pathlib import Path
import pickle
import torch
import torch.nn.functional as F
from torchvision import transforms
from models import CNN_Encoder, LSTM_Decoder_With_Attention, Cnn_Dec_Attention
import progressbar
from progressbar import ProgressBar
import copy

beam_size = 2
length_penalty = 0.6
encoder_size = 512
embed_size = 512
decoder_size = 1024
attention_size = 512
dropout_rate = 0.5
alpha_c = 1.0
vocabsize = 10000
gpu = 0
limit = False
max_length = 50

dataDir = '/home/nakamoto/workspace/Pytorch/Caption/dataset/coco/'
#modelDir = '/home/nakamoto/workspace/Pytorch/Caption/result/test/full/vgg19/'
#modelDir = '/home/nakamoto/workspace/Pytorch/Caption/result/homogeneous_yunjey_xavier/orig_data/full/vgg19/'
modelDir = '/home/nakamoto/workspace/Pytorch/PairUnitCaption/result/1104_vgg19/full/vgg19/'
#trainName = 'large_coco_train2014_ptb.pkl'
#restvalName = 'coco_val2014_ptb_restval.pkl'
#valName = 'coco_val2014_ptb_val5k.pkl'
checkpoint = 'best_checkpoint.pth.tar'

trainName = 'karpathy_coco_train2014.pkl'
#restvalName = 'karpathy_coco_val2014.pkl'
valName = 'karpathy_coco_test2014.pkl'


def evaluate(model, vocab, loader, references):
    
    hypotheses = list()
    bleu_hypo = list()
    k = beam_size

    label2word = dict()
    for word, label in vocab.items():
        label2word[label] = word
    
    p = ProgressBar(0, len(loader))

    model.eval()
    for i, (images, _, _, _) in enumerate(loader):

        features = model.cnn_encoder(images.to(gpu)) #(1, 14, 14, 512)
        batchsize = features.size(0)
        channels = features.size(3)

        '''flatten images vector'''
        features = features.view(batchsize, -1, channels)
        num_pixels = features.size(1)

        '''treat the problem as having a batch size of k'''
        features = features.expand(k, num_pixels, channels) #(2, 196, 512)

        '''tensor to store top k previous words at each step; now they're just <start>'''
        k_prev_words = torch.LongTensor([[vocab['<sos>']]] * k).to(gpu) #(k, 1)

        '''tensor to store top k sequences; now they're just <start>'''
        seqs = k_prev_words

        '''tensor to store top k sequences' scores; now they're just 0'''
        top_k_scores = torch.zeros(k, 1).to(gpu) #(k, 1)

        '''list to store completed sequences and scores'''
        completed_seqs = list()
        completed_seqs_scores = list()
        
        '''start decoding'''
        h, c = model.decoder.init_hidden_state(features) #(2, 512)

        for step in range(max_length):
            words = k_prev_words.squeeze(-1).to(gpu) #[1, 1], a torch tensor of dimension (2,)
            words_embed = model.decoder.embed(words) #(2, 512)
            mask = (words != 0)
            mask = mask.unsqueeze(1)
            z, _ = model.decoder.attention(features, h, mask) #(2, 512)
            beta = model.decoder.sigmoid(model.decoder.f_beta(h)) #(2, 512)
            z = beta * z
            
            h, c = model.decoder.lstm(torch.cat((words_embed, z), dim=1), (h, c)) #(2, 512), (2, 512)
                
            out = words_embed + model.decoder.linear_h(h) + model.decoder.linear_z(z) #(2, 512)
            predicts = model.decoder.fc(out)
            scores = F.log_softmax(predicts, dim=1)
            
            scores = top_k_scores.expand_as(scores) + scores                #(2, 10004)

            if (step+1) == 1:
                top_k_scores, top_k_words = torch.topk(scores[0], k, dim=0)
            else:
                top_k_scores, top_k_words = torch.topk(scores.view(-1), k, dim=0)

            prev_word_inds = top_k_words / len(vocab)
            next_word_inds = top_k_words % len(vocab)

            seqs = torch.cat((seqs[prev_word_inds], next_word_inds.unsqueeze(1)), dim=1)
           
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != vocab['<eos>']]
           
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
           

            if len(complete_inds) > 0:
                completed_seqs.extend(seqs[complete_inds].tolist())
                completed_seqs_scores.extend(top_k_scores[complete_inds])

            if len(complete_inds) == k:
                break

            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            features = features[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            
        new_completed_scores = list()
        for seq, score in zip(completed_seqs, completed_seqs_scores):
            ln = len(seq) - 1
            lp = ((5 + ln) ** length_penalty) / ((5 + 1) ** length_penalty)
            #lp = 1
            new_completed_scores.append(score/lp)

        ind = new_completed_scores.index(max(new_completed_scores))
        seq = completed_seqs[ind][1:]
        seq = eos_truncate(seq, vocab['<eos>'])

        labels = [label2word.get(label, '<unk>') for label in seq]
        bleu_hypo.append(labels)
        hypotheses.append(' '.join(labels))
        
        p.update(i)

    bleu = corpus_bleu(references, bleu_hypo)
    print('\n nltk-bleu : {}'.format(bleu))

    with open(modelDir+'beam{}_hypotheses.txt'.format(beam_size), 'w') as f:
        print('\n'.join(hypotheses), file=f)
        
def eos_truncate(labels, eos_label):
    # eos 以降を切り捨て
    if eos_label in labels:
        eos_index = labels.index(eos_label)
        labels = labels[:eos_index]
    return labels
            
def main():
       
    train_path = Path(dataDir + trainName)
    #restval_path = Path(dataDir + restvalName)
    val_path = Path(dataDir + valName)

    '''Data Loading Code'''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
    transform = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize])

    '''辞書獲得'''
    '''
    train_loader, train_size, _ = make_loader(1, shuffle=True, num_workers=0,
                                     pin_memory=False, annPickle=train_path, rest_annPickle=restval_path,
                                     transform=transform, limit_data=limit)
    '''

    train_loader = HomogeneousData(annPickle=train_path, transform=transform, batchsize=1, \
                                   vocabsize=vocabsize, limit_data=limit)
    train_size = train_loader.__len__()
    vocab = copy.copy(train_loader.vocab)
    
    val_loader, val_size, val_refes = \
                        make_loader(1, shuffle=False, num_workers=0,
                                    pin_memory=False, annPickle=val_path,
                                    transform=transform, limit_data=True, make_reference=True)
    val_loader.dataset.vocab = vocab


    print('vocab size : {}'.format(len(vocab)))
    print('train dataset size : {}'.format(train_size))
    print('evaluate dataset size : {}'.format(val_size))
    print('beam size : {}'.format(beam_size))

    model = Cnn_Dec_Attention(CNN_Encoder(model='vgg19'),
                              LSTM_Decoder_With_Attention(len(vocab), embed_size, decoder_size,
                                    attention_size, encoder_size, dropout_rate),
                              vocab['<sos>'], vocab['<eos>'], alpha_c, False)

    model.to(gpu)
    state = torch.load(modelDir+checkpoint)
    model.load_state_dict(state['model'])
    
    evaluate(model, vocab, val_loader, val_refes)    

if __name__ == '__main__':
    main()
