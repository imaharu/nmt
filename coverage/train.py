import torch
from torch import tensor as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
import time
import torch.optim as optim
from define import *
from model import *
from tqdm import tqdm
from dataset import *
from evaluate_util import *

def train(model, source, target):
    loss = model(source=source, target=target, train=True)
    return loss

if __name__ == '__main__':
    if args.is_short_data:
        logger.debug("訓練文書数: " +  str(20000))
    else:
        logger.debug("訓練文書数: " +  str(100000))
    logger.debug("hidden_size: " + str(hidden_size))
    logger.debug("embed_size: " +  str(embed_size))
    logger.debug("epoch : " + str(epochs))
    logger.debug("batch size : " +  str(batch_size))

    start = time.time()
    device = torch.device('cuda:0')
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    data_set = MyDataset(train_source, train_target)
    train_iter = DataLoader(data_set, batch_size=batch_size, collate_fn=data_set.collater, shuffle=True)

    val_set = EvaluateDataset(val_source)
    val_iter = DataLoader(val_set, batch_size=1, collate_fn=val_set.collater)
    opts = { "bidirectional" : args.none_bid, "coverage_vector": args.coverage }
    model = EncoderDecoder(source_size, target_size, opts).cuda(device=device)
    if args.set_state:
        optimizer = torch.optim.Adam( model.parameters(), lr=1e-3, weight_decay=1e-6)
        set_epoch = 0
    else:
        checkpoint = torch.load("trained_model/{}".format(str(args.model_path)))
        epochs -= checkpoint['epoch']
        set_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer = torch.optim.Adam( model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])

    model.train()
    print(model)

    max_score = 0
    score = 0
    save_model_dir = "{}/{}".format("trained_model", args.save_path)
    best_model_dir = "{}/{}".format("trained_model", "best-model")

    calc_blue = Evaluate(target_dict, val=1, gold_sentence_file=val_ja, val_iter=val_iter)

    for epoch in range(epochs):
        real_epoch = epoch + set_epoch + 1
        print("epoch", real_epoch)
        tqdm_desc = "[Epoch{:>3}]".format(epoch + 1)
        tqdm_bar_format = "{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        tqdm_kwargs = {'desc': tqdm_desc, 'smoothing': 0.1, 'ncols': 100,
                    'bar_format': tqdm_bar_format, 'leave': False}

        model.train()
        for iters in tqdm(train_iter, **tqdm_kwargs):
            optimizer.zero_grad()
            loss = train(model, iters[0].cuda(), iters[1].cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

#        score = calc_blue.GetBlueScore(model)
#        print("mac_score: {}".format(max_score))
#        print("score: {}".format(score))
#        if max_score < score:
#            max_score = score
#            best_model_filename = "{}-epoch{}{}".format(save_model_dir, str(epoch + 1),".model")
#            torch.save(model.state_dict(), best_model_filename)

        if (real_epoch) == args.epoch or (real_epoch) % 2 == 0 and args.mode == "train":
            if not os.path.exists(save_model_dir):
                os.mkdir(save_model_dir)
            save_model_filename = "{}/epoch-{}.model".format(save_model_dir, str(real_epoch))
            states = {
                'epoch': real_epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_model_filename)

        elapsed_time = time.time() - start
        print("時間:",elapsed_time / 60.0, "分")
