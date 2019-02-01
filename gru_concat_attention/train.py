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
    logger.debug("epoch : " + str(epoch))
    logger.debug("batch size : " +  str(batch_size))

    start = time.time()
    device = torch.device('cuda:0')
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    data_set = MyDataset(train_source, train_target)
    train_iter = DataLoader(data_set, batch_size=batch_size, collate_fn=data_set.collater, shuffle=True)

    model = EncoderDecoder(source_size, target_size, hidden_size).cuda(device=device)
    model.train()
    optimizer = torch.optim.Adam( model.parameters(), lr=1e-3, weight_decay=1e-6)

    max_score = 0
    score = 0

    save_model_dir = "{}/{}".format("trained_model", args.save_path)

    for epoch in range(args.epoch):
        print("epoch",epoch + 1)
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

        if (epoch + 1) == args.epoch and args.mode == "train":
            save_model_filename = save_model_dir + str(epoch + 1) + ".model"
            torch.save(model.state_dict(), save_model_filename)
        elapsed_time = time.time() - start
        print("時間:",elapsed_time / 60.0, "分")
