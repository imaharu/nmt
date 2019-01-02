import torch
from torch import tensor as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
import time
import torch.optim as optim
from define_variable import *
from model import *
from tqdm import tqdm

def train(model, source, target):
    loss = torch.mean(model(source, target), 0)
    return loss

if __name__ == '__main__':
    logger.debug("訓練文書数: " +  str(train_doc_num))
    logger.debug("hidden_size: " + str(hidden_size))
    logger.debug("embed_size: " +  str(embed_size))
    logger.debug("epoch : " + str(epoch))
    logger.debug("batch size : " +  str(batch_size))

    start = time.time()
    device = torch.device('cuda:0')
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    data_set = MyDataset(source_data, target_data)
    train_iter = DataLoader(data_set, batch_size=batch_size, collate_fn=data_set.collater, shuffle=True)

    model = EncoderDecoder(source_size, target_size, hidden_size)
    model.train()
    model = nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam( model.parameters(), lr=1e-3, weight_decay=1e-6)

    for epoch in range(args.epoch):
        print("epoch",epoch + 1)
        tqdm_desc = "[Epoch{:>3}]".format(epoch)
        tqdm_bar_format = "{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        tqdm_kwargs = {'desc': tqdm_desc, 'smoothing': 0.1, 'ncols': 100,
                    'bar_format': tqdm_bar_format, 'leave': False}

        for iters in tqdm(train_iter, **tqdm_kwargs):
            optimizer.zero_grad()
            loss = train(model, iters[0], iters[1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            outfile = "trained_model/" + str(save_path) + "/" + str(epoch + 1) + ".model"
            torch.save(model.state_dict(), outfile)
        elapsed_time = time.time() - start
        print("時間:",elapsed_time / 60.0, "分")
