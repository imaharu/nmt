#source_path = "../train_data/100000.train.en"
#target_path = "../train_data/100000.train.ja"
source_path = "../../train_data/train.en"
target_path = "../../train_data/train.ja"

def GetVocab(file_name):
    vocab = {}
    with open(file_name) as lines:
        for line in lines:
            words = line.split()
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab) + 1
    return vocab

source_vocab = GetVocab(source_path)
target_vocab = GetVocab(target_path)

with open("source_20000vocab", "w") as f:
    for key in source_vocab.keys():
        f.write(key + "\n")

with open("target_20000vocab", "w") as f:
    for key in target_vocab.keys():
        f.write(key + "\n")
