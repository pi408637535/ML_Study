import torch as t
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchtext import data
from torchtext import datasets
from sklearn.metrics import accuracy_score
import time

MAX_VOCAB_SIZE = 25002
BATCH_SIZE = 64
SEED = 1
HIDDEN_SIZE = 256
OUTPUT_SIZE = 1
LAYER = 1
EMBED_SIZE = 100

USE_CUDA = t.cuda.is_available()


class MyModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, layer, output_size):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.rnn_type = "LSTM"
        self._hidden_size = hidden_size
        self._layer = layer

    # text: batch,seq_len
    # hidden: batch,hidden_size
    def forward(self, text, hidden):
        # embedding: batct, seq_len, embed_size,
        embedding = self.embedding(text)

        # output: batch,seq_len, direction*hidden_size
        # hidden:(h_0, c_0)
        # h_0: batch, layers*diection,hidden_size
        # c_0: batch, layers*diection,hidden_size
        print(hidden[0].shape)
        output, hidden = self.lstm(embedding)
        # print("aaaa",hidden[0].shape)

        out = self.fc(t.squeeze(hidden[0]))
        # print("out",out.shape)
        return out, hidden

    def init_hidden(self, batch, requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad),
                    weight.new_zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad))
        else:
            return weight.new_zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad)


def binary_accuracy(preds, y):
    rounded_preds = t.round(t.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, crit):
    epoch_loss, epoch_acc = 0., 0.

    model.train()
    hidden = model.init_hidden(BATCH_SIZE)
    for batch in iter(iterator):
        data, target = batch.text, batch.label
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        data.t_()
        output, hidden = model(data, hidden)

        print("train", target.shape, output.shape)
        loss = crit(t.unsqueeze(target, -1), output)
        acc = binary_accuracy(output, t.unsqueeze(target, -1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(iterator)


def eval(model, iterator, optimizer, crit):
    epoch_loss, epoch_acc = 0., 0.

    model.eval()
    hidden = model.init_hidden(BATCH_SIZE)
    for batch in iter(iterator):
        data, target = batch.text, batch.label
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        data.t_()
        output, hidden = model(data, hidden)

        # print("train",target.shape, output.shape)
        loss = crit(t.unsqueeze(target, -1), output)
        acc = binary_accuracy(output, t.unsqueeze(target, -1))

        epoch_loss += loss.item()
        epoch_acc += acc

    model.train()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    t.manual_seed(SEED)
    t.cuda.manual_seed(SEED)
    # t.backends.cudnn.deterministic = True #的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    TEXT = data.Field(tokenize='spacy')
    LABEL = data.LabelField(dtype=t.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    print("train len", len(train_data))
    print("test len", len(test_data))

    # print(vars(train_data.exanples[0]))

    import random

    train_data, valid_data = test_data.split(random_state=random.seed(SEED))

    # TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors = "glove.6B.100d", unk_init = t.Tensor().normal_)
    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)

    LABEL.build_vocab(train_data)

    # print("text.vocab", len(TEXT.vocab))
    # print("label.vocab", len(LABEL.vocab))

    device = t.device("cuda" if USE_CUDA else "cpu")

    # BucketIterator将长度差不多的句子放到同一个batch中,使每个batch中不会出现太多的padding
    # 好一点的处理，把pad产生的输出要消除
    # train_iterator: seq_len,batch
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device
    )

    # batch = next(iter(train_iterator))
    # print(batch.text)

    # PAD_IDX = TEXT.vocab.stoi(TEXT.pad_token)
    # UNK_IDX = TEXT.vocab.stoi(TEXT.unk_token)
    # pretrained_embedding = TEXT.vocab.vectors
    # model.embed.weight.data.copy_(pretrained_embedding)
    # model.embed.weight.data[PAD_INDEX] = t.zeros(EMBEDDING_SIZE)
    # model.embed.weight.data[UNK_IDX] = t.zeros(EMBEDDING_SIZE)

    lr = 1e-4
    model = MyModel(MAX_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, LAYER, OUTPUT_SIZE)
    optimizer = optim.Adam(model.parameters(), lr)
    crit = nn.BCEWithLogitsLoss()
    model = model.to(device)
    crit = crit.to(device)

    N_EPOCHS = 10
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, crit)
        valid_loss, valid_acc = eval(model, train_iterator, optimizer, crit)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print("saving model")

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')










