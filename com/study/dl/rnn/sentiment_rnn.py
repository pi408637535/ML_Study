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

MAX_VOCAB_SIZE = 25004
BATCH_SIZE = 64
SEED = 1
HIDDEN_SIZE = 256
OUTPUT_SIZE = 1
LAYER = 1
EMBED_SIZE = 100

USE_CUDA = t.cuda.is_available()
if USE_CUDA:
    device = t.device("cuda")
else:
    device = t.device("cpu")

class MyModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, layer, output_size):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, layer, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.rnn_type = "LSTM"
        self._hidden_size = hidden_size
        self._layer = layer

    # text: batch,seq_len
    # hidden: batch,hidden_size
    def forward(self, text, hidden):
        # embedding: batch, seq_len, embed_size,
        embedding = self.embedding(text)

        # output: batch, seq_len, direction*hidden_size
        # hidden:(h_0, c_0)
        # h_0: layers*diection, batch, ,hidden_size
        # c_0: layers*diection, batch, ,hidden_size

        output, hidden = self.lstm(embedding)

        #out:batch,output_size
        out = self.fc(t.squeeze(hidden[0]))
        return out, hidden

    def init_hidden(self, batch, requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad),
                    weight.new_zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad))
        else:
            return weight.new_zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad)


class SentimentNet(nn.Module):
    def __init__(self, batch, vocab_size, embed_size, hidden_size, layer, output_size, drop_prob=0.5):
        super(SentimentNet, self).__init__()
        self._hidden_size = hidden_size
        self._layer = layer
        self._batch = batch
        self._output_size = output_size


        self._embedding = nn.Embedding(vocab_size, embed_size)
        self._lstm = nn.LSTM(
            vocab_size,
            hidden_size,
            layer,
            batch_first=True
        )
        self._fc = nn.Linear(hidden_size, output_size)
        self._sigmod = F.sigmoid()

    #text: batch,seq_len
    #h_state: layer * direction, batch, hidden_size
    def forward(self, text, h_state):

        #embed: batch,seq_len,embed_size
        embed = self._embedding(text)
        #output: batch,seq_len,direction_h_state
        #h_state: layer*direction, batch, hidden_size
        output,h_state = self._lstm(embed, h_state)

        output = output.continuous().view(-1, self._hidden_size)
        #output: batch * seq_len, output
        output = self._fc(output)

        #output: batch * seq_len, output
        output = self._sigmod(output)

        #seq_len: it's dynamic in running.
        #output: batch, seq_len * output_size
        output = output.view(self._batch, -1)

        output = output[:, -1]

        return output, h_state

    def init_hidden(self, batch, requires_grad=True):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            hidden = (weight.new(self._layer, batch, self._hidden_size).zero_().to(device),
                      weight.new(self._layer, batch, self._hidden_size).zero_().to(device))
            return hidden
        else:
            return t.zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad)


def binary_accuracy(preds, y):
    rounted_preds = t.round(t.sigmoid(preds))
    return accuracy_score(rounted_preds.cpu().numpy(), y.numpy())


def train(model, iterator, optimizer, crit):
    epoch_loss, epoch_acc = 0., 0.

    model.train()
    hidden = model.init_hidden(BATCH_SIZE)
    for batch in iter(iterator):
        data, target = batch.text, batch.label
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        data.t_()
        print("data=",data.shape)
        output, hidden = model(data, hidden)
        print("output=",output.shape,"target=",target.shape )
        loss = crit(t.squeeze(output), target)
        #acc = binary_accuracy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(loss.item())
        #epoch_acc += acc

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
        loss = crit(output, target)

        acc = binary_accuracy(output, target)
        epoch_loss += loss.item() * len(batch)
        epoch_acc += acc * len(batch)

        optimizer.step()

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

    imdb_dir = '/home/demo1/womin/piguanghua/.data/imdb/aclImdb'
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, path=imdb_dir)
    print("train len", len(train_data))
    print("test len", len(test_data))

    # print(vars(train_data.exanples[0]))

    import random

    train_data, valid_data = test_data.split(random_state=random.seed(SEED))

    # TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors = "glove.6B.100d", unk_init = t.Tensor().normal_)
    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)

    LABEL.build_vocab(train_data)

    print("text.vocab", len(TEXT.vocab))
    print("label.vocab", len(LABEL.vocab))

    #device = t.device("cuda" if USE_CUDA else "cpu")

    # BucketIterator将长度差不多的句子放到同一个batch中,使每个batch中不会出现太多的padding
    # 好一点的处理，把pad产生的输出要消除
    # train_iterator: seq_len,batch
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device
    )

    batch = next(iter(train_iterator))
    print(batch.text)

    # PAD_IDX = TEXT.vocab.stoi(TEXT.pad_token)
    # UNK_IDX = TEXT.vocab.stoi(TEXT.unk_token)
    # pretrained_embedding = TEXT.vocab.vectors
    # model.embed.weight.data.copy_(pretrained_embedding)
    # model.embed.weight.data[PAD_INDEX] = t.zeros(EMBEDDING_SIZE)
    # model.embed.weight.data[UNK_IDX] = t.zeros(EMBEDDING_SIZE)

    lr = 1e-4
    #batch, vocab_size, embed_size, hidden_size, layer, output_size
    model = SentimentNet(BATCH_SIZE, MAX_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, LAYER, OUTPUT_SIZE)
    #model = SentimentNet(MAX_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, LAYER, OUTPUT_SIZE)
    optimizer = optim.Adam(model.parameters(), lr)
    #crit = nn.BCEWithLogitsLoss()
    crit = nn.BCELoss()
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










