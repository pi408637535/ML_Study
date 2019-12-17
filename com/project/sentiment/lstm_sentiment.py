import os
from  tqdm import tqdm
from collections import Counter
import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
import re
from torchtext import vocab as vc
import itertools
import random

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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#/home/demo1/womin/piguanghua/.data/imdb/aclImdb
imdb_dir = '/home/demo1/womin/piguanghua/.data/imdb/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
test_dir = os.path.join(imdb_dir, 'test')
## Load the each text into texts list
## corresponding label will be in labels list.
def read_test_train_dir(path,):
    labels = []
    texts = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
    return texts,labels

train_texts,train_labels = read_test_train_dir(train_dir)
test_texts, test_labels = read_test_train_dir(test_dir)

def get_paragraph_words(text):
    return (flatten([word_tokenize(s) for s in sent_tokenize(text)]))

sent_tokenize = nltk.sent_tokenize
word_tokenize = RegexpTokenizer(r'\w+').tokenize



def word_tokenize_para(text):
    return [word_tokenize(s) for s in sent_tokenize(text)]

def flatten(l):
    return [item for sublist in l for item in sublist]


vocab_counter = Counter(flatten([get_paragraph_words(text) for text in train_texts]))
w2v = vc.Vocab(vocab_counter, max_size=20000, min_freq=3)


# function to get vocabular indices from text returns list of indices (cut-off at maxSeqLength)
def stoiForReview(w2v, text, maxSeqLength):
    # trim the sentence to maxSeqLength, otherwise return with original length.
    return [w2v.stoi[word] for word in get_paragraph_words(text)[0:maxSeqLength]]


# function to get word vectors for review - returns tensor of size 1, min(len(review),maxSeqLength),embedded_dim
def wordVectorsForReview(w2v, text, maxSeqLength):
    indexes = stoiForReview(w2v, text, maxSeqLength)
    # returns tensor with size [num_words,1,embedding_dim]
    # That extra 1 dimension is because PyTorch assumes everything is in batches - we’re just using a batch size of 1 here.

    #error
    #sent_word_vectors = t.cat([w2v.vectors[i].view(1, -1) for i in indexes]).view(len(indexes), 1, -1)

    # batch first (1,seq_len,embedding_dim)
    # seq_len has been maximized to maxSeqLength

    # error
    #sent_word_vectors = sent_word_vectors.view(1, len(sent_word_vectors), -1)

    return t.unsqueeze(t.Tensor(indexes),0)

def get_batch(t_set, str_idx, end_idx):
    training_batch_set = t_set[str_idx:end_idx]

    input_texts, labels = zip(*training_batch_set)

    # convert texts to vectors shape - Batch(=1),seq_length(cut-off at maxSeqLength),embedded_dim
    input_vectors = [wordVectorsForReview(w2v, text, maxSeqLength) for text in input_texts]

    # convert to variable w/ long tensor
    labels = t.LongTensor(labels)

    seq_lens = t.LongTensor([i.shape[1] for i in input_vectors])

    # batch_inputs  - [batch_size, seq_len ]
    batch_inputs = t.zeros((len(seq_lens), seq_lens.max() ))
    for idx, (seq, seqlen) in enumerate(zip(input_vectors, seq_lens)):
        batch_inputs[idx, :seqlen] = seq
    seq_lens, perm_idx = seq_lens.sort(0, descending=True)
    batch_inputs = batch_inputs[perm_idx]
    batch_inputs = pack_padded_sequence(batch_inputs, seq_lens.numpy(), batch_first=True)
    labels = labels[perm_idx]
    return (batch_inputs, labels)

USE_CUDA = t.cuda.is_available()
MAX_VOCAB_SIZE = 25004
BATCH_SIZE = 64
SEED = 1
HIDDEN_SIZE = 256
OUTPUT_SIZE = 1
LAYER = 1
EMBED_SIZE = 100


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

def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = t.round(t.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def train(model, inputs, labels, optimizer, crit):
    epoch_loss, epoch_acc = 0., 0.

    model.train()
    hidden = model.init_hidden(BATCH_SIZE)
    for batch in iter(iterator):
        data, target = batch.text, batch.label
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        data.t_()
        output, hidden = model(data, hidden)
        loss = crit(output, t.unsqueeze(target, -1))
        acc = binary_accuracy(output, target)

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
        loss = crit(output, target)

        acc = binary_accuracy(output, t.unsqueeze(target, -1))
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

    print(len(vocab_counter))

    # randomly shuffle the training data
    training_set = list(zip(train_texts, train_labels))
    # shuffle works inplace and returns None .
    random.shuffle(training_set)

    # randomly shuffle the training data
    testing_set = list(zip(test_texts, test_labels))
    # shuffle works inplace and returns None .
    random.shuffle(testing_set)

    maxSeqLength = 250


    batch_size = 50
    num_passes = int(25000 / batch_size)

    lr = 1e-4
    device = t.device("cuda" if USE_CUDA else "cpu")
    model = MyModel(MAX_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, LAYER, OUTPUT_SIZE)
    optimizer = optim.Adam(model.parameters(), lr)
    crit = nn.BCEWithLogitsLoss()
    model = model.to(device)
    crit = crit.to(device)

    N_EPOCHS = 10
    best_valid_loss = float('inf')

    num_epochs = 10
    for epoch in range(num_epochs):
            for i in range(num_passes):
                str_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                inputs, labels = get_batch(training_set, str_idx, end_idx)

                start_time = time.time()

                optimizer.zero_grad()  # zero the gradient buffer
                outputs, hidden = model(inputs, None)
                loss = crit(outputs, labels)
                loss.backward()
                optimizer.step()

                end_time = time.time()

                if(i + 1) % 100 == 0:
                    print('pass [%d/%d], in epoch [%d/%d] Loss: %.4f'
                        % (i + 1, num_passes, epoch, num_epochs, loss.data[0]))
