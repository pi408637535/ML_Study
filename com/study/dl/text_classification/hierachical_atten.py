# -*- coding: utf-8 -*-
# @Time    : 2020/5/7 11:17
# @Author  : piguanghua
# @FileName: hierachical_atten.py
# @Software: PyCharm


import os
import sys
import math
from collections import Counter
import numpy as np
import random

import torch.nn as nn
import torch.nn.functional as F

import torch as t
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import csv
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


import nltk

#hypotrical parameter
USE_CUDA = t.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
t.manual_seed(53113)
if USE_CUDA:
    t.cuda.manual_seed(53113)

EMBEDDING_SIZE = 300
HIDDEN_SIZE = 500
EPOCHS = 10
UNK_IDX = 0
PAD_IDX = 1
NUM_EPOCHS = 1
BATCH_SIZE = 1  # the batch size
LEARNING_RATE = 1e-3  # the initial learning rate


def pre_train():
    class CustomDataSet(Dataset):

        def __init__(self):
            self.data = self._load_dataset()

        def _load_dataset(self):
            all_data = []
            for i in range(4):
                data = {
                    'source': np.random.randint(5, size = (25,30)),
                    'target': 1,
                }
                all_data.append(data)
            return all_data

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return self.data[index]

    return CustomDataSet()

class Sentence_Atten(nn.Module):
    #Uw hyper-parameter
    def __init__(self, hidden, Us):
        super(Sentence_Atten, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden * 2, Us, bias=True),
            nn.Tanh(),
            nn.Linear(Us, 1)
        )

    def forward(self, sentence_hidden):
        #sentence_hidden: batch,seq,hidden
        word_hidden = sentence_hidden.float()
        Us_dot_Ui =  self.net(word_hidden)  #u_it batch,seq, 1
        Uw_dot_Uit = t.squeeze(Us_dot_Ui, dim=2)
        scores =  F.softmax(Uw_dot_Uit,dim=1)
        scores = t.unsqueeze(scores, dim= 2) #scores: batch,seq,1
        v = t.sum(scores * word_hidden, dim= 1)  #v:batch, hidden
        return v

class Word_Atten(nn.Module):
    #Uw hyper-parameter
    def __init__(self, hidden, Uw):
        super(Word_Atten, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden * 2, Uw, bias=True),
            nn.Tanh(),
            nn.Linear(Uw, 1)
        )

    def forward(self, word_hidden):
        #word_hidden: batch,seq,hidden
        word_hidden = word_hidden.float()
        Uw_dot_Uit =  self.net(word_hidden)  #u_it batch,seq, 1
        Uw_dot_Uit = t.squeeze(Uw_dot_Uit, dim=2)
        scores =  F.softmax(Uw_dot_Uit,dim=1)
        scores = t.unsqueeze(scores, dim= 2) #scores: batch,seq,1
        s_i = t.sum(scores * word_hidden, dim= 1)  #dim= 1基于senquence   ,si:batch, hidden
        return s_i

class HAN(nn.Module):
    def __init__(self, vocab, dim, hidden, layer, word_atten, sentence_atten,
                 sentence_hidden,class_num,dropout = 0.2,):
        super(HAN, self).__init__()
        self.word_embed = nn.Embedding(vocab, dim)
        self.word_gru = nn.GRU(dim, hidden, layer, batch_first = True,
                          dropout=dropout, bidirectional= True)

        self.sentence_gru = nn.GRU(hidden * 2, sentence_hidden, layer, batch_first=True,
                          dropout=dropout, bidirectional=True)

        self.word_atten = word_atten
        self.sentence_atten = sentence_atten
        self.out = nn.Sequential(
            nn.Linear(hidden * 2, class_num),
            nn.LogSoftmax(dim = 1)
        )


    def forward(self, sentences):
        #sentences: batch,sentence, words
        #word_embed: batch, sentences, words, embeds
        word_embed = self.word_embed(sentences)

        #input: batch, seq, hidden
        #h_0: layer, batch, hidden
        batch, sentences, words = sentences.shape[0], sentences.shape[1], sentences.shape[2]
        word_embed = word_embed.contiguous().view(batch * sentences, words, -1 )


        output, h_0 = self.word_gru(word_embed)
        output = output.contiguous().view(batch,sentences, words, -1 ) #output:batch,sentences,words,embed
        
        s_stack = []
        for index in range(sentences):
            fragment = output[:, index, :, :] # fragment：batch,seq,hidden
            sentences_vector_fragment = self.word_atten(fragment) #sentences_vector_fragment: batch,hidden
            s_stack.append(sentences_vector_fragment)  #需要按照列进行堆叠

        sentences_vector = t.stack(s_stack, dim = 1) #sentences_vector:batch,sentences,hidden

        #sentences_vector =  self.word_atten(output) #sentences_vector:batch,sentences,hidden

        sentence_output, _ = self.sentence_gru(sentences_vector)
        phrase_vector =  self.sentence_atten(sentence_output) #phrase_vector:batch,hidden

        result = self.out(phrase_vector)
        return result


def train(model, train_dataloader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):

        for batch_id, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()

            source = batch_data['source'].to(device)
            target = batch_data['target'].to(device)



            output = model(source)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()


if __name__ == '__main__':
    train_dataloader = pre_train()
    train_dataloader = DataLoader(dataset=train_dataloader, batch_size=BATCH_SIZE)


    vocab, dim, hidden, layer, sentence_hidden, class_num = 10, 20, 20, 2, 20,5
    sentence_atten = Sentence_Atten(hidden, dim)
    word_atten = Word_Atten(hidden, dim)

    model = HAN(vocab, dim, hidden, layer, word_atten, sentence_atten,
                 sentence_hidden, class_num)
    device = torch.device('cuda' if False else 'cpu')
    optimizer = t.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    train(model, train_dataloader, optimizer, criterion, EPOCHS, device)





