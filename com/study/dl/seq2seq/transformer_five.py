# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 16:34
# @Author  : piguanghua
# @FileName: transformer_five.py
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
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from numpy import random
import matplotlib.pyplot as plt
from torchtext import data
from tqdm import tqdm

class Embedding(nn.Module):
    def __init__(self, vocab, dim):
        self.embed = nn.Embedding(vocab, dim)

    def forward(self, text):
        return self.embed(text)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        pass

    def forward(self, q, k, mask):
        pass

class MultiHeadAttention(nn.Module):
    def __init__(self, head, dim, d_k):
        self.head = head
        self.dim = dim
        self.d_k = d_k
        self.Q = nn.Linear(self.dim, self.d_k * self.head)
        self.K = nn.Linear(self.dim, self.d_k * self.head)
        self.V = nn.Linear(self.dim, self.d_k * self.head)

        self.Z = nn.Linear(self.d_k * self.head, self.dim)

    def forward(self, embed):
        pass


class FFN(nn.Module):
    def __init__(self, dim, internal=1024):
        self.fc1 =  nn.Linear(dim, internal)
        self.fc2 = nn.Linear(internal, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, data):
        temp = self.fc1(data)
        temp = self.fc2(temp)

        return self.norm(temp + data)


class EncoderLayer(nn.Module):
    def __init__(self, head, dim, d_k):
        self.head = head
        self.dim = dim
        self.d_k = d_k
        self.multi_head_attention = MultiHeadAttention(self.head, self.dim, self.d_k)
        self.ffn = FFN(self.dim)

    def forward(self, data):
        data, atten = self.multi_head_attention(data)
        return self.ffn(data),atten

class Encoder(nn.Module):
    def __init__(self, head, dim, d_k, layer, vocab):
        self.head = head
        self.dim = dim
        self.d_k = d_k
        self.vocab = vocab

        self.embed = Embedding( self.vocab, self.dim)
        self.encoder_block = nn.ModuleList([ EncoderLayer(self.head, self.dim, self.d_k) for i in range(layer) ])

    def forward(self, text):
        input = self.embed(text)

        attentions = []
        for layer in self.encoder_block:
            input,atten = layer(input)
            attentions.append(atten)

        return input, attentions


class MyDataset(data.Dataset):
    def __init__(self, path, text_field, label_field, test=False, aug=False, **kwargs):
        fields = [("id", None), ("comment_text", text_field), ("toxic", label_field)]
        examples = []

        

        csv_data = pd.read_csv(path)
        print('read data from {}'.format(path))

        if test:
            for text in tqdm(csv_data['comment_text']):
                examples.append(data.Example.fromlist([None, text, None], fields))
        else:
            for text, label in tqdm(zip(csv_data['comment_text'], csv_data['toxic'])):
                if aug:
                    rate = random.random()
                    if rate > 0.5:
                        text = self.dropout(text)
                    else:
                        text = self.shuffle(text)
                examples.append(data.Example.fromlist([None, text, label - 1], fields))
        super(MyDataset, self).__init__(examples, fields, **kwargs)

    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    def dropout(self, text, p=0.5):
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)



if __name__ == '__main__':

