import torch as t
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
from torch.utils.data import Dataset, DataLoader
import os
import copy,time
import numpy as np

'''
lstm:
    embedding
    lstm
    fc
    train_op
数据封装模块
    next_batch
词表封装：
    setense2id(text_setnece)
类别封装：
    category2id
'''

num_embedding_size = 16 #词向量长度
num_timesteps = 50 #可以是变长
num_lstm_nodes = [32,32]
num_lstm_layers = 2
num_fc_nodes = 32
batch_size = 100
clip_lstm_grads = 1.0 #梯度爆炸,消失不改结构如果解决
num_word_threshold = 25

seg_train_file = "/Users/piguanghua/Downloads/cnews/cnews.train.seg.txt"

seg_test_file = "/Users/piguanghua/Downloads/cnews/cnews.train.test.txt"
vocab_file = "/Users/piguanghua/Downloads/cnews/cnews.train.vocab.txt"
category_file = "/Users/piguanghua/Downloads/cnews/category.txt"
seg_val_file = "/Users/piguanghua/Downloads/cnews/cnews.train.val.txt"

class Vocab:
    def __init__(self, filename, num_word_threshold):
        self._word_to_id = {}
        self._unk = -1
        self._num_word_threshold = num_word_threshold
        self._read_dict(filename)

    def _read_dict(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            word, frequency = line.split('\t')
            frequency = int(frequency)
            if frequency < self._num_word_threshold:
                continue
            idx = len(self._word_to_id) + 1
            if word == '<UNK>':
                self._unk = idx
            self._word_to_id[word] = idx

    def _word_to_id_fun(self, word):
        return self._word_to_id.get(word, self._unk)

    @property
    def unk(self):
        return self._unk
    @property
    def size(self):
        return len(self._word_to_id)

    def sentence_to_id(self, sentence):
        word_ids = [self._word_to_id_fun(cur_word) for cur_word in  sentence.split()]
        return word_ids


class CategoryDict:
    def __init__(self, filename):
        self._category_to_id = {}
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            idx = len(self._category_to_id)
            self._category_to_id[line.strip()] = idx

    def category_to_id(self, category):
        if category not in self._category_to_id:
            raise Exception("%s is not in our category" % category)
        return self._category_to_id[category]

    @property
    def size(self):
        return len(self._category_to_id)


class TextDataSet():
    def __init__(self, filename, vocab, category_vocab, num_timestpes):
        self._vocab = vocab
        self._category_vocab = category_vocab
        self._num_timestpes = num_timestpes
        #matrix
        self._inputs = []
        self._outputs = []
        self._indicator = 0
        self._parse_file(filename)

    def _parse_file(self, filename):
        with open(filename, 'r')  as f:
            lines = f.readlines()
        for line in lines:
            label, content = line.split("\t")
            id_label = self._category_vocab.category_to_id(label)
            id_words = self._vocab.sentence_to_id(content)

            id_words = id_words[0 : self._num_timestpes]
            padding_num = self._num_timestpes - len(id_words)
            id_words = id_words + [self._vocab.unk for i in range(padding_num)]
            self._inputs.append(id_words)
            self._outputs.append(id_label)

        self._inputs = np.array(self._inputs, dtype=np.int32)
        self._outputs = np.array(self._outputs, dtype=np.int32)

        self._random_shuffle()

    def _random_shuffle(self):
        p = np.random.permutation(len(self._inputs))
        self._inputs = self._inputs[p]
        self._outputs = self._outputs[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size

        if end_indicator > len(self._inputs):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        if end_indicator > len(self._inputs):
            raise Exception("batch_size %d is too large" % batch_size)

        batch_inputs = self._inputs[self._indicator: end_indicator]
        batch_outputs = self._outputs[self._indicator: end_indicator]

        self._indicator = end_indicator

        return batch_inputs,batch_outputs

class LSTM(nn.Module):
    def __init__(self, vacab_size, dim, hidden_dim, num_layer, category_num):
        super(LSTM, self).__init__()
        #单词数量
        self.embedding = nn.Embedding(vacab_size, dim)

        self.rnn = nn.LSTM(dim, hidden_dim, num_layer, bidirectional=False)

        self.fc = nn.Linear(hidden_dim, category_num)

        self.dropout = nn.Dropout(0.5)

    def forwar(self, x):
        # [seq, b, 1] 多少单词，几句话，索引 -> seq,b,句子向量
        embedding = self.dropout(self.embedding(x))
        out,




vocab = Vocab(vocab_file, num_word_threshold)
'''
print(vocab.size)
test_str = '我 是 中国心'
print(vocab.sentence_to_id(test_str))
'''
category = CategoryDict(category_file)
#print(category.category_to_id("体育"))

train_dataset = TextDataSet(seg_train_file, vocab, category, num_timesteps)
test_dataset = TextDataSet(seg_test_file, vocab, category, num_timesteps)

batch_inputs,batch_outputs =  train_dataset.next_batch(2)
print(batch_inputs)
print(batch_outputs)

