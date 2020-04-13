# -*- coding: utf-8 -*-
# @Time    : 2020/4/10 11:08
# @Author  : piguanghua
# @FileName: SkipGramModel.py
# @Software: PyCharm

from matplotlib import pyplot as plt
import numpy as np
import random
import torch as t
import torch.nn as nn
import numpy as np
import sklearn.datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import codecs
from collections import Counter
from collections import OrderedDict
import copy

#hypotrical parameter
USE_CUDA = t.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
t.manual_seed(53113)
if USE_CUDA:
    t.cuda.manual_seed(53113)

# 设定一些超参数

K = 100  # number of negative samples
C = 3  # nearby words threshold
NUM_EPOCHS = 2  # The number of epochs of training
MAX_VOCAB_SIZE = 40000  # the vocabulary size
BATCH_SIZE = 128  # the batch size
LEARNING_RATE = 0.2  # the initial learning rate
EMBEDDING_SIZE = 300

text = None  #文本
def preproccess():
    def word_tokenize(text):
        return text.split()

    with codecs.open("/home/demo1/womin/piguanghua/data/pre_data/text8.train.txt", "r") as fin:
        text = fin.read()

    tokens = [w for w in word_tokenize(text.lower())]
    words = dict(Counter(tokens).most_common(MAX_VOCAB_SIZE - 1))
    #words = copy.deepcopy(tokens)
    words["<unk>"] = len(tokens) - np.sum(list(words.values()))

    # 决定负采样的频率
    # ?有问题
    totoal_words_times = np.sum(list(words.values()))
    print(words["the"])

    print(totoal_words_times)
    totoal_words_times = 1e7

    # frequents = [word  for word in words]
    frequents = {k: (v / totoal_words_times) for k, v in words.items()}
    print(frequents["the"])

    frequents = {k: (v ** (3. / 4.)) for k, v in frequents.items()}
    totoal_words_times = np.sum(list(frequents.values()))
    # 据说有优化 ?有问题
    frequents = {k: (v / totoal_words_times) for k, v in frequents.items()}
    words = sorted(words, key=words.get, reverse=True)
    word2idx = {o: i for i, o in enumerate(words)}
    idx2word = {i: o for i, o in enumerate(words)}

    return word2idx, idx2word, frequents, tokens

class MyDataset(Dataset):

    def __init__(self, word2idx, idx2word, frequents, tokens):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.frequents = frequents
        self.tokens = tokens
        #self.word_encoder = [word2idx[token] for token in tokens]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        center_word = self.tokens[index]
        center_label = self.word2idx.get(center_word, word2idx["<unk>"])

        pos_indexs = list(range(index - C, index)) + list(range(index + 1, index + C + 1))
        pos_indexs = [i % len(self.tokens) for i in pos_indexs]
        pos_labels = [ self.word2idx.get(self.tokens[pos_index], word2idx["<unk>"]) for pos_index in pos_indexs]

        negative_word = t.multinomial(t.from_numpy(np.array(list(self.frequents.values()))),
                                      K * 2 * C)

        return t.Tensor(center_label), t.Tensor(pos_labels), negative_word

class Skip_Gram(nn.Module):
    def __init__(self, vocab_size,embed_size):
        super(Skip_Gram, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)


    def forward(self, input_labels, pos_labels, neg_labels):
        '''
        :param input_labels: input_labels: 中心词, [batch_size]
        :param pos_labels: pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        :param neg_labels: neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]
        :return:
        '''
        batch_size = input_labels.size(0)
        input_embedding = self.in_embed(input_labels)  # B * embed_size

        pos_embedding = self.out_embed(pos_labels)  # B * (2*C) * embed_size
        neg_embedding = self.out_embed(neg_labels)  # B * (2*C * K) * embed_size

        log_pos = t.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()  # B * (2*C)
        log_neg = t.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()  # B * (2*C*K)

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)  # batch_size

        loss = log_pos + log_neg
        return loss


def train(model, dataloader, optimizer):
    for e in range(NUM_EPOCHS):
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):

            # TODO
            input_labels = input_labels.long()
            pos_labels = pos_labels.long()
            neg_labels = neg_labels.long()
            if USE_CUDA:
                input_labels = input_labels.cuda()
                pos_labels = pos_labels.cuda()
                neg_labels = neg_labels.cuda()

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                    print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.item()))



        embedding_weights = model.input_embeddings()
        np.save("embedding-{}".format(EMBEDDING_SIZE), embedding_weights)
        #torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))

if __name__ == '__main__':
    word2idx, idx2word, frequents, tokens = preproccess()
    model = Skip_Gram(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
    batch_size = 1
    lr = 1e-4
    optimizer = t.optim.SGD(model.parameters(), lr=lr)

    dataloader = DataLoader(dataset=MyDataset(word2idx, idx2word, frequents, tokens), batch_size=batch_size)
    train(model, dataloader, optimizer)
