# -*- coding: utf-8 -*-
# @Time    : 2020/4/16 15:59
# @Author  : piguanghua
# @FileName: NNLModel.py
# @Software: PyCharm

#this model train word embedding NNLM
#Impletation refer to <A Neural Probabilistic Language Model> pdf:http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

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
import pandas as pd

#hypotrical parameter
USE_CUDA = t.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
t.manual_seed(53113)
if USE_CUDA:
    t.cuda.manual_seed(53113)

NUM_EPOCHS = 200
BATCH_SIZE = 1  # the batch size
LEARNING_RATE = 0.2  # the initial learning rate
EMBEDDING_SIZE = 300
N_GRAM = 2
HIDDEN_UNIT = 128
UNK = "<unk>"

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

tokens = test_sentence
trigram = [ ((test_sentence[i],test_sentence[i+1]),test_sentence[i+2]) for i in  range(len(tokens)-N_GRAM) ]
words = dict(Counter(tokens).most_common())
def cmp(a,b):
    return (a > b) - (a < b)
words = sorted(iter(words.keys()), key=words.get, reverse=False)
words += UNK
word2id = { k:i  for i,k in enumerate(words) }
id2word = { i:k  for i,k in enumerate(words) }

H =  N_GRAM * EMBEDDING_SIZE
U =  HIDDEN_UNIT


class MyDataset(Dataset):

    def __init__(self, word2id, id2word, tokens):
        self.word2id = word2id
        self.id2word = id2word
        self.tokens = tokens
        #self.word_encoder = [word2idx[token] for token in tokens]

    def __len__(self):
        return len(self.tokens) - N_GRAM

    def __getitem__(self, index):
        ((word_0, word_1), word_2) = trigram[index]
        word_0 = self.word2id[word_0]
        word_1 = self.word2id[word_1]
        word_2 = self.word2id[word_2]

        return word_0, word_1, word_2

class NNLM(nn.Module):
    def __init__(self,vocab, dim):
        super(NNLM, self).__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.H = nn.Parameter(t.randn(EMBEDDING_SIZE * N_GRAM, HIDDEN_UNIT))
        self.d = nn.Parameter(t.randn(HIDDEN_UNIT))
        self.U = nn.Parameter(t.randn(HIDDEN_UNIT, vocab))
        self.b = nn.Parameter(t.randn(vocab))
        self.W = nn.Parameter(t.randn(EMBEDDING_SIZE * N_GRAM, vocab))
    '''
        words:batch,sequence
        # x: [batch_size, n_step*n_class]
    '''
    def forward(self, word_0, word_1):
        batch = word_0.shape[0]
        word_0 = self.embed(word_0)
        word_1 = self.embed(word_1)
        words = t.cat((word_0, word_1), dim=1)
        words = words.view(batch, -1) #batch,sequence*dim
        tanh = t.tanh(t.mm(words,self.H) + self.d) #tanh:batch,HIDDEN_UNIT
        hidden_output = t.mm(tanh, self.U) + self.b  #hidden_output：batch,vocab
        y = hidden_output + t.mm(words, self.W)
        y = F.log_softmax(y,1)

        return -y


def evaluate(model, word_0, word_1):
    model.eval()
    word_0 = word_0.long()
    word_1 = word_1.long()

    softmax = model(word_0, word_1)
    predict = t.argmax(softmax,1)

    word_0 = word_0.cpu().detach().numpy()
    word_1 = word_1.cpu().detach().numpy()
    predict = predict.cpu().detach().numpy()
    word_sequence = [ (( id2word[word_0[i]], id2word[word_1[i]]),id2word[predict[i]]) for i in range(len(word_0)) ]
    print(word_sequence)
    model.train()

def train(model, dataloader, optimizer, criterion):
    model.train()
    for e in range(NUM_EPOCHS):
        for i, (word_0, word_1, word_2) in enumerate(dataloader):

            word_0 = word_0.long()
            word_1 = word_1.long()
            word_2 = word_2.long()
            if USE_CUDA:
                word_0 = word_0.cuda()
                word_1 = word_1.cuda()
                word_2 = word_2.cuda()

            optimizer.zero_grad()

            softmax = model(word_0, word_1)
            loss = criterion(softmax, word_2)
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                    print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.item()))
                    evaluate(model, word_0, word_1)


        #embedding_weights = model.input_embeddings()
        #np.save("embedding-{}".format(EMBEDDING_SIZE), embedding_weights)
        t.save(model.state_dict(), "embedding-{}.pth".format(EMBEDDING_SIZE))

if __name__ == '__main__':
    word2idx, idx2word, = word2id,id2word
    dim = EMBEDDING_SIZE
    hidden = HIDDEN_UNIT

    model = NNLM(len(word2id.keys()), dim)

    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    model.to(t.device("cuda" if USE_CUDA else 'cpu'))

    lr = 1e-4
    optimizer = t.optim.SGD(model.parameters(), lr=lr)

    dataloader = DataLoader(dataset=MyDataset(word2id, id2word, tokens), batch_size=BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    train(model, dataloader, optimizer, criterion)






