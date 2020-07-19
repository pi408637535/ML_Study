import pandas as pd
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

from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm
import pandas as pd
from torchtext.data import Iterator, BucketIterator
import matplotlib.pyplot as plt

import jieba
import random

def chinese_tokenizer(text):
    return [tok for tok in jieba.lcut(text)]

class MyDataset(data.Dataset):
    def __init__(self, path, text_field, label_field, test=False, aug=False, **kwargs):
        fields = [("id", None), ("comment_text", text_field), ("toxic", label_field)]
        examples = []

        '''
        inter_data = {"comment_text":["备胎是硬伤！",
                                 "油耗显示13升还多一点，希望慢慢下降。没有倒车雷达真可恨",
                                 "空调不太凉，应该是小问题。",
                                 "最满意的就是动力了和外观了"],
                      "toxic":[0,0,0,1] }
        '''

        inter_data = {"comment_text": [
                                       "油耗显示13升还多一点，希望慢慢下降。没有倒车雷达真可恨",
                                        ],
                      "toxic": [0]}


        csv_data = pd.DataFrame(inter_data)
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
                examples.append(data.Example.fromlist([None, text, label], fields))
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

class BilinearAtten(nn.Module):
    def __init__(self, hidden, bidirection):
        super(BilinearAtten, self).__init__()
        self.hidden = hidden
        self.dir = 2 if bidirection == True else  1
        self.W = nn.Parameter( t.randn( (self.dir * hidden, self.dir * self.hidden )  ))

    def forward(self, lstm_output, last_hidden):
        batch = last_hidden.shape[0]
        hidden = last_hidden.permute((1, 0, 2)).contiguous().view(batch, -1, 1) #batch, -1 ,1

        #b:batch,s:seq,h:hidden,
        atten1 = t.einsum("bsh, hh, bhk->bs", [lstm_output, self.W, hidden]) #batch,seq,hidden @ hidden，hidden,
        alpha1 = F.softmax(atten1, dim=-1).unsqueeze(dim=-1)  # alpha:batch,seq,1
        context1 = t.einsum("bsh,bsi->bh", [lstm_output, alpha1])
        return context1, alpha1

class CascadeAtten(nn.Module):
    def __init__(self, hidden, bidirection):
        super(CascadeAtten, self).__init__()
        self.hidden = hidden
        self.dir = 2 if bidirection == True else  1
        self.lstm_w = nn.Parameter( t.randn( (self.dir * hidden, 1 )  ))
        self.hidden_w = nn.Parameter(t.randn((self.dir * hidden, 1)))

    def forward(self, lstm_output, last_hidden):
        batch = last_hidden.shape[0]
        hidden = last_hidden.permute((1, 0, 2)).contiguous().view(batch, -1, 1)  # batch, -1 ,1

        lstm_hidden = t.einsum("bsh,ht->bs",[lstm_output, self.lstm_w]) #batch,seq,hidden
        hidden = t.einsum("ht,bht->b", [self.hidden_w, hidden])  # batch,hidden

        atten = t.einsum("bs,b->bs",[lstm_hidden,hidden ] ) #Cascade

        alpha1 = F.softmax(atten, dim=-1).unsqueeze(dim=-1)  # alpha:batch,seq,1
        context1 = t.einsum("bsh,bsi->bh", [lstm_output, alpha1])
        return context1, alpha1

class MLPAtten(nn.Module):
    def __init__(self, hidden, bidirection,seq):
        super(MLPAtten, self).__init__()
        self.hidden = hidden
        self.dir = 2 if bidirection == True else  1
        self.lstm_w = nn.Parameter( t.randn( (self.dir * hidden, 1 )  ))
        self.hidden_w = nn.Parameter(t.randn((self.dir * hidden, 1)) )
        self.all_w = nn.Linear(seq, self.dir * self.hidden)

    def forward(self, lstm_output, last_hidden):
        batch = last_hidden.shape[0]
        hidden = last_hidden.permute((1, 0, 2)).contiguous().view(batch, -1, 1)  # batch, -1 ,1

        lstm_hidden = t.einsum("bsh,ht->bs",[lstm_output, self.lstm_w]) #batch,seq,hidden
        hidden = t.einsum("ht,bht->b", [self.hidden_w, hidden])  # batch,hidden

        atten =  self.all_w(  F.tanh( t.einsum("bs,b->bs",[lstm_hidden,hidden ] )) ) #batch,seq,1

        alpha1 = F.softmax(atten, dim=-1).unsqueeze(dim=-1)  # alpha:batch,seq,1
        context1 = t.einsum("bsh,bsi->bh", [lstm_output, alpha1])
        return context1, alpha1


class DotAtten(nn.Module):
    def __init__(self, hidden, bidirection):
        super(DotAtten, self).__init__()
        self.hidden = hidden
        self.dir = 2 if bidirection == True else  1

    def forward(self, lstm_output, last_hidden):
        batch = lstm_output.shape[0]
        #lstm_output:batch,seq,hidden
        #last_hidden: layer*dir,batch,hidden ->batch,layer*dir,hidden ->batch, layer * dir * hidden, 1
        hidden = last_hidden.permute((1, 0, 2)).contiguous().view(batch, -1, 1)
        atten = t.bmm(lstm_output, hidden).squeeze(dim=-1)  #atten:batch,seq,1 -> batch,seq
        alpha = F.softmax(atten, dim=-1).unsqueeze(dim=-1) #alpha:batch,seq,1
        context = t.bmm(lstm_output.permute((0, 2, 1)), alpha).squeeze(-1)  #batch,hidden,1 -> batch,hidden

        atten1 = t.einsum("bij,bjk->bi",[lstm_output, hidden])
        alpha1 = F.softmax(atten1, dim=-1).unsqueeze(dim=-1)#alpha:batch,seq,1
        context1 = t.einsum("bsh,bsi->bh",[lstm_output,alpha])


        return context1,alpha1


class LMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,tgt_size, bidirection,atten):
        super(LMModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.GRU(embed_size, hidden_size, bidirectional = bidirection,batch_first=True)
        bidir = 2 if bidirection == True else 1
        self.fc = nn.Linear(hidden_size * bidir , tgt_size)
        self.atten = atten

    def forward(self, text):
        #text:batch,seq
        embed = self.embed(text)
        output,h_n = self.lstm(embed)
        context,alpha = self.atten(output,h_n)
        output = self.fc(context)
        return output,alpha

if __name__ == '__main__':
    TEXT = data.Field(sequential=True, tokenize=chinese_tokenizer, batch_first=True ,fix_length=20)
    LABEL = data.Field(sequential=False, use_vocab=False)

    train = MyDataset("", text_field=TEXT, label_field=LABEL, test=False, aug=0)

    TEXT.build_vocab(train)
    # 统计词频
    TEXT.vocab.freqs.most_common(1)

    # 同时对训练集和验证集进行迭代器的构建
    train_iter,_ = BucketIterator.splits(
        (train,train),  # 构建数据集所需的数据集
        batch_sizes=(2,2),
        device=-1,  # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: len(x.comment_text),
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=False,
        # we pass repeat=False because we want to wrap this Iterator layer.
    )


    cls = 2
    seq = 20
    vocab_size, embed_size, hidden_size, tgt_size  = len(TEXT.vocab.itos) + 2, 10, 10, cls
    bidirection = True

    atten_dict = { "dot":DotAtten(hidden_size, bidirection),
                   "biliner":BilinearAtten(hidden_size, bidirection),
                   "cascade": CascadeAtten(hidden_size, bidirection),
                   "mlp": MLPAtten(hidden_size, bidirection, seq)}

    atten = atten_dict["dot"]
    model = LMModel(vocab_size, embed_size, hidden_size,tgt_size, bidirection, atten)

    LR = 1e-4
    criterion = t.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    total_list = []
    total = 100

    for tt in range(total):

        for epoch, batch in enumerate(train_iter):
            output,alpha = model(batch.comment_text)
            loss = criterion(output, batch.toxic)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            alpha = alpha.squeeze(-1).squeeze(0) #batch,seq
            total_list.append(alpha.detach().numpy())

    data = np.array(total_list)

    fig = plt.figure(figsize=(seq, total))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.matshow(data, cmap='viridis')
    x_tick_label = ["w_{0}".format(i) for i in range(total)]

    ax.set_xticklabels(x_tick_label, fontdict={'fontsize': 14}, rotation=90)
    plt.colorbar(im)
    plt.show()


