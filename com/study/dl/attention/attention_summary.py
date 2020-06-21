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


import jieba
import random

def chinese_tokenizer(text):
    return [tok for tok in jieba.lcut(text)]

class MyDataset(data.Dataset):
    def __init__(self, path, text_field, label_field, test=False, aug=False, **kwargs):
        fields = [("id", None), ("comment_text", text_field), ("toxic", label_field)]
        examples = []

        inter_data = {"comment_text":["备胎是硬伤！",
                                 "油耗显示13升还多一点，希望慢慢下降。没有倒车雷达真可恨",
                                 "空调不太凉，应该是小问题。",
                                 "最满意的就是动力了和外观了"],
                      "toxic":[0,0,0,1] }

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


class DotAtten(nn.Module):
    def __init__(self, hidden, bidirection):
        super(DotAtten, self).__init__()
        self.hidden = hidden
        self.dir = 2 if bidirection == True else  1

    def forward(self, lstm_output, last_hidden):
        batch = last_hidden.shape[0]
        #lstm_output:batch,seq,hidden
        #last_hidden: layer*dir,batch,hidden ->batch,layer*dir,hidden ->batch, layer * dir * hidden, 1
        hidden = last_hidden.permute((1, 0, 2)).contiguous().view(batch, -1, 1)
        atten = t.bmm(lstm_output, hidden).squeeze(dim=-1)  #atten:batch,seq,1 -> batch,seq
        alpha = F.softmax(atten, dim=-1).unsqueeze(dim=-1) #alpha:batch,seq,1
        context = t.bmm(lstm_output.permute((0, 2, 1)), alpha).squeeze(-1)  #batch,hidden,1 -> batch,hidden

        atten1 = t.einsum("bij,bjk->bi", [lstm_output, hidden])
        alpha1 = F.softmax(atten1, dim=-1).unsqueeze(dim=-1)  # alpha:batch,seq,1
        context1 = t.einsum("bsh,bsi->bh", [lstm_output, alpha])

        return context1, alpha1

        return context,alpha


class LMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,tgt_size, bidirection,atten):
        super(LMModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.GRU(embed_size, hidden_size, bidirectional = bidirection,batch_first=True)
        self.fc = nn.Linear(hidden_size, tgt_size)
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
    vocab_size, embed_size, hidden_size, tgt_size  = len(TEXT.vocab.itos) + 2, 10, 10, cls
    bidirection = True
    atten = DotAtten(hidden_size, bidirection)
    model = LMModel(vocab_size, embed_size, hidden_size,tgt_size, bidirection, atten)

    LR = 1e-4
    criterion = t.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch, batch in enumerate(train_iter):
        output,alpha = model(batch.comment_text)
        loss = criterion(output, batch.toxic)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

