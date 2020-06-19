# -*- coding: utf-8 -*-
# @Time    : 2020/6/19 09:48
# @Author  : piguanghua
# @FileName: torchtext_study].py
# @Software: PyCharm

#步骤
#1.需要的Field,Label 2.DateSet 3.Iterable 4.词典 5.enumerate

#Example
#https://blog.nowcoder.net/n/3a8d2c1b05354f3b942edfd4966bb0c1
#http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/

import spacy
spacy_en = spacy.load('en')

from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm
import pandas as pd
from torchtext.data import Iterator, BucketIterator

def tokenizer(text): # create a tokenizer function
    # 返回 a list of <class 'spacy.tokens.token.Token'>
    return [tok.text for tok in spacy_en.tokenizer(text)]


import jieba


def chinese_tokenizer(text):
    return [tok for tok in jieba.lcut(text)]

 # get_dataset构造并返回Dataset所需的examples和fields
def get_dataset(csv_data, text_field, label_field, test=False):
        # id数据对训练在训练过程中没用，使用None指定其对应的field
        fields = [("id", None),  # we won't be needing the id, so we pass in None as the field
                  ("comment_text", text_field), ("toxic", label_field)]
        examples = []

        if test:
            # 如果为测试集，则不加载label
            for text in tqdm(csv_data['comment_text']):
                examples.append(data.Example.fromlist([None, text, None], fields))
        else:
            for text, label in tqdm(zip(csv_data['comment_text'], csv_data['toxic'])):
                examples.append(data.Example.fromlist([None, text, label], fields))
        return examples, fields

from torchtext import data
from torchtext.vocab import Vectors
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import random
import os


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
    #构建Field对象
    tokenize = lambda x: x.split()
    # fix_length指定了每条文本的长度，截断补长
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=20)
    LABEL = data.Field(sequential=False, use_vocab=False)

    #train_data = pd.read_csv('/Users/piguanghua/Downloads/torch_data/train_one_label.csv')
    #valid_data = pd.read_csv('/Users/piguanghua/Downloads/torch_data/valid_one_label.csv')
    #test_data = pd.read_csv("/Users/piguanghua/Downloads/torch_data/test.csv")

    train_data = '/Users/piguanghua/Downloads/torch_data/train_one_label.csv'
    valid_data = '/Users/piguanghua/Downloads/torch_data/valid_one_label.csv'
    test_data = "/Users/piguanghua/Downloads/torch_data/test.csv"

    train = MyDataset(train_data, text_field=TEXT, label_field=LABEL, test=False, aug=0)
    valid = MyDataset(valid_data, text_field=TEXT, label_field=LABEL, test=False, aug=0)
    test = MyDataset(test_data, text_field=TEXT, label_field=LABEL, test=True, aug=0)


    # 构建Dataset数据集
    #train = data.Dataset(train_examples, train_fields)
    #valid = data.Dataset(valid_examples, valid_fields)
    #test = data.Dataset(test_examples, test_fields)

    TEXT.build_vocab(train)
    # 统计词频
    TEXT.vocab.freqs.most_common(10)

    # 同时对训练集和验证集进行迭代器的构建
    train_iter, val_iter = BucketIterator.splits(
        (train, valid),  # 构建数据集所需的数据集
        batch_sizes=(8, 8),
        device=-1,  # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: len(x.comment_text),
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=False,
        repeat=False  # we pass repeat=False because we want to wrap this Iterator layer.
    )

    test_iter = Iterator(test, batch_size=8, device="cpu", sort=False, sort_within_batch=False, repeat=False)

    for epoch, batch in enumerate(train_iter):
        print(batch.comment_text)


