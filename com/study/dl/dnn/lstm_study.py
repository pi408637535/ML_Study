import torch as t
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import torchtext
import random

USE_CUDA = t.cuda.is_available()

if __name__ == '__main__':
    #保证实验结果可复现，需要把各种random seed固定在某只
    random.seed(0)
    np.random.seed(0)
    t.manual_seed(0)
    if USE_CUDA:
        t.cuda.random.seed(0)

    BATCH_SIZE = 32
    EMBEDDING_SIZE = 100
    MAX_VOCAB_SIZE = 50000

    TEXT = torchtext.data.Field(lower=True)
    path = "/Users/piguanghua/Downloads/text8"
    train,val,test = torchtext.datasets.LanguageModelingDataset.splits(
        path=path,
        train="text8.train.txt",
        validation="text8.dev.txt",
        test="text8.test.txt",
        text_field = TEXT
    )
    #构造 term-num 结构
    TEXT.build_vocab(train, max_size = MAX_VOCAB_SIZE)
    print(len(TEXT.vocab))
    print(TEXT.vocab.itos[:10])
    print(TEXT.vocab.stoi["and"])

    device = t.device("cuda" if USE_CUDA else "cpu" )

    '''
        bptt_len bp回传长度
        repeat单文档不会重复
    '''
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator\
        .splits( (train, val, test), batch_size=BATCH_SIZE,
                 device=device, bptt_len = 50, repeat = False,
                 shuffle = True)
