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




if __name__ == '__main__':
    dim = 300
    output =
    hidden = 8

    model = LinerModel(input, hidden, output)
    criterion = nn.CrossEntropyLoss()
    lr = 1e-4
    optimizer = t.optim.SGD(model.parameters(), lr=lr)
    epochs = 15

    use_gpu = True if t.cuda.is_available() else False
    device = t.device('cuda' if use_gpu else 'cpu')
    batch_size = 50
    dataloader = DataLoader(dataset=MyDataset(), batch_size=batch_size)

    for epoch in range(epochs):
        train(model, dataloader, criterion, device)
        eval(model, dataloader)
