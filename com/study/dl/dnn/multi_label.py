# -*- coding: utf-8 -*-
# @Time    : 2020/4/7 11:11
# @Author  : piguanghua
# @FileName: mutilizer_label.py
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
from sklearn.datasets import load_iris



class MyDataset(Dataset):

    def __init__(self, vocab=None, opt=None):
        self.iris = load_iris()
        self.data = self._load_dataset()

    def _load_dataset(self):

        all_data = []
        for i in range(len(self.iris.data)):
            data = { "data": self.iris.data[1,:],  "label":self.iris.target[i]}
            all_data.append(data)
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class LinerModel(nn.Module):

    def __init__(self, input, hidden,output):
        super(LinerModel, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output)
        )

    def forward(self, data):
        data = t.FloatTensor(data.numpy())
        data = t.squeeze(data)
        out = self.sequential(data)
        return out


def train(model, dataloader, criterion, device):
    model.train()
    for batch_id, batch_data in enumerate(dataloader, 0):
        targets = batch_data['label'].to(device)
        inputs = batch_data['data'].to(device)

        output = model(inputs)
        loss = criterion(output, targets)
        print(loss.item()  / inputs.shape[0])

def eval(model, dataloader):
    model.eval()
    for batch_id, batch_data in enumerate(dataloader, 0):
        inputs = batch_data['data'].to(device)

        output = model(inputs)
        output = t.squeeze(output)
        outputs = t.max(output.data, 1)[1]


if __name__ == '__main__':
    input = 4
    output = 3
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