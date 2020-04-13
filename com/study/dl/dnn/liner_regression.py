# -*- coding: utf-8 -*-
# @Time    : 2020/4/7 09:13
# @Author  : piguanghua
# @FileName: liner_regression.py
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

'''
1.Pytroch Liner仅能接受Float
2.Liner Input n,*,input Ouput:N,*,output
'''

class MyDataset(Dataset):

    def __init__(self, vocab=None, opt=None):
        self.data = self._load_dataset()

    def _load_dataset(self):

        all_data = []
        num_inputs = 2
        num_examples = 1000
        true_w = [2, -3.4]
        true_b = 4.2
        features = np.random.normal(size=(num_examples, num_inputs))
        labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
        labels += np.random.normal(scale=0.01, size=labels.shape)

        for i in range(num_examples):
            data = { "data": np.reshape(features[i,:],(-1,1)).astype(np.float),  "label":labels[i]}
            all_data.append(data)
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class LinerModel(nn.Module):

    def __init__(self, input, output):
        super(LinerModel, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input, output),
        )

    def forward(self, data):
        data = t.FloatTensor(data.numpy())
        data = t.squeeze(data)
        out = self.sequential(data)
        return out

if __name__ == '__main__':
    '''
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = np.random.normal(size=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += np.random.normal(scale=0.01, size=labels.shape)

    plt.scatter(features[:, 0], labels)
    plt.scatter(features[:, 1], labels)

    plt.show()
    '''

    input = 2
    output = 1
    model = LinerModel(input, output)
    criterion = nn.MSELoss()
    lr = 1e-4
    optimizer = t.optim.SGD(model.parameters(),lr = lr)
    epochs = 5

    use_gpu = True if t.cuda.is_available() else False
    device = t.device('cuda' if use_gpu else 'cpu')
    batch_size = 50
    dataloader = DataLoader(dataset=MyDataset(), batch_size=batch_size)

    for epoch in range(epochs):
        for batch_id, batch_data in enumerate(dataloader, 0):
            model.train()
            targets = batch_data['label'].to(device)
            inputs = batch_data['data'].to(device)

            output = model(inputs)
            output = t.squeeze(output)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            print(loss.item())

    for name,parameters in model.named_parameters():
        print(name,':',parameters)




