import pandas as pd
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import csv
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import copy,time

'''
    no drop
    no non-liner
    no Batch
    https://github.com/miracleyoo/DPCNN-TextCNN-Pytorch-Inception/blob/master/models/DPCNN.py
'''



class DPCNN(nn.Module):
    def __init__(self,vocab, dim):
        super(DPCNN, self).__init__()
        self.embed = nn.Embedding(vocab, dim)
        out_channels = 250
        in_channels = 1
        class_num = 4

        self.region_embedding = nn.Conv2d(in_channels, out_channels, kernel_size=(3,dim), stride=1)

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1))

        self.pad1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.pad2 = nn.ZeroPad2d((0, 0, 0, 1))

        self.downsampling = nn.MaxPool2d(kernel_size = (3,1), stride = 2)

        self.fc = nn.Linear(out_channels, class_num)

    def forward(self, text):
        text = self.embed(text)

        text = t.unsqueeze(text, dim=1)

        data = self.region_embedding(text) #data: batch,out_channel,seq-3+1,1

        data = self.pad1(data)
        data = self.conv1(data)

        data = self.pad1(data)
        data = self.conv1(data)

        while data.shape[2] > 2:
           data = self._block(data)

        data = data.squeeze()
        return self.fc(data)



    def _block(self, data):
        data = self.pad2(data)

        px = self.downsampling(data) #data:batch,channel,h/2,w

        data = self.pad1(px) #pad要先于conv，如果顺序相反，这样最后一次卷积，就会导致conv核大于数据边框
        data = self.conv1(data)

        data = self.pad1(data)
        data = self.conv1(data)


        return data + px


if __name__ == '__main__':

    vocab, dim = 50, 300
    data = t.randint(vocab, (2, vocab))
    model = DPCNN(vocab, dim)
    result = model(data)
    result