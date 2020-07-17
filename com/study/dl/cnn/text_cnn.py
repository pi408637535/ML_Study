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

class TextCNNMode(nn.Module):
    def __init__(self, vocab, dim):
        super(TextCNNMode, self).__init__()
        self.embed = nn.Embedding(vocab, dim)
        ngram = [2,3,4]
        in_channels = 1
        out_channels = 2
        class_num = 4
        self.filters = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=(ele, dim)) for ele in ngram ])
        self.fc = nn.Linear( out_channels * len(ngram), class_num)

    @staticmethod
    def conv_filter(conv, text):
        #text: batch,in_channel,h,w: b,in,seq,dim
        data = conv(text) #data: batch,out_channel,H,1
        data = t.squeeze(data, dim = -1) #data: batch,out_channel,H
        data = F.max_pool1d(data, kernel_size=(data.shape[-1])) #data: batch, channel, 1\
        data = t.squeeze(data, dim=-1)
        return data

    def forward(self, text):
        text = self.embed(text)
        text = t.unsqueeze(text, dim = 1)
        data = [self.conv_filter(cnn_filter, text) for cnn_filter in self.filters ]
        data = t.cat(data, dim = 1)
        return self.fc(data)





if __name__ == '__main__':
    vocab, dim = 7,5
    data = t.randint(vocab, (1,7))
    model = TextCNNMode(vocab, dim)
    result = model(data)



