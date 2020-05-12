import os
import sys
import math
from collections import Counter
import numpy as np
import random

import torch.nn as nn
import torch.nn.functional as F

import torch as t
import torch
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
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


EMBEDDING_SIZE = 300

class SelfAttentio(nn.Module):
    def __init__(self, dim, q):
        super(SelfAttentio, self).__init__()
        self.W_Q = nn.Linear(dim, q)
        self.W_K = nn.Linear(dim, q)
        self.W_V = nn.Linear(dim, q)
        self.d_k = math.sqrt(q)

    def forward(self, input):
        #input:batch,seq,512
        q = self.W_Q(input)
        k = self.W_K(input)
        v = self.W_V(input)
        
        #q,k,v batch,seq,64
        scores = t.matmul(q, k.contiguous().transpose_(1,2)), #scores:batch,seq,seq
        scores = scores / self.d_k
        scores = F.softmax(scores, dim= 2)
        z = t.matmul(scores, v) #z:batch,seq,64
        return z

class PositionFeedForward(nn.Module):
    def __init__(self, dim):
        #dim = 512
        super(PositionFeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

    def forward(self, input):
        #input batch,seq,512
        return self.ffn(input)




class EncoderLayer(nn.Module):
    def __init__(self, head, dim = 512, q = 64):
        super(EncoderLayer, self).__init__()
        self.multi_attention = nn.ModuleList([SelfAttentio(dim = dim, q = q) for i in range(head)])

    def

class Encoder(nn.Module):
    def __init__(self, head):
        super(Encoder, self).__init__()
        self.net = nn.ModuleList([EncoderLayer for i in range(head)])


    def forward(self, input):
        #input:batch,seq,512
        pass


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, vocab, embed):
        super(Transformer, self).__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, texts):
        #text:batch,seq
        embed = self.embed(texts)
        self.encoder(embed)
        pass
