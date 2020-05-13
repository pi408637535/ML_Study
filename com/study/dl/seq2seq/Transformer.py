# -*- coding: utf-8 -*-
# @Time    : 2020/5/8 16:07
# @Author  : piguanghua
# @FileName: Transformer.py
# @Software: PyCharm

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


dim=512
dk_dim = 64

class Self_Attention(nn.Module):
    def __init__(self, dim, dk):
        super(Self_Attention, self).__init__()
        self.Wq = nn.Linear(dim, dk)
        self.Wk = nn.Linear(dim, dk)
        self.Wv = nn.Linear(dim, dk)
        self.squar_q = math.sqrt(dk)

    def forward(self, text):
        #batch,seq,embed: batch,seq,512
        q = self.Wq(text)  #batch,seq,64
        k = self.Wk(text) #batch,seq,64
        v = self.Wv(text) #batch,seq,64
        k = k.contiguous().transpose(1,2)
        q_dot_k = t.matmul(q, k) / self.squar_q
        softmax_v = t.matmul(F.softmax(q_dot_k, dim= 2), v)
        return t.sum(softmax_v, dim = 1) #batch,seq


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, dim):
        #dim = 512
        self.net = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )
    def forward(self, input):
        return self.net(input)


class EndocerLayer(nn.Module):
    def __init__(self, vacab, dim, k, head,self_att_mode, ffn):
        self.embed = nn.Embedding(vacab, dim)
        #batch,seq,
        self.head = head
        self.Wz = nn.Linear(dim * head, dim)
        self.self_att_mode = self_att_mode
        self.ffn = ffn

    def forward(self, input):
        #input:batch,seq
        embed = self.embed(input)
        #embed:batch,seq,512
        attentions = []
        for i in range(self.head):
            attentions.append(self.self_att_mode(input))
        multi_atten = t.cat(attentions, dim = 2) #multi_atten:batch,seq,512
        z  = self.Wz(multi_atten)
        atten_output = F.normalize((z + input), dim=2)
        ffn_output = self.ffn(atten_output)

        output = F.normalize(ffn_output + atten_output)
        return output


class Encoder(nn.Module):
    def __init__(self, vocab, dim,  ):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab, dim,)
        self.positional_embed =  None


    def forward(self, texts, enc_self_attn_mask):


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, text):
        pass

