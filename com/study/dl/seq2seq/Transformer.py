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


#dim=512
dk_dim = 64
dv_dim = 64
dk_dim = 64

class Multi_Attention(nn.Module):
    def __init__(self, dim, di, dim_k, dim_v):
        self.Wq = nn.Linear(dim, dim_q)
        self.Wk = nn.Linear(dim, dim_k)
        self.Wv = nn.Linear(dim, dim_v)

    def forward(self, texts):
        #batch,
        q = self.Wq(texts)

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

