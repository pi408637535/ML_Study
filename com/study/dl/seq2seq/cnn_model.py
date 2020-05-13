# -*- coding: utf-8 -*-
# @Time    : 2020/5/12 18:01
# @Author  : piguanghua
# @FileName: cnn_model.py
# @Software: PyCharm

import pandas as pd
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

class LetNet(nn.Module):
    def __init__(self):
        super(LetNet, self).__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.s2 = nn.MaxPool2d(6, 6, kernel_size=14)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.MaxPool2d(16, kernel_size=6)
        self.c5_1 = nn.Linear(120, 120)
        self.c5 = nn.Linear(120, 84)
        self.f6 = nn.Linear(84, 10)


    def forward(self, picture):
        #picture:batch,channel,w,h:batch,1,W,H

        c1 = self.c1(picture)
        s2 = self.s2(c1)
        c3 = self.c3(s2)
        s4 = self.s4(c3)
        c3 = s4.view()
        c5 = self.c5(s4)
        f6 = self.f6(c5)
        pass
