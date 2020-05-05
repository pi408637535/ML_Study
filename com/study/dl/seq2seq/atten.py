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
#https://github.com/ShusenTang/Dive-into-DL-PyTorch/blob/master/code/chapter10_natural-language-processing/10.12_machine-translation.ipynb

def attention_model(input_size, attention_size):
    model = nn.Sequential(nn.Linear(input_size, attention_size, bias=False),
                          nn.Tanh(),
                          nn.Linear(attention_size, 1, bias=False))
    return model

def attention_forward(model, enc_states, dec_state):
    """
    enc_states: (batch, seq, hidden)
    dec_state: (batch, hidden)
    """
    # 将解码器隐藏状态广播到和编码器隐藏状态形状相同后进行连结
    batch, seq, hidden = enc_states.shape
    dec_states = dec_state[:,None,:].repeat(1,seq,1)

    enc_and_dec_states = t.cat((enc_states, dec_states), dim=2)
    e = model(enc_and_dec_states)  # (batch, seq, 1)
    alpha = F.softmax(t.squeeze(e, dim = 2), dim=1)  # 在时间步维度做softmax运算
    alpha = t.unsqueeze(alpha, dim=2)
    return (alpha * enc_states).sum(dim=1)  # 返回背景变量

if __name__ == '__main__':
    seq_len, batch_size, num_hiddens = 10, 4, 8
    model = attention_model(2 * num_hiddens, 10)
    enc_states = t.zeros((batch_size, seq_len, num_hiddens))
    dec_state = t.zeros((batch_size, num_hiddens))
    print(attention_forward(model, enc_states, dec_state).shape)