# -*- coding: utf-8 -*-
# @Time    : 2020/7/22 13:07
# @Author  : piguanghua
# @FileName: batch_norm.py
# @Software: PyCharm


import torch as t
import torch.nn as nn
import numpy as np
import sklearn.datasets
import torch.nn as nn
import torch.nn.functional as F

class BN(nn.Module):
    def __init__(self,  num_features, momentum, eps=0.001):
        super(BN, self).__init__()
        self._running_mean = 0
        self._running_var = 1

        self._momentum = momentum
        self._eps = eps

        self._beta = t.zeros((num_features,))
        self._gamma = t.ones((num_features,))

    def forward(self, x):
        x_mean = x.mean(axis=0)
        x_var = x.var(axis=0)
        # 对应running_mean的更新公式
        self._running_mean = (1 - self._momentum) * x_mean + self._momentum * self._running_mean
        self._running_var = (1 - self._momentum) * x_var + self._momentum * self._running_var
        # 对应论文中计算BN的公式
        x_hat = (x - x_mean) / np.sqrt(x_var + self._eps)
        y = self._gamma * x_hat + self._beta
        return y
