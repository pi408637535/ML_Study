# -*- coding: utf-8 -*-
# @Time    : 2020/7/22 13:07
# @Author  : piguanghua
# @FileName: LayerNorm.py
# @Software: PyCharm

import torch as t
import torch.nn as nn
import numpy as np
import sklearn.datasets
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    a = np.array([[[-0.66676328, -0.95822262, 1.2951657, 0.67924618],
                   [-0.46616455, -0.39398589, 1.95926177, 2.36355916],
                   [-0.39897415, 0.80353481, -1.46488175, 0.55339737]],

                  [[-0.66223895, -0.16435625, -1.96494932, -1.07376919],
                   [1.30338369, -0.19603094, -1.43136723, -1.0207508],
                   [0.8452505, -0.08878595, -0.5211611, 0.10511936]]])

    u = np.mean(a, axis=(2,))
    s = np.std(a, axis=(2,))
    y = a - u[..., None]
    y = y / s[..., None]
    print(y)

    input = t.tensor(a)
    y = F.layer_norm(input, (4,))
    print(y)