import torch as t
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    embedding = nn.Embedding(3,2)
    print(embedding)
    print(embedding.weight)
    input = t.arange(0,3).view(-1,1).long()
    print(input.shape)
    #input = t.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    a = embedding(input)  # 输出2*4*3
    print(a)
