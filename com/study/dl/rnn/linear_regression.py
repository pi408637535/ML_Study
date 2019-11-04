import torch
import numpy as np
import torch as t
from torch.autograd import Variable
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython import display


device = t.device('cpu')

def get_fake_data(batch_size = 8):
    x = t.rand((batch_size, 1) ,device=device) * 5
    y = x * 2 + 3 + t.randn((batch_size, 1), device=device)
    return x,y

def get_init_parameter():
    w = t.rand(1,1).to(device)
    b = t.zeros(1,1).to(device)
    lr = 0.02
    return w,b,lr

if __name__ == '__main__':

    w,b,lr = get_init_parameter()

    for ii in range(500):
        x, y = get_fake_data()
        y_pred = x.mm(w) + b.expand_as(y)
        loss = 0.5 * (y_pred - y) ** 2
        loss = loss.mean()

        #backward
        dloss = 1
        dy_pred = dloss * (y_pred - y)
        #dw = x.t().mm(dy_pred)
        dw = x.t().mm(dy_pred)
        db = dy_pred.sum()

        #delta
        w.sub_(lr * dw)
        b.sub_(lr * dw)

        if ii % 50 == 0:
            # 画图
            display.clear_output(wait=True)
            x = t.arange(0, 6).view(-1, 1)
          #  y = x.mm(w) + b.expand_as(x)
            y = x.numpy() * w.numpy() + b.numpy()
            plt.plot(x.numpy(), y)  # predicted

            x2, y2 = get_fake_data(batch_size=32)
            plt.scatter(x2.numpy(), y2.numpy())  # true data

            plt.xlim(0, 5)
            plt.ylim(0, 13)
            plt.show()
            plt.pause(0.5)


    print(w,b)