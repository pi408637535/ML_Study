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
import matplotlib.pyplot as plt

'''
1.通过输入sin值去预测cos
'''

# Hyper Parameters
TIME_STEP = 10  # rnn time step
INPUT_SIZE = 1  # rnn input size
LR = 0.02  # learning rate



class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size = input_size,
            hidden_size = hidden_size,
            batch_first = True
        )
        self.out = nn.Linear(hidden_size, output_size)

    '''
    :param x:  batch,seq_len,input_size: 1,10,1
    :param h_state: batch, h_size
    :return:
        因为本函数做作的是的，输入con_y来模拟sin_y，所以每个input的prediction都要
    作为损失的依据。
    '''
    def forward(self, x, h_state):
        #output: batch, seq_len,direction * h_size: 1,10,32
        #h_state:batch,layer*direction, h_size : 1,1,32

        output,h_n = self.rnn(x, h_state)

        out = self.out(t.squeeze(output))
        return out,h_state

class MyL2LSTM(nn.Module):
    def __init__(self, input_size, layer, hidden_size, output_size):
        super(MyL2LSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            batch_first = True,
            num_layers = layer
            #bidirectional=True
        )
        self.out = nn.Linear(hidden_size , output_size)
        self.rnn_type = "LSTM"
        self._hidden_size = hidden_size
        self._layer = layer
    '''
    :param x:  batch,seq_len,input_size: 1,10,1
    :param h_state: batch, h_size
    :return:
        因为本函数做作的是的，输入con_y来模拟sin_y，所以每个input的prediction都要
    作为损失的依据。
    '''
    def forward(self, x, h_state):
        #output: batch, seq_len,direction * h_size: 1,10,32
        #h_state[0]:batch,layer*direction, h_size : 1,1,32

        #output,h_state = self.rnn(x, h_state)
        output, h_state = self.rnn(x, h_state)

        out = self.out(t.squeeze(output))
        return out,h_state

    def init_hidden(self, batch, requires_grad=True):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            hidden = (weight.new(self._layer, batch, self._hidden_size).zero_().to(device),
                      weight.new(self._layer, batch, self._hidden_size).zero_().to(device))
            return hidden
        else:
            return t.zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad)

class MyLBiLSTM(nn.Module):
    def __init__(self, input_size, layer, hidden_size, output_size):
        super(MyLBiLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            batch_first = True,
            num_layers = layer,
            bidirectional=True
        )
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.rnn_type = "LSTM"
        self._hidden_size = hidden_size
        self._layer = layer
    '''
    :param x:  batch,seq_len,input_size: 1,10,1
    :param h_state: batch, h_size
    :return:
        因为本函数做作的是的，输入con_y来模拟sin_y，所以每个input的prediction都要
    作为损失的依据。
    '''
    def forward(self, x, h_state):
        #output: batch, seq_len,direction * h_size: 1,10,32
        #h_state[0]:batch,layer*direction, h_size : 1,1,32

        #output,h_state = self.rnn(x, h_state)
        output, h_state = self.rnn(x, h_state)

        out = self.out(t.squeeze(output))
        return out,h_state

    def init_hidden(self, batch, requires_grad=True):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            hidden = (weight.new(self._layer * 2, batch, self._hidden_size).zero_().to(device),
                      weight.new(self._layer * 2, batch, self._hidden_size).zero_().to(device))
            return hidden
        else:
            return t.zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad)



def repackage_hidden(h):
    if isinstance(h, t.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


if __name__ == '__main__':
    input_size = 1
    hidden_size = 32
    output_size = 1

    rnn = MyRNN(input_size, hidden_size, output_size)
    optimizer = optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    h_state = t.zeros(1,1,32)

    predict_list = []
    y_list = []
    steps_list = []

    for step in range(60):
        start, end = step * np.pi, (step + 1) * np.pi
        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
        x_np = np.sin(steps)
        y_np = np.cos(steps)

        x = t.Tensor(t.from_numpy(x_np[np.newaxis, :, np.newaxis]))
        y = t.Tensor(t.from_numpy(y_np[np.newaxis, :, np.newaxis]))

        #x shape batch,seq_len,dim : 1,10,1
        h_state = repackage_hidden(h_state)
        predict, h_n = rnn(x, h_state)
        #h_state = repackage_hidden(h_state)
        loss = loss_func(predict, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        steps_list.extend(steps.tolist())
        predict_list.extend(predict.data.numpy().flatten())
        y_list.extend(y_np.flatten())

    plt.plot(steps_list, y_list, 'r-')
    plt.plot(steps_list, predict_list, 'b-')
    plt.draw()
    plt.show()
