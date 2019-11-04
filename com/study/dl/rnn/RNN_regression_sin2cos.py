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

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

def train_data():
    steps = np.linspace(0, np.pi * 2, 100)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    plt.plot(steps, y_np, 'r-')
    plt.plot(steps, x_np, 'b-')
    plt.show()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            INPUT_SIZE,
            32,
            1,
            batch_first=True
        )

        self.out = nn.Linear(32,1)

    def forward(self, x, h_state):
        #h_state 最后一步的hidden_state
        #r_out 每一步的out_put
        # x (batch, time_step, input_size)
        # h_state (n_layer, batch, hidden_size)
        # r_out(batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return t.stack(outs, dim=1),h_state




if __name__ == '__main__':
    rnn = RNN()
    optimizer = t.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    h_state = None
    for step in range(60):
        start, end = step * np.pi, (step+1) * np.pi
        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
        x_np = np.sin(steps)
        y_np = np.cos(steps)

        #shape (batch, time_step, input_size)
        x = t.Tensor(t.from_numpy(x_np[np.newaxis, : ,np.newaxis]))
        y = t.Tensor(t.from_numpy(y_np[np.newaxis, :, np.newaxis]))

        predict,h_state = rnn(x, h_state)
        h_state = t.Tensor(h_state.data)

        loss = loss_func(predict, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # plotting
        plt.plot(steps, y_np.flatten(), 'r-')
        plt.plot(steps, predict.data.numpy().flatten(), 'b-')
        plt.draw();
        plt.pause(0.05)

    plt.ioff()
    plt.show()


