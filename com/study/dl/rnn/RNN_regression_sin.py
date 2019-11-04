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
让rnn预测sin序列的前一步
'''

TIME_STEP = 10      # rnn time step / image height
INPUT_SIZE = 1      # rnn input size / image width
LR = 0.02           # learning rate
DOWNLOAD_MNIST = False  # set to True if haven't download the data


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        self.lc = nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.shape[1]):
            outs.append(self.lc(r_out[:, time_step, :]))
        return t.stack(outs, dim=1), h_state



if __name__ == '__main__':
    rnn = RNN()
    optimizer = t.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    h_state = None

    for i in range(100):
        #start = np.random.randint(3, size=1)[0]
        start, end = i * np.pi, (i + 1) * np.pi
        steps = np.linspace(start, end, TIME_STEP + 1, dtype=np.float32)
        #steps = np.linspace(start, start+10, TIME_STEP, dtype=np.float32)
        x_np = np.sin(steps)
        steps = steps[:-1]
        x = t.Tensor(t.from_numpy(x_np[:-1][np.newaxis, :, np.newaxis]))
        y = t.Tensor(t.from_numpy(x_np[1:][np.newaxis, :, np.newaxis]))

        predict, h_state = rnn(x, h_state)
        h_state = t.Tensor(h_state)

        loss = loss_func(predict, y)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        plt.plot(steps, x.numpy().flatten(), 'r-')
        plt.plot(steps, predict.data.numpy().flatten(), 'b-')
        plt.draw()
        plt.pause(0.05)
    plt.ioff()
    plt.show()
