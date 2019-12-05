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

#define
TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    '''
        x: batch,seq_len,input_size:1,10,1
        h_state:batch,layer*direction, h_size
    '''
    def forward(self, x, h_state):
        #output: batch,seq_len,direction*hidden_size: 1,10,1*32
        #h_state:batch,layer*direction, h_size: 1,1,32
        output, h_n =  self.rnn(x, h_state)
        out = self.out(t.squeeze(output))
        return out, h_n

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
    h_state = t.zeros(1, 1, 32)

    predict_list = []
    y_list = []
    steps_list = []

    for step in range(60):
        start, end = step * np.pi, (step + 1) * np.pi
        steps = np.linspace(start+1, end+1, TIME_STEP, dtype=np.float32)
        x_np = np.sin(steps)
        y_np = np.cos(np.linspace(start, end, TIME_STEP, dtype=np.float32))

        #x : batch,seq_len,input_size: 1,10,1
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