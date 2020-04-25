from elmoformanylangs import Embedder
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
from torchtext import data
from torchtext import datasets
from sklearn.metrics import accuracy_score

class MyModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, layer, output_size):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.rnn_type = "LSTM"
        self._hidden_size = hidden_size
        self._layer = layer

    # text: batch,seq_len
    # hidden: batch,hidden_size
    def forward(self, text, hidden):
        # embedding: batct, seq_len, embed_size,
        embedding = self.embedding(text)

        # output: batch,seq_len, direction*hidden_size
        # hidden:(h_0, c_0)
        # h_0: batch, layers*diection,hidden_size
        # c_0: batch, layers*diection,hidden_size
        output, hidden = self.lstm(embedding)

        #out:batch,output_size
        out = self.fc(t.squeeze(hidden[0]))
        return out, hidden

    def init_hidden(self, batch, requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad),
                    weight.new_zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad))
        else:
            return weight.new_zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad)


e = Embedder('/home/demo1/womin/piguanghua/data/pre_data/zhs.model')
sents = [['今', '天', '天氣', '真', '好', '阿'],
['潮水', '退', '了', '就', '知道', '誰', '沒', '穿', '褲子']]
print(e.sents2elmo(sents))