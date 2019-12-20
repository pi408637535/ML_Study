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
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

USE_CUDA = t.cuda.is_available()
device = t.device("cuda" if USE_CUDA else "cpu")

class LstmMode(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(LstmMode, self).__init__()
        self._lstm = nn.LSTM(
            vocab_size,
            hidden_size,
            batch_first=True
        )
        self._layer = 1
        self._hidden_size = hidden_size
        self._rnn_type = "LSTM"

    def forward(self, text, seq_lengths):

        batch = text.shape[0]
        self.hidden = self.init_hidden(batch)

        sorted_seq_lengths, indices = t.sort(seq_lengths, descending=True)
        inputs = text[indices]

        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs,
                                                          sorted_seq_lengths.cpu().numpy(),
                                                          batch_first=True)

        packed_out, hid = self._lstm(packed_inputs)

        padded_res, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        _, desorted_indices = t.sort(indices, descending=False)
        out = padded_res[desorted_indices] #gain original order

        out = out[desorted_indices.long()].contiguous()
        #hid[0] = hid[0][:,desorted_indices.long()].contiguous()
        #hid[1] = hid[1][:, desorted_indices.long()].contiguous()

        #out = out[original_idx.long()].contiguous()
        #hid = hid[:, original_idx.long()].contiguous()

        return out, (hid[0][:,desorted_indices.long()].contiguous(), hid[1][:, desorted_indices.long()].contiguous())

    def init_hidden(self,  batch, requires_grad=True):
        weight = next(self.parameters()).data
        if self._rnn_type == 'LSTM':
            hidden = (weight.new(self._layer, batch, self._hidden_size).zero_().to(device),
                      weight.new(self._layer, batch, self._hidden_size).zero_().to(device))
            return hidden
        else:
             return t.zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad)



if __name__ == '__main__':
    batch = 3
    seq_len = 5
    input_dim = 1
    hidden_size = 2

    text = t.randn([batch, seq_len, input_dim])
    seq_lengths = t.tensor([3, 5, 2], dtype=t.long)

    model = LstmMode(input_dim, hidden_size)


    encoder_out, hid = model(text, seq_lengths)
    print(encoder_out, hid)

    model = nn.LSTM(input_dim, hidden_size)
    encoder_out, hid = model(text)
    print(encoder_out, hid)
