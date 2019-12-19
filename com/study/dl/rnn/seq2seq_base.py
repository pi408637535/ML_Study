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


class PlainEocoder(nn.Module):
    def __init__(self, vocab_size, embed_size,hidden_size):
        super(MyRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self._lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            batch_first=True
        )

    #text: batch,seq
    #seq_lengths:每句长度
    def forward(self, text, seq_lengths):
        batch = text.shape[0]

        sorted_seq_lengths, indices = t.sort(seq_lengths, descending=True)
        inputs = text[indices]

        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs,
                                                          sorted_seq_lengths.cpu().numpy(),
                                                          batch_first=True)

        packed_out, hid = self._lstm(packed_inputs)

        padded_res, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        _, desorted_indices = t.sort(indices, descending=False)
        desorted_res = padded_res[desorted_indices]

        # out = out[original_idx.long()].contiguous()
        # hid = hid[:, original_idx.long()].contiguous()
