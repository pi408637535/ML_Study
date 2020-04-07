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


class PlainEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size,hidden_size):
        super(PlainEocoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self._lstm = nn.GRU(
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

        embed = self.embedding(inputs)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(embed,
                                                          sorted_seq_lengths.cpu().numpy(),
                                                          batch_first=True)

        packed_out, hid = self._lstm(packed_inputs)

        padded_res, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        _, desorted_indices = t.sort(indices, descending=False)
        out = padded_res[desorted_indices].contiguous()
        hid = hid[:, desorted_indices].contiguous()



        # out = out[original_idx.long()].contiguous()
        # hid = hid[:, original_idx.long()].contiguous()
        return out, hid


class PlainDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size,hidden_size):
        super(PlainDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self._lstm = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self._fc = nn.Linear(hidden_size, vocab_size)


    #text: batch,seq
    #seq_lengths:每句长度
    def forward(self, text, seq_lengths, hid):
        batch = text.shape[0]

        sorted_seq_lengths, indices = t.sort(seq_lengths, descending=True)
        inputs = text[indices]

        embed = self.embedding(inputs)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(embed,
                                                          sorted_seq_lengths.cpu().numpy(),
                                                          batch_first=True)

        packed_out, hid = self._lstm(packed_inputs, hid)

        padded_res, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        _, desorted_indices = t.sort(indices, descending=False)
        out = padded_res[desorted_indices].contiguous()
        hid = hid[:, desorted_indices].contiguous()

        output = F.log_softmax(self._fc(out), -1)

        return output, hid


class PlainSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(PlainDecoder, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, x, x_length, y, y_length):
        encoder_out, hid = self._encoder(x, x_length)
        output, hid = self._decoder(y, y_length, hid)
        return output,None

    def translate(self, x, x_lengths, y, max_length = 0):
        encoder_out, hid = self._encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        for i in range(max_length):
            output, hid = self._decoder(y, t.ones(batch_size).long().to(device),
                                        hid)
            y = output.max(2)[1]
            preds.append(y)
        return t.cat(preds, 1), None

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # input: (batch_size * seq_len) * vocab_size
        input = input.contiguous().view(-1, input.size(2))


