# -*- coding: utf-8 -*-
# @Time    : 2020/5/8 16:07
# @Author  : piguanghua
# @FileName: Transformer.py
# @Software: PyCharm

import os
import sys
import math
from collections import Counter
import numpy as np
import random

import torch.nn as nn
import torch.nn.functional as F

import torch as t
import torch
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
from torch.utils.data import Dataset, DataLoader


dim=512
dk_dim = 64

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

#利用对角线之上及其对角线上依次为0
def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class  ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, atten_mask):
        scores = t.matmul(Q, K.contiguous().transpose(-2,-1)) / np.sqrt(self.d_k)
        #score:batch,8,seq,seq
        scores.masked_fill_(atten_mask, -1e9)
        atten =  F.softmax(scores, dim = -1)
        context = t.matmul(atten, V)

        #atten:batch, head, seq,seq
        #context:batch, head,seq,64
        return atten, context

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, d_k, head, atten_model):
        super(MultiHeadAttention, self).__init__()
        self.Wq = nn.Linear(dim, d_k * head)
        self.Wk = nn.Linear(dim, d_k * head)
        self.Wv = nn.Linear(dim, d_k * head)
        self.Wz = nn.Linear(d_k * head, dim)
        self.head = head
        self.d_k = d_k
        self.atten_model = atten_model

    #设计失误啊,对于Encoder来说:Q,K,V都来自Encoder但对于Decoder来说Q来自Decoder的input，K,V来自Encoder的output
    def forward(self, q, k, v, atten_mask):
        #batch,seq,embed: batch,seq,512
        batch = q.shape[0]
        Q = self.Wq(q)  #batch,seq,64 * 8
        K = self.Wk(k) #batch,seq,64 * 8
        V = self.Wv(v) #batch,seq,64 * 8

        Q = Q.contiguous().view(batch,-1, self.head, self.d_k).\
            contiguous().transpose(1, 2) #Q:batch,head,seq,d_k:batch,8,seq,64

        K = K.contiguous().view(batch, -1, self.head, self.d_k). \
            contiguous().transpose(1, 2)  # Q:batch,head,seq,d_k:batch,8,seq,64

        V = V.contiguous().view(batch, -1, self.head, self.d_k). \
            contiguous().transpose(1, 2)  # Q:batch,head,seq,d_k:batch,8,seq,64

        atten_mask = t.unsqueeze(atten_mask, dim = 1) #batch,1,seq,seq, 第一个seq代表句长，第二个seq代表这个单词与其所在的seq dot score
        atten, context = self.atten_model(Q, K, V, atten_mask)
        context = context.contiguous().transpose(1,2).contiguous().\
            view(batch, -1, self.head * self.d_k)
        Z =  self.Wz(context)

        return nn.LayerNorm(512)(Z + q),atten

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, dim):
        super(PoswiseFeedForwardNet, self).__init__()
        #dim = 512
        self.net = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )
    def forward(self, input):
        return self.net(input)


class EncoderLayer(nn.Module):
    def __init__(self,self_att_mode, ffn):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = self_att_mode
        self.ffn = ffn

    def forward(self, enc_inputs, enc_self_attn_mask):

        #enc_inputs:batch,seq,512
        #enc_self_attn_mask:batch,seq,seq
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]

        #enc_outputs: batch,seq,dim:batch,seq,512
        #atten:bat,head,seq,seq
        return enc_outputs,attn


class Encoder(nn.Module):
    def __init__(self, vocab, dim, d_k, head):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab, dim)
        #self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len + 1, dim), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(MultiHeadAttention(dim, d_k, head, ScaledDotProductAttention(d_k)),
                                                  PoswiseFeedForwardNet(dim) ) for _ in range(head)])

    def forward(self, enc_inputs):  # enc_inputs : [batch_size x source_len]
        #enc_outputs = self.embed(enc_inputs) +
        enc_outputs = self.embed(enc_inputs)

        #?enc_self_attn_mask:batch,seq,seq
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attens = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attens.append(enc_self_attn)
        return enc_outputs, enc_self_attens

class DecoderLayer(nn.Module):
    def __init__(self,self_att_mode, enc_deco_attn, ffn):
        super(DecoderLayer, self).__init__()
        self.deco_self_attn = self_att_mode
        self.enc_deco_attn = enc_deco_attn
        self.ffn = ffn

    def forward(self, deco_inputs, enc_output, deco_self_attn_mask):
        #deco_inputs:batch,seq,dim:batch,seq,512
        #deco_self_attn_mask:batch,seq,seq
        #enc_output:batch, seq, 512
        deco_outputs, deco_attn = self.deco_self_attn(deco_inputs,deco_inputs
                                                 ,deco_inputs,deco_self_attn_mask)
        deco_outputs, dec_enc_attn = self.enc_deco_attn(deco_outputs, enc_output,
                                                          enc_output, deco_self_attn_mask)

        enc_outputs = self.ffn(deco_outputs)
        return enc_outputs, deco_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self, vocab, dim, d_k, head):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab, dim)
        #self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len + 1, dim), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer(MultiHeadAttention(dim, d_k, head, ScaledDotProductAttention(d_k)),
                                                  MultiHeadAttention(dim, d_k, head, ScaledDotProductAttention(d_k)),
                                                  PoswiseFeedForwardNet(dim) ) for _ in range(head)])

    def forward(self, deco_inputs, enc_outputs):  # enc_inputs : [batch_size x source_len]
        #enc_outputs = self.embed(enc_inputs) +
        deco_outputs = self.embed(deco_inputs)

        #?enc_self_attn_mask:batch,seq,seq
        deco_self_attn_mask = get_attn_subsequent_mask(deco_inputs)
        deco_self_attens = []
        deco_enc_attns = []
        for layer in self.layers:
            deco_outputs, deco_attn, dec_enc_attn = layer(deco_outputs, enc_outputs, deco_self_attn_mask)
            deco_self_attens.append(deco_attn)
            deco_enc_attns.append(dec_enc_attn)
        return deco_outputs, deco_self_attens, deco_enc_attns



class Transformer(nn.Module):
    def __init__(self, encoder, decoder, dim, vocab_size):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projection = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, text):
        enc_outputs, enc_self_atten = encoder(data)
        deco_outputs, deco_self_attens, deco_enc_attns = decoder(data, enc_outputs)
        dec_logits = self.projection(deco_outputs) #dec_logits:batch,seq,vocab
        return dec_logits, enc_self_atten, deco_self_attens, deco_enc_attns


if __name__ == '__main__':
    batch, seq = 2, 5
    data = t.randint(10, size=(batch , seq))
    vocab, dim, d_k, head = 20, 512, 64, 8
    encoder = Encoder(vocab, dim, d_k, head)
    decoder = Decoder(vocab, dim, d_k, head)
    model = Transformer(encoder, decoder, dim, vocab)

    enc_outputs, enc_self_atten = encoder(data)
    deco_outputs, deco_self_attens, deco_enc_attns = decoder(data, enc_outputs)
    dec_logits, enc_self_atten, deco_self_attens, deco_enc_attns = model(data)
    print(dec_logits.shape)

