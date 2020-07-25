# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 16:34
# @Author  : piguanghua
# @FileName: transformer_five.py
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
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from numpy import random
import matplotlib.pyplot as plt
from torchtext import data
from tqdm import tqdm

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
USE_CUDA = t.cuda.is_available()
random.seed(53113)
np.random.seed(53113)
t.manual_seed(53113)
if USE_CUDA:
    t.cuda.manual_seed(53113)

from torchtext.data import Iterator, BucketIterator

def positional_encoding(seq_len, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx) / d_model)

    def get_posi_angle_vec(pos):
        return [cal_angle(pos, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(seq_len)])
    sinusoid_table[:, 0::2] = np.sin( sinusoid_table[:, 0::2] )
    sinusoid_table[:, 1::2] = np.cos( sinusoid_table[:, 1::2] )

    return t.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def get_attn_subsequent_mask(seq_q):
    batch_size, len_q = seq_q.size()

    subsequent_mask = t.triu(t.ones((len_q,len_q)), diagonal=1).unsqueeze(0)
    subsequent_mask = subsequent_mask.expand(batch_size, len_q, len_q)


    return  subsequent_mask.byte() # batch_size x len_q x len_k



class Embedding(nn.Module):
    def __init__(self, vocab, dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab, dim)

    def forward(self, text):
        return self.embed(text)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k


    def forward(self, q, k, v, mask):
        scores = t.matmul(q, k.transpose(2, 3)) / np.sqrt(self.d_k)
        scores.masked_fill_(mask.unsqueeze(dim=1), -1e9)
        atten = F.softmax(scores, dim=-1)
        context = t.matmul(atten, v)

        return atten, context

class MultiHeadAttention(nn.Module):
    def __init__(self, head, dim, d_k):
        super(MultiHeadAttention, self).__init__()
        self.head = head
        self.dim = dim
        self.d_k = d_k
        self.Wq = nn.Linear(self.dim, self.d_k * self.head)
        self.Wk = nn.Linear(self.dim, self.d_k * self.head)
        self.Wv = nn.Linear(self.dim, self.d_k * self.head)

        self.Wz = nn.Linear(self.d_k * self.head, self.dim)
        self.norm = nn.LayerNorm(dim)
        self.atten_model = ScaledDotProductAttention(d_k)

    def forward(self, q, k, v, mask):
        batch, seq_len = q.shape[0],q.shape[1]
        residual = q

        Q = self.Wq(q)  # batch,seq,64 * 8
        K = self.Wk(k)  # batch,seq,64 * 8
        V = self.Wv(v)  # batch,seq,64 * 8

        Q = Q.view(batch,seq_len, self.d_k,self.head).permute(0, 3, 1, 2)  # Q:batch,head,seq,d_k:batch,8,seq,64

        K = K.view(batch,seq_len, self.d_k,self.head).permute(0, 3, 1, 2)  # Q:batch,head,seq,d_k:batch,8,seq,64

        V = V.view(batch,seq_len, self.d_k,self.head,).permute(0, 3, 1, 2)  # Q:batch,head,seq,d_k:batch,8,seq,64

        #atten:batch,head,seq,seq
        #batch,head,seq, d_64
        atten, context = self.atten_model(Q, K, V, mask)

        context = context.view(batch, seq_len, self.head * self.d_k)
        Z = self.Wz(context)

        return atten,self.norm(Z + residual)




class FFN(nn.Module):
    def __init__(self, dim, internal=1024):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(dim, internal)
        self.fc2 = nn.Linear(internal, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, data):
        temp = self.fc1(data)
        temp = self.fc2(temp)

        return self.norm(temp + data)


class EncoderLayer(nn.Module):
    def __init__(self, head, dim, d_k):
        super(EncoderLayer, self).__init__()
        self.head = head
        self.dim = dim
        self.d_k = d_k
        self.multi_head_attention = MultiHeadAttention(self.head, self.dim, self.d_k)
        self.ffn = FFN(self.dim)

    def forward(self, enc_inputs, mask):
        atten, data= self.multi_head_attention(enc_inputs, enc_inputs, enc_inputs, mask)
        return atten, self.ffn(data)

class Encoder(nn.Module):
    def __init__(self, head, dim, d_k, layer, vocab, seq_len):
        super(Encoder, self).__init__()
        self.head = head
        self.dim = dim
        self.d_k = d_k
        self.vocab = vocab
        self.seq_len = seq_len

        self.embed = Embedding( self.vocab, self.dim)
        self.pos_emb = nn.Embedding.from_pretrained(positional_encoding(self.seq_len, self.dim), freeze=True)
        self.encoder_block = nn.ModuleList([ EncoderLayer(self.head, self.dim, self.d_k) for i in range(layer) ])

    def forward(self, text):
        input = self.embed(text) + self.pos_emb(text)
        enc_self_attn_mask = get_attn_pad_mask(text, text)

        attentions = []
        for layer in self.encoder_block:
            atten, input = layer(input, enc_self_attn_mask)
            attentions.append(atten)

        return attentions,input

class DecoderLayer(nn.Module):
    def __init__(self, head, dim, d_k):
        super(DecoderLayer, self).__init__()
        self.head = head
        self.dim = dim
        self.d_k = d_k
        self.dec_self_attn = MultiHeadAttention(self.head, self.dim, self.d_k)
        self.dec_enc_attn = MultiHeadAttention(self.head, self.dim, self.d_k)
        self.ffn = FFN(self.dim)

    def forward(self, dec_input, enc_output, dec_self_mask, enc_dec_self_mask):
        dec_self_attn, dec_outputs = self.dec_self_attn(dec_input, dec_input, dec_input, dec_self_mask)
        dec_enc_attn, dec_outputs = self.dec_enc_attn(dec_outputs, enc_output, enc_output, enc_dec_self_mask)
        return dec_self_attn, dec_enc_attn, self.ffn(dec_outputs)


class Decoder(nn.Module):
    def __init__(self, head, dim, d_k, layer, vocab, seq_len):
        super(Decoder, self).__init__()
        self.head = head
        self.dim = dim
        self.d_k = d_k
        self.vocab = vocab
        self.seq_len = seq_len

        self.embed = Embedding(self.vocab, self.dim)
        self.pos_emb = nn.Embedding.from_pretrained(positional_encoding(self.seq_len, self.dim), freeze=True)
        self.decoder_block = nn.ModuleList([DecoderLayer(self.head, self.dim, self.d_k) for i in range(layer)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        embed = self.embed(dec_inputs) + self.pos_emb(dec_inputs)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        #dec_self_attn_pad_mask only 1,0
        #dec_self_attn_subsequent_mask only 0,1
        dec_self_attn_mask = t.gt(dec_self_attn_pad_mask + dec_self_attn_subsequent_mask,0)

        enc_dec_self_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns = []
        dec_enc_attns = []
        for layer in self.decoder_block:
            dec_self_attn, dec_enc_attn, dec_inputs = layer(embed, enc_outputs, dec_self_attn_mask, enc_dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_self_attns,dec_enc_attns,dec_inputs


class Transformer(nn.Module):
    def __init__(self, head, dim, d_k, layer, src_vocab, src_seq_len, dec_vocab, dec_seq_len):
        super(Transformer, self).__init__()
        self.encoder = Encoder(head = head, dim = dim, d_k = d_k, layer = layer, vocab=src_vocab, fix_length=src_seq_len)
        self.decoder = Decoder(head = head, dim = dim, d_k = d_k, layer = layer, vocab=dec_vocab, fix_length=dec_seq_len)
        self.projection = nn.Linear(dim, dec_vocab, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_self_attns, enc_outputs, = self.encoder(enc_inputs)
        dec_self_attn, dec_enc_attn, dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]

        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns



def tokenizer(text): # create a tokenizer function
    # 返回 a list of <class 'spacy.tokens.token.Token'>
    return [tok for tok in text.split(" ")]

if __name__ == '__main__':
    fix_length = 25
    SRC = data.Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True , fix_length=fix_length)
    TRG = data.Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True, fix_length=fix_length)
    data_fields = [("id",None),('src', SRC), ('trg', TRG)]
    file_name = "/Users/piguanghua/Downloads/translate_transformer_temp.csv"
    train = data.TabularDataset(file_name, format='csv', fields=data_fields)

    SRC.build_vocab(train)
    TRG.build_vocab(train)


    train_iter = BucketIterator(train, batch_size=1,
        device=-1,  # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: (len(x.src), len(x.trg)),
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=False,
        repeat=False)

    encoder = Encoder(head = 8, dim = 512, d_k = 64, layer = 6, vocab=len(SRC.vocab.itos), seq_len=fix_length)
    decoder = Decoder(head=8, dim=512, d_k=64, layer=6, vocab=len(TRG.vocab.itos), seq_len=fix_length)

    for i,ele in enumerate(train_iter):
        print(ele)
        attens, encoder_output = encoder(ele.src)
        dec_self_attns,dec_enc_attns,dec_inputs = decoder(ele.trg, ele.src, encoder_output)
        dec_inputs




