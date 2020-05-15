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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from numpy import random
import matplotlib.pyplot as plt

#hypotrical parameter
USE_CUDA = t.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
t.manual_seed(53113)
if USE_CUDA:
    t.cuda.manual_seed(53113)
UNK_IDX = 0
PAD_IDX = 1
NUM_EPOCHS = 1
BATCH_SIZE = 32  # the batch size
LEARNING_RATE = 1e-3  # the initial learning rate
EPOCHS = 5000

EN_WORD2ID = {}
EN_ID2WORD = {}
CH_WORD2ID = {}
CH_ID2WORD = {}
EN_VOCAB = None
CH_VOCAB = None
MAX_LEN = 25

def pre_train():
    global EN_WORD2ID, EN_ID2WORD, CH_WORD2ID, CH_ID2WORD, EN_VOCAB, CH_VOCAB, MAX_LEN

    def build_dict(sentences, max_words=21116):
        word_count = Counter()
        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1
        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict["UNK"] = UNK_IDX
        word_dict["PAD"] = PAD_IDX
        return word_dict, total_words

    train_en = ["Anthony Fauci says the best evidence shows the virus behind the pandemic was not made in a lab in China.".split(" ")]
    train_cn = ["安东尼·福奇 说，最 有力的 证据 表明，造成 疫情 的 新冠病毒 并 不是 中国的 实验室 人为 制造 的。".split(" ")]

    en_dict, en_total_words = build_dict(train_en)
    cn_dict, cn_total_words = build_dict(train_cn)
    EN_VOCAB = en_total_words
    CH_VOCAB = cn_total_words

    en_word2id = {k: v for k, v in en_dict.items()}
    en_id2word = {v: k for k, v in en_dict.items()}

    cn_word2id = {k: v for k, v in cn_dict.items()}
    cn_id2word = {v: k for k, v in cn_dict.items()}

    EN_WORD2ID, EN_ID2WORD, CH_WORD2ID, CH_ID2WORD = en_word2id, en_id2word, cn_word2id, cn_id2word

    # 把单词全部转变成数字
    def encode(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len=False, maxlen=20):
        '''
            Encode the sequences.
        '''

        length = len(en_sentences)
        out_en_sentences = [[en_dict.get(w, 0) for w in sent] for sent in en_sentences]
        out_cn_sentences = [[cn_dict.get(w, 0) for w in sent] for sent in cn_sentences]

        # sort sentences by english lengths

        # out_en_sentences = [out_en_sentences[i] for i in out_en_sentences]
        # out_cn_sentences = [out_cn_sentences[i] for i in out_cn_sentences]

        return out_en_sentences, out_cn_sentences

    train_en, train_cn = encode(train_en, train_cn, en_dict, cn_dict)
    #dev_en, dev_cn = encode(dev_en, dev_cn, en_dict, cn_dict)

    class CustomDataSet(Dataset):

        def __init__(self, source_sen, target_sen, transform=None):
            self.data = self._load_dataset(source_sen, target_sen)
            self._transform = transform

        def _load_dataset(self, source_sen, target_sen):

            def pad_input(sentence, seq_len):
                length = len(sentence)
                feature = np.zeros(seq_len, dtype=int)
                if len(sentence) != 0:
                    feature[:length] = sentence[:seq_len]
                return feature

            all_data = []
            for index, item in enumerate(source_sen):
                data = {
                    'source': pad_input(source_sen[index], MAX_LEN),
                    'source_len': len(source_sen[index]),
                    'target': pad_input(target_sen[index], MAX_LEN),
                    'target_len': len(target_sen[index]),
                }
                all_data.append(data)
            return all_data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    train_data_set = CustomDataSet(train_en, train_cn)
    #dev_data_set = CustomDataSet(dev_en, dev_cn)
    return train_data_set


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

class ScaledDotProductAttention(nn.Module):
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
        self.norm = nn.LayerNorm(512)

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

        return self.norm(Z + q),atten

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
    def __init__(self, vocab, dim, d_k, head, device):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab, dim)
        #self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len + 1, dim), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(MultiHeadAttention(dim, d_k, head, ScaledDotProductAttention(d_k)),
                                                  PoswiseFeedForwardNet(dim) ) for _ in range(head)])
        self.device = device

    def forward(self, enc_inputs):  # enc_inputs : [batch_size x source_len]
        #enc_outputs = self.embed(enc_inputs) +
        enc_outputs = self.embed(enc_inputs)

        #?enc_self_attn_mask:batch,seq,seq
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attn_mask = enc_self_attn_mask.to(device)
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
    def __init__(self, vocab, dim, d_k, head, device):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab, dim)
        #self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len + 1, dim), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer(MultiHeadAttention(dim, d_k, head, ScaledDotProductAttention(d_k)),
                                                  MultiHeadAttention(dim, d_k, head, ScaledDotProductAttention(d_k)),
                                                  PoswiseFeedForwardNet(dim) ) for _ in range(head)])
        self.device = device

    def forward(self, deco_inputs, enc_outputs):  # enc_inputs : [batch_size x source_len]
        #enc_outputs = self.embed(enc_inputs) +
        deco_outputs = self.embed(deco_inputs)

        #?enc_self_attn_mask:batch,seq,seq
        deco_self_attn_mask = get_attn_subsequent_mask(deco_inputs)
        deco_self_attn_mask = deco_self_attn_mask.to(self.device)

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


    def forward(self, source, target):
        enc_outputs, enc_self_atten = encoder(source)
        deco_outputs, deco_self_attens, deco_enc_attns = decoder(target, enc_outputs)
        dec_logits = self.projection(deco_outputs) #dec_logits:batch,seq,vocab
        return dec_logits, enc_self_atten, deco_self_attens, deco_enc_attns

    '''
    def translate(self, sources):
        #sources: batch,seq,512
        global CH_ID2WORD, EN_WORD2ID
        batch = sources.shape[0]

        for item in range(batch):
            setence = target[item]
            # setence = t.unsqueeze(setence, dim = 0)
            setence = t.argmax(setence, dim=1)
            setence = setence.detach().cpu().numpy()
            cn_setence = [CH_ID2WORD[word] for word in setence]

            source = sources[item].detach().cpu().numpy()[:25]
            result = []
            for word in cn_setence:
                if word != "EOS":
                    result.append(word)
            print("{}_{}".format([EN_ID2WORD[word] for word in source], result))
    '''

class LanguageCriterion(nn.Module):
    def __init__(self):
        super(LanguageCriterion, self).__init__()

    '''
    :arg
        model_target:batch,seq,vocab
        text_target: batch,seq
        mask: batch,seq
    '''
    def forward(self, model_target, text_target, mask):
        batch_max_length = mask.shape[1]
        text_target = text_target[:, :batch_max_length]
        model_target = model_target[:, :batch_max_length, :]
        batch,seq = model_target.shape[0],model_target.shape[1]
        model_target = model_target.contiguous().view(batch * seq, -1)
        model_target = F.log_softmax(model_target, dim= 1)

        text_target = text_target.contiguous().view(-1,1)

        loss = -t.gather(model_target, dim=1, index = text_target).view(batch,-1) * mask
        loss = loss.sum() / mask.sum()
        return loss


def showgraph(attn, file_name):
    n_heads = 8
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    #ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    #ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})

    train_en = "Anthony Fauci says the best evidence shows the "
    train_cn = "安东尼·福奇 说，最 有力的 证据 表明，造成 疫情 "

    ax.set_xticklabels(['']+train_en.split(" "), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+train_cn.split(" "), fontdict={'fontsize': 14})

    path = "/home/demo1/{}.png".format(file_name)
    plt.savefig(path)
    plt.show()



def train(model, train_dataloader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):

        for batch_id, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()

            source = batch_data['source'].to(device)
            source_len = batch_data['source_len'].to(device)
            target = batch_data['target'].to(device)
            target_len = batch_data['target_len'].to(device)



            dec_logits, enc_self_atten, deco_self_attens, deco_enc_attns = model(source, target)

            batch = target_len.shape[0]

            target_max_len = t.max(target_len).item()
            mask = t.arange(target_max_len).repeat(batch, 1).to(device) < target_len.view(-1, 1).expand(-1,target_max_len)
            mask = mask.float()
            loss = criterion(dec_logits, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            if batch_id % 1e3 == 0:
                print(loss.item())


    print('first head of last state enc_self_attns')
    showgraph(enc_self_atten[-1].cpu(), "enc_self_atten")

    print('first head of last state dec_self_attns')
    showgraph(deco_self_attens[-1].cpu(), "deco_self_attens")

    print('first head of last state dec_enc_attns')
    showgraph(deco_enc_attns[-1].cpu(), "deco_enc_attns")

    model.eval()
    output = model(source, target)
    #print(model.translate(source))



if __name__ == '__main__':

    batch, seq = 2, 5
    data = t.randint(10, size=(batch , seq))
    vocab, dim, d_k, head = 20, 512, 64, 8



    #enc_outputs, enc_self_atten = encoder(data)
    #deco_outputs, deco_self_attens, deco_enc_attns = decoder(data, enc_outputs)
    #dec_logits, enc_self_atten, deco_self_attens, deco_enc_attns = model(data)

    train_dataloader = pre_train()
    train_dataloader = DataLoader(dataset=train_dataloader, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if USE_CUDA else 'cpu')
    criterion = LanguageCriterion()
    encoder = Encoder(vocab, dim, d_k, head, device)
    decoder = Decoder(vocab, dim, d_k, head, device)
    model = Transformer(encoder, decoder, dim, vocab)
    optimizer = t.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.to(device)


    train(model, train_dataloader, optimizer, criterion, EPOCHS, device)



