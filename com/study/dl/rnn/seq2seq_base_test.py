import os
import sys
import math
from collections import Counter
import numpy as np
import random

import torch.nn as nn
import torch.nn.functional as F

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

import nltk


def load_data(in_file):
    cn = []
    en = []
    num_examples = 0
    with open(in_file, 'r') as f:
        for line in f:
            line = line.strip().split("\t")

            en.append(["BOS"] + nltk.word_tokenize(line[0].lower()) + ["EOS"])
            # split chinese sentence into characters
            cn.append(["BOS"] + [c for c in line[1]] + ["EOS"])
    return en, cn


train_file = "/home/demo1/womin/piguanghua/data/cmn.txt"
dev_file = "/home/demo1/womin/piguanghua/data/cmn.txt"
train_en, train_cn = load_data(train_file)
dev_en, dev_cn = load_data(dev_file)

UNK_IDX = 0
PAD_IDX = 1
def build_dict(sentences, max_words=21116):
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1
    ls = word_count.most_common(max_words)
    total_words = len(ls) + 2
    word_dict = {w[0]: index+2 for index, w in enumerate(ls)}
    word_dict["UNK"] = UNK_IDX
    word_dict["PAD"] = PAD_IDX
    return word_dict, total_words

en_dict, en_total_words = build_dict(train_en)
cn_dict, cn_total_words = build_dict(train_cn)
inv_en_dict = {v: k for k, v in en_dict.items()}
inv_cn_dict = {v: k for k, v in cn_dict.items()}


# 把单词全部转变成数字
def encode(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len=True):
    '''
        Encode the sequences.
    '''
    length = len(en_sentences)
    out_en_sentences = [[en_dict.get(w, 0) for w in sent] for sent in en_sentences]
    out_cn_sentences = [[cn_dict.get(w, 0) for w in sent] for sent in cn_sentences]

    # sort sentences by english lengths
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    # 把中文和英文按照同样的顺序排序
    if sort_by_len:
        sorted_index = len_argsort(out_en_sentences)
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]

    return out_en_sentences, out_cn_sentences


train_en, train_cn = encode(train_en, train_cn, en_dict, cn_dict)
dev_en, dev_cn = encode(dev_en, dev_cn, en_dict, cn_dict)


#把全部句子分成batch
def get_minibatches(n, minibatch_size, shuffle=True):
    idx_list = np.arange(0, n, minibatch_size) # [0, 1, ..., n-1]
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches

def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)

    x = np.zeros((n_samples, max_len)).astype('int32')
    x_lengths = np.array(lengths).astype("int32")
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    return x, x_lengths #x_mask

def gen_examples(en_sentences, cn_sentences, batch_size):
    minibatches = get_minibatches(len(en_sentences), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_en_sentences = [en_sentences[t] for t in minibatch]
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]
        mb_x, mb_x_len = prepare_data(mb_en_sentences)
        mb_y, mb_y_len = prepare_data(mb_cn_sentences)
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len))
    return all_ex

batch_size = 64
train_data = gen_examples(train_en, train_cn, batch_size)
random.shuffle(train_data)
dev_data = gen_examples(dev_en, dev_cn, batch_size)


class PlainEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size,hidden_size):
        super(PlainEncoder, self).__init__()

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
        super(PlainSeq2Seq, self).__init__()
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
        data = t.cat(t.Tensor(preds),t.Tensor(1))
        return data, None
        #return t.cat(preds, 1), None

# masked cross entropy loss
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # input: (batch_size * seq_len) * vocab_size
        input = input.contiguous().view(-1, input.size(2))
        # target: batch_size * 1
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout = 0.2
hidden_size = 100
embed_size  = 100
encoder = PlainEncoder(vocab_size=en_total_words,
                       embed_size = embed_size,
                      hidden_size=hidden_size)
decoder = PlainDecoder(vocab_size=cn_total_words,
                        embed_size = embed_size,
                      hidden_size=hidden_size)
model = PlainSeq2Seq(encoder, decoder)
model = model.to(device)
loss_fn = LanguageModelCriterion().to(device)
optimizer = torch.optim.Adam(model.parameters())


def evaluate(model, data):
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len-1).to(device).long()
            mb_y_len[mb_y_len<=0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    print("Evaluation loss", total_loss/total_num_words)


def train(model, data, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        total_num_words = total_loss = 0.
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = t.from_numpy(mb_x).to(device).long()
            mb_x_len = t.from_numpy(mb_x_len).to(device).long()
            mb_input = t.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = t.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = t.from_numpy(mb_y_len - 1).to(device).long()
            mb_y_len[mb_y_len <= 0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = t.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = t.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

            # 更新模型
            optimizer.zero_grad()
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            if it % 100 == 0:
                print("Epoch", epoch, "iteration", it, "loss", loss.item())

        print("Epoch", epoch, "Training loss", total_loss / total_num_words)
        if epoch % 5 == 0:
            evaluate(model, dev_data)


#train(model, train_data, num_epochs=20)


def translate_dev(i):
    en_sent = " ".join([inv_en_dict[w] for w in dev_en[i]])
    print(en_sent)
    cn_sent = " ".join([inv_cn_dict[w] for w in dev_cn[i]])
    print("".join(cn_sent))

    mb_x = t.from_numpy(np.array(dev_en[i]).reshape(1, -1)).long().to(device)
    mb_x_len = t.from_numpy(np.array([len(dev_en[i])])).long().to(device)
    bos = t.Tensor([[cn_dict["BOS"]]]).long().to(device)

    translation, attn = model.translate(mb_x, mb_x_len, bos)
    print("translate_dev")
    translation = [inv_cn_dict[i] for i in translation.data.cpu().numpy().reshape(-1)]
    trans = []
    for word in translation:
        if word != "EOS":
            trans.append(word)
        else:
            break
    print("".join(trans))

for i in range(101,120):
    translate_dev(i)
    print()