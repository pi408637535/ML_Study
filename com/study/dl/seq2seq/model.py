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


import nltk

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
NUM_EPOCHS = 20
BATCH_SIZE = 32  # the batch size
LEARNING_RATE = 1e-3  # the initial learning rate
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 500
EPOCHS = 10



EN_WORD2ID = {}
EN_ID2WORD = {}
CH_WORD2ID = {}
CH_ID2WORD = {}
EN_VOCAB = None
CH_VOCAB = None
MAX_LEN = 20





def pretrain():

    global EN_WORD2ID,EN_ID2WORD,CH_WORD2ID,CH_ID2WORD,EN_VOCAB,CH_VOCAB,MAX_LEN

    def load_data(in_file):
        cn = []
        en = []
        num_examples = 0
        with open(in_file, 'r') as f:
            for line in f:
                line = line.strip().split("\t")

                if len(line[0]) > MAX_LEN or len(line[1]) > MAX_LEN:
                    continue

                en.append(["BOS"] + nltk.word_tokenize(line[0].lower()) + ["EOS"])
                # split chinese sentence into characters
                cn.append(["BOS"] + [c for c in line[1]] + ["EOS"])
        return en, cn

    train_file = "/home/demo1/womin/piguanghua/data/cmn.txt"
    #dev_file = "/home/demo1/womin/piguanghua/data/cmn.txt"
    train_en, train_cn = load_data(train_file)
    #dev_en, dev_cn = load_data(dev_file)
    print(len(train_en))

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

    en_dict, en_total_words = build_dict(train_en)
    cn_dict, cn_total_words = build_dict(train_cn)
    EN_VOCAB = en_total_words
    CH_VOCAB = cn_total_words

    en_word2id = { k:v for k, v in en_dict.items()}
    en_id2word = { v: k for k, v in en_dict.items()}

    cn_word2id = {k: v for k, v in cn_dict.items()}
    cn_id2word = {v: k for k, v in cn_dict.items()}

    EN_WORD2ID,EN_ID2WORD,CH_WORD2ID,CH_ID2WORD = en_word2id,en_id2word,cn_word2id,cn_id2word

    # 把单词全部转变成数字
    def encode(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len=True, maxlen = 20):
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
            for index,item in enumerate(source_sen):

                data = {
                    'source': pad_input(source_sen[index], MAX_LEN),
                    'source_len': len(source_sen[index]),
                    'target':  pad_input( target_sen[index], MAX_LEN),
                    'target_len': len(target_sen[index]),
                }
                all_data.append(data)
            return all_data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    return CustomDataSet(train_en, train_cn)

#No attention seq2seq finsh
class PlainEncoder(nn.Module):
    def __init__(self, vocab, dim, hidden, dropout = 0.2, layer = 2):
        super(PlainEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Embedding(vocab, dim),
            nn.Dropout(dropout),
            nn.GRU(dim, hidden, num_layers= layer, batch_first = True,
                   bidirectional = True, dropout = dropout)
        )

    '''
        text:batch,seq
        embed:batch,seq,embed
        gpu:batch,seq,embed
    '''
    def forward(self, text):
        '''
             (seq_len, batch, num_directions * hidden_size):
        '''
        output,h_n = self.net(text)
        '''
           output: seq,batch,direction*hidden
           h_0: layer*direction,batch,hidden 
        '''
        return output,h_n

class PlainDecoder(nn.Module):
    def __init__(self, vocab, dim, hidden, dropout = 0.2, layer = 2):
        super(PlainDecoder, self).__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(dim, hidden, num_layers=layer, batch_first=True,
                   bidirectional=True, dropout=dropout)
        self.fc = nn.Linear( 2  * hidden,vocab) #2 is bidirection


    def forward(self, text, hidden):
        embed = self.embed(text)
        embed = self.dropout(embed)
        output, h_n = self.gru(embed, hidden)
        '''
            output:batch,seq,direction*hidden
            h_0:layer*direction,batch,hidden
        '''
        output = self.fc(output) #batch,seq,vocab
        output = F.log_softmax(output, dim= 2)
        return output,None


class PlainSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(PlainSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, source, source_len, target, target_len):
        output,h_0 = self.encoder(source)
        target,atten = self.decoder(target, h_0)
        #target：seq,batch,direction*hidden
        return target,atten

    def translate(self, sources, source_len, target, target_len):
        global CH_ID2WORD,EN_WORD2ID
        output, h_0 = self.encoder(sources)
        target, atten = self.decoder(target, h_0)
        #batch,seq,vocab
        batch = target.shape[0]
        for item in range(batch):
            setence = target[item]
            #setence = t.unsqueeze(setence, dim = 0)
            setence = t.argmax(setence, dim = 1)
            setence = setence.detach().cpu().numpy()
            en_setence = [ EN_ID2WORD[word]  for word in setence]

            source = sources[item].detach().cpu().numpy()[:source_len[item].detach().cpu().item()]
            result = []
            for word in en_setence:
                if word != "EOS":
                    result.append(word)
            print( "{}_{}".format([ CH_ID2WORD[word]  for word in source], result) )

class Attention(nn.Module):
    def __init__(self,input_size,attention_size):
        super(Attention, self).__init__()
        self.net = nn.Sequential(
            nn.Linear( input_size, attention_size),
            F.tanh(),
            nn.Linear( attention_size,1)
        )

    '''
        enc_states:batch,seq, 2 * hidden #2 birdirection
        dec_state:batch,2 * hidden
        
    '''
    def forward(self, enc_states, dec_state):
        seq = enc_states.shape[1]

        dec_state = dec_state.expand(seq)
        pass


class AttenDecoder(nn.Module):
    def __init__(self, vocab, dim, hidden, atten, dropout = 0.2, layer = 2):
        super(PlainDecoder, self).__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(dim, hidden, num_layers=layer, batch_first=True,
                   bidirectional=True, dropout=dropout)
        self.fc = nn.Linear( 2  * hidden,vocab) #2 is bidirection
        self.atten = atten


    def forward(self, text, hidden):
        embed = self.embed(text)
        embed = self.dropout(embed)
        output, h_n = self.gru(embed, hidden)

        self.atten()

        '''
            output:batch,seq,direction*hidden
            h_0:layer*direction,batch,hidden
        '''
        output = self.fc(output) #batch,seq,vocab
        output = F.log_softmax(output, dim= 2)
        return output,None







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



def train(model, train_dataloader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):

        for batch_id, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()

            source = batch_data['source'].to(device)
            source_len = batch_data['source_len'].to(device)
            target = batch_data['target'].to(device)
            target_len = batch_data['target_len'].to(device)



            output = model(source, source_len, target, target_len)
            batch = target_len.shape[0]

            target_max_len = t.max(target_len).item()
            mask = t.arange(target_max_len).repeat(batch, 1).to(device) < target_len.view(-1, 1).expand(-1,target_max_len)
            mask = mask.float()
            loss = criterion(output[0], target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            if batch_id % 1e3 == 0:
                print(loss.item())

    save_dir = "./model"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_mode_path = '{}/{}.pkl'.format(save_dir,"seq2seq")


    model.eval()
    with t.no_grad():
        for batch_id, batch_data in enumerate(train_dataloader):
            source = batch_data['source'].to(device)
            source_len = batch_data['source_len'].to(device)
            target = batch_data['target'].to(device)
            target_len = batch_data['target_len'].to(device)
            model.translate(source, source_len, target, target_len)

    '''
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch
    }, best_mode_path)
    '''

def dev(model, train_dataloader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):

        for batch_id, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()

            source = batch_data['source'].to(device)
            source_len = batch_data['source_len'].to(device)
            target = batch_data['target'].to(device)
            target_len = batch_data['target_len'].to(device)



            output = model(source, source_len, target, target_len)
            batch = target_len.shape[0]

            target_max_len = t.max(target_len).item()
            mask = t.arange(target_max_len).repeat(batch, 1).to(device) < target_len.view(-1, 1).expand(-1,target_max_len)
            mask = mask.float()
            loss = criterion(output[0], target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            if batch_id % 1e3 == 0:
                print(loss.item())

    save_dir = "./model"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_mode_path = '{}/{}.pkl'.format(save_dir,"seq2seq")

    torch.save({
        'model': model.state_dict(),
        'epoch': epoch
    }, best_mode_path)


'''
 In order to accelerate replicating,I simply test procedure. 
'''
if __name__ == '__main__':
    train_dataloader = pretrain()
    dataloader = DataLoader(dataset=train_dataloader, batch_size=BATCH_SIZE)
    encoder = PlainEncoder(EN_VOCAB, EMBEDDING_SIZE, HIDDEN_SIZE)
    decoder = PlainDecoder(CH_VOCAB, EMBEDDING_SIZE, HIDDEN_SIZE)
    seq2seq = PlainSeq2Seq(encoder, decoder)
    criterion = LanguageCriterion()

    optimizer = t.optim.Adam(seq2seq.parameters(), lr=LEARNING_RATE)
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    seq2seq.to(device)
    train(seq2seq, dataloader, optimizer, criterion, EPOCHS, device)