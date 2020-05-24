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
NUM_EPOCHS = 1
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

'''
    MAX_LEN:To limit the length of sentence.
'''





def pretrain():

    global EN_WORD2ID,EN_ID2WORD,CH_WORD2ID,CH_ID2WORD,EN_VOCAB,CH_VOCAB,MAX_LEN

    def load_data(in_file):
        cn = []
        en = []
        num_examples = 0
        with open(in_file, 'r') as f:
            for line in f:
                line = line.strip().split("\t")

                if len(line[0].split(" ")) > MAX_LEN or len(line[1].split(" ")) > MAX_LEN:
                    continue

                en.append(["BOS"] + nltk.word_tokenize(line[0].lower()) + ["EOS"])
                # split chinese sentence into characters
                #Todo chatbot进行修改
                cn.append(["BOS"] + nltk.word_tokenize(line[1].lower()) + ["EOS"])
                #cn.append(["BOS"] + [c for c in line[1]] + ["EOS"])
        return en, cn

    train_file = "/home/demo1/womin/piguanghua/data/cmn.txt"
    dev_file = "/home/demo1/womin/piguanghua/data/cmn_dev.txt"

    #train_file = "/home/demo1/womin/piguanghua/data/buaa/source_target.txt"
    #dev_file = "/home/demo1/womin/piguanghua/data/buaa/source_target.txt"

    train_en, train_cn = load_data(train_file)
    dev_en, dev_cn = load_data(dev_file)

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
    def encode(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len=False, maxlen = 20):
        '''
            Encode the sequences.
        '''



        length = len(en_sentences)
        out_en_sentences = [[en_dict.get(w, 0) for w in sent] for sent in en_sentences]
        out_cn_sentences = [[cn_dict.get(w, 0) for w in sent] for sent in cn_sentences]

        # sort sentences by english lengths


        #out_en_sentences = [out_en_sentences[i] for i in out_en_sentences]
        #out_cn_sentences = [out_cn_sentences[i] for i in out_cn_sentences]

        return out_en_sentences, out_cn_sentences

    train_en, train_cn = encode(train_en, train_cn, en_dict, cn_dict)
    dev_en, dev_cn = encode(dev_en, dev_cn, en_dict, cn_dict)

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

    train_data_set = CustomDataSet(train_en, train_cn)
    dev_data_set = CustomDataSet(dev_en, dev_cn)
    return train_data_set,dev_data_set

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
        return output,None  #batch,seq,vocab


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
            nn.Linear(input_size, attention_size, bias=False),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False))

    '''
        enc_states:batch,seq, 2 * hidden #2 birdirection
        dec_state:batch,2 * hidden
    '''
    def forward(self, enc_states, dec_state):
        """
            enc_states: (batch, seq, hidden)
            dec_state: (batch, hidden)
            """
        # 将解码器隐藏状态广播到和编码器隐藏状态形状相同后进行连结
        batch, seq, hidden = enc_states.shape
        dec_states = dec_state[:, None, :].repeat(1, seq, 1)

        enc_and_dec_states = t.cat((enc_states, dec_states), dim=2)
        enc_and_dec_states = enc_and_dec_states.float()
        e = self.net(enc_and_dec_states)  # (batch, seq, 1)  batch,word,score
        alpha = F.softmax(t.squeeze(e, dim=2), dim=1)  # 在时间步维度做softmax运算
        alpha = t.unsqueeze(alpha, dim=2) #batch,word,1 ->
        enc_states = enc_states.float()
        return (alpha * enc_states).sum(dim=1) #batch,hidden. context以这个维度返回主要是为了与Y(t-1)保持同一个维度


class AttenDecoder(nn.Module):
    def __init__(self, vocab, dim, hidden, atten, dropout = 0.2, layer = 2):
        super(AttenDecoder, self).__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(dim + hidden * 2, hidden, num_layers=layer, batch_first=True,
                   bidirectional=True, dropout=dropout)
        self.fc = nn.Linear( 2  * hidden,vocab) #2 is bidirection
        self.atten = atten

    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        return enc_state

    def forward(self, cur_input, y_state, enc_states):
        '''
            cur_input shape: batch
            y_state: num_layers, batch,hidden
            enc_states: batch,seq, hidden
        '''
        # 解码器在最初时间步的输入是BOS
        batch = enc_states.shape[0]
        if True: #bidirectional
            y_state_last = y_state[-2:, :, :] #2,batch,hidden
            #dec_state:batch,2 * hidden
            y_state_last = y_state_last.contiguous().transpose(0,1).contiguous().view(batch,-1)


        c = self.atten(enc_states, y_state_last) #对于Decoder,只选择最上层的hidden, c：batch,hidden

        #cat c which belong to this moment and cur_input
        input_and_c = torch.cat((self.embed(cur_input), c), dim=1) #input_and_c:batch,hidden

        #input_and_c.unsqueeze(1)——>batch,seq,hidden:batch,1,hidden
        output, state = self.gru(input_and_c.unsqueeze(1), y_state)

        #batch,1,hidden->batch,1,vocab -> batch,vocab
        output = self.fc(output).squeeze(dim=1)

        #
        output = F.log_softmax(output, dim=1)
        #output:batch,vocab
        #state:layer,batch,hidden
        #c:batch,hidden
        return output,state,c

class AttenSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder,device):
        super(AttenSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device


    def forward(self, source, source_len, target, target_len):

        global EN_WORD2ID
        #enc_output: batch,seq,embed
        #enc_state:layer, batch,hidden
        enc_output,enc_state = self.encoder(source)
        batch = source.shape[0]
        #num_layers, batch,hidden
        dec_state = self.decoder.begin_state(enc_state)
        # 解码器在最初时间步的输入是BOS
        dec_input = torch.tensor([EN_WORD2ID['BOS']] * batch).to(self.device)

        list_dec_output = []
        list_atten = []

        seq = target.shape[1]

        for index in range(seq):  # Y shape: (batch, seq_len)
            #dec_state: layer,batch,hidden:2*2,32,500
            #dec_input:batch:32
            #enc_output:batch,seq,hidden:32,20,1000
            dec_output, dec_state,c = self.decoder(dec_input, dec_state, enc_output)
            #dec_output：batch,vocab
            list_dec_output.append(dec_output)
            list_atten.append(c)

            dec_input = target[:,index]  # 使用强制教学 逐个输入dec_input

        #target：seq,batch,direction*hidden
        model_output = t.stack(list_dec_output, dim = 1)
        attens = t.stack(list_atten, dim = 1)
        #model_output:batch,seq,vocab: 32,20,vacab
        #attens:batch,hidden:32,1000
        return model_output,attens


    def translate(self, sources, source_len, target):
        global CH_ID2WORD,EN_WORD2ID
        #target:batch,seq,vocab
        #attens:batch,hidden:32,1000

        #batch,seq,vocab
        batch = target.shape[0]
        for item in range(batch):
            setence = target[item]
            #setence = t.unsqueeze(setence, dim = 0)
            setence = t.argmax(setence, dim = 1)
            setence = setence.detach().cpu().numpy()
            cn_setence = [ CH_ID2WORD[word]  for word in setence]

            source = sources[item].detach().cpu().numpy()[:source_len[item].detach().cpu().item()]
            result = []
            for word in cn_setence:
                if word != "EOS":
                    result.append(word)
            print( "{}_{}".format([ EN_ID2WORD[word]  for word in source], result) )

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

    '''
    save_dir = "./model"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_mode_path = '{}/{}.pkl'.format(save_dir,"seq2seq")
    '''



    '''
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch
    }, best_mode_path)
    '''

def dev(model, train_dataloader, optimizer, criterion, epochs, device):
    model.eval()
    with t.no_grad():
        for batch_id, batch_data in enumerate(train_dataloader):
            source = batch_data['source'].to(device)
            source_len = batch_data['source_len'].to(device)
            target = batch_data['target'].to(device)
            target_len = batch_data['target_len'].to(device)

            output = model(source, source_len, target, target_len)
            model.translate(source, source_len, output[0])


'''
 In order to accelerate replicating,I simply test procedure. 
'''

def plain_seq2seq():
    train_dataloader,dev_dataloader = pretrain()
    dataloader = DataLoader(dataset=train_dataloader, batch_size=BATCH_SIZE)
    encoder = PlainEncoder(EN_VOCAB, EMBEDDING_SIZE, HIDDEN_SIZE)
    decoder = PlainDecoder(CH_VOCAB, EMBEDDING_SIZE, HIDDEN_SIZE)
    seq2seq = PlainSeq2Seq(encoder, decoder)
    criterion = LanguageCriterion()

    optimizer = t.optim.Adam(seq2seq.parameters(), lr=LEARNING_RATE)
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    seq2seq.to(device)
    train(seq2seq, dataloader, optimizer, criterion, EPOCHS, device)

def attention_seq2seq():
    train_dataloader, dev_dataloader = pretrain()
    train_dataloader = DataLoader(dataset=train_dataloader, batch_size=BATCH_SIZE)
    dev_dataloader = DataLoader(dataset=dev_dataloader, batch_size=BATCH_SIZE)

    encoder = PlainEncoder(EN_VOCAB, EMBEDDING_SIZE, HIDDEN_SIZE)
    atten = Attention(HIDDEN_SIZE * 4, 300)  # attention neural cell

    decoder = AttenDecoder(CH_VOCAB, EMBEDDING_SIZE, HIDDEN_SIZE, atten)

    device = torch.device('cuda' if USE_CUDA else 'cpu')
    seq2seq = AttenSeq2Seq(encoder, decoder, device)
    criterion = LanguageCriterion()

    optimizer = t.optim.Adam(seq2seq.parameters(), lr=LEARNING_RATE)

    seq2seq.to(device)
    train(seq2seq, train_dataloader, optimizer, criterion, EPOCHS, device)
    dev(seq2seq, dev_dataloader, optimizer, criterion, EPOCHS, device)

if __name__ == '__main__':
    plain_seq2seq()
    attention_seq2seq()