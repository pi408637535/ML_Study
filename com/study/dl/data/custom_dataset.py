import pandas as pd
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
from torch.utils.data import Dataset, DataLoader
import os
import copy,time
import sklearn.datasets
import matplotlib.pyplot as plt



class CustomDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        path = os.path.join(root_dir, csv_file)
        self.landmarks_frame = pd.read_csv(path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame.index)

    def __getitem__(self, index):
        if t.is_tensor(index):
            index = index.tolist()
        single_image_label = self.landmarks_frame["Label"][index]
        data = self.landmarks_frame["pixel_1"][index]

        return (data, single_image_label)



words = {k:v for k,v in {'the':1, 'book':5}.items() }
words =  sorted(words, key=words.get, reverse=True)
words = ['PAD','UNK'] + words
word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}

class DFCustomDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        path = os.path.join(root_dir, csv_file)

        self.root_dir = root_dir
        self.transform = transform
        self._seq_len = 150
        self._df = self._load_dataset(path)

    def _load_dataset(self, data_path):

        df = pd.read_csv(data_path)

        sentences = []
        for i, sentence in enumerate(df["setence"].tolist()):
            # Looking up the mapping dictionary and assigning the index to the respective words
            df["setence"][i] = []
            for word in sentence.split(" "):
                df["setence"][i].append(word2idx[word.lower()] if word.lower() in word2idx else 0)

        seq_len = self._seq_len
        def pad_input(sentence):

            feature = np.zeros(seq_len, dtype=int)
            if len(sentence) != 0:
                feature[:len(sentence)] = sentence[:seq_len]
            return feature


        def pad_input2(sentences, seq_len):
            features = np.zeros((len(sentences), seq_len), dtype=int)
            for ii, review in enumerate(sentences):
                if len(review) != 0:
                    features[ii, :len(review)] = review[:seq_len]
                    # features[ii, -len(review):] = np.array(review)[:seq_len]
            return features

        seq_len = 150  # The length that the sentences will be padded/shortened to

        #df["setence"] = df["setence"].apply(pad_input)
        df = df.apply(pad_input, axis=0)

        #df = df.drop(['setence'], axis=1)
        return df


    def __len__(self):
        return self._df.shape[0]

    #label,setence,length
    def __getitem__(self, index):
        if t.is_tensor(index):
            index = index.tolist()

        return self._df.iloc[index].tolist()




USE_CUDA = t.cuda.is_available()
if USE_CUDA:
    device = t.device("cuda")
else:
    device = t.device("cpu")

class TwoClassClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, tgt_size):
        super(TwoClassClassifier, self).__init__()
        self._embed = nn.Embedding(vocab_size, embed_size)
        self._lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self._fc = nn.Linear(hidden_size, tgt_size)
        self._hidden_size = hidden_size

    def forward(self, text, h_state):
        embed = self._embed(text)
        #output:batch,seq_len, direction*hidden_size
        #h_state: layer*direction,batch,hidden_size
        output, h_state = self._lstm(embed, h_state)


        output = output[:, -1]  #batch,1, direction*hidden_size
        output = t.squeeze(output, 1) #batch, direction*hidden_size
        output = self._fc(output) # batch, tgt_size
        #print("forward end",output.shape)
        return output

    def init_hidden(self, batch, requires_grad=True):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            hidden = (weight.new(self._layer, batch, self._hidden_size).zero_().to(device),
                      weight.new(self._layer, batch, self._hidden_size).zero_().to(device))
            return hidden
        else:
            return t.zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad)

if __name__ == '__main__':
    custom_from_csv = DFCustomDataSet("data.csv",
                                    "/Users/piguanghua/Downloads/")
    dataset_loader = t.utils.data.DataLoader(dataset=custom_from_csv,
                                                 batch_size=2,
                                                 shuffle=False)

    for labels, text, length in dataset_loader:
        print(labels.shape)



    vocab_size, embed_size, hidden_size, tgt_size = len(words), 100, 250, 1
    model = TwoClassClassifier(vocab_size, embed_size, hidden_size, tgt_size)
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr)
    crit = nn.BCEWithLogitsLoss()
    model = model.to(device)
    crit = crit.to(device)

    epochs = 1
    counter = 0
    print_every = 100
    clip = 5
    valid_loss_min = np.Inf

    model.train()
    for i in range(epochs):
        h = None

        for labels, text, length in dataset_loader:
            # print(inputs.shape, labels.shape)
            counter += 1
            # h = tuple([e.data for e in h])
            inputs, labels = text.to(device), labels.to(device)

            output = model(text, h)

            loss = crit(output.squeeze(1), labels.float())

            if counter % 100 == 0:
                print("loss", loss.item())

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()


