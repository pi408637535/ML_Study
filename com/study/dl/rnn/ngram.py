import torch as t
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
from torch.utils.data import Dataset, DataLoader

#define
TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

line_str = "Steam-filled and tranquil, public baths in Japan have been a haven from the stresses of daily life for more than 1,000 years. While the natural hot springs known as onsen are familiar worldwide, and can be private or public, there are also the lesser-known sento – public baths relying on regular, filtered water. Found in almost every neighbourhood and requiring complete nudity, both types of communal bathhouses have a set of strict rules on washing etiquette before entering the pristine, soap-free waters and offer a space for friends, families and even co-workers to relax and connect. These days, nearly every home in Japan has a deep-set tub perfect for a private soak and the popularity of a public dip is waning, but nowhere near as much as you might expect."
line_list = line_str.split(" ")
line_set = set(line_list)
WORD_NUM = len(line_set)
WORD_DIM = 100


index = 0
vocab = {}
for index, word in enumerate(line_set):
    vocab[word] = vocab.get(word, 0 ) + index
str2id = {k:v for k,v in vocab.items()}
id2str = {v:k for k,v in vocab.items()}

line_num = [vocab[item] for item in line_list]


word_paid = [((line_num[i], line_num[i+1]),line_num[i+2]) for i in range(len(line_num) - 2)]


class CustomDataSet(Dataset):
    def __init__(self, transform=None):
        self._data = self._load_dataset(word_paid)
        self._transform = transform

    def _load_dataset(self, word_paid):
        all_data = []
        for item in word_paid:
            data = {
                #'source': np.array(item[0]),
                'source': np.array(item[0]),

                'target': item[1]
            }
            all_data.append(data)
        return all_data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


class Ngram(nn.Module):
    def __init__(self, word_size, embedding_dim, hidden_size, output_size):
        super(Ngram, self).__init__()
        self._embed = nn.Embedding(word_size,embedding_dim)
        self._rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self._out = nn.Linear(hidden_size, output_size)

    '''
        x: batch,seq_len,input_size:batch,2,1
        h_state:batch,layer*direction, h_size
    '''
    def forward(self, x):
        embed = self._embed(x) #embed batch,seq,embed_size
        output, h_state = self._rnn(embed)
        h_n = t.squeeze(h_state[0].transpose(0,1).contiguous())
        return self._out(h_n) #batch,seq,output_size

class NgramBook(nn.Module):
    def __init__(self, word_size, embedding_dim, hidden_size, output_size):
        super(NgramBook, self).__init__()
        self._embed = nn.Embedding(word_size,embedding_dim)
        self._liner1 = nn.Linear( 2 * embedding_dim, 128)
        self._liner2 = nn.Linear(128, output_size)

    '''
        x: batch,seq_len,input_size:batch,2,1
        h_state:batch,layer*direction, h_size
    '''
    def forward(self, x):
        batch = x.shape[0]
        embed = self._embed(x) #embed batch,seq,embed_size
        embed = embed.view(batch, 1, -1)
        embed = t.squeeze(embed)
        out = self._liner1(embed)
        out = F.relu(out)
        out = self._liner2(out)
        prob = F.log_softmax(out)
        #prob = t.squeeze(prob)
        return prob


if __name__ == '__main__':
    custom_dataset = CustomDataSet()
    dataset_loader = t.utils.data.DataLoader(dataset=custom_dataset,
                                             batch_size=4,
                                             shuffle=False)

    word_size, embedding_dim, hidden_size, output_size = WORD_NUM, WORD_DIM, 100, len(line_set)
   # model = NgramBook(word_size, embedding_dim, hidden_size, output_size)
    model = Ngram(word_size, embedding_dim, hidden_size, output_size)
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for batch_id, batch_data in enumerate(dataset_loader):
            targets = batch_data['target']
            inputs = batch_data['source']
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            avg_loss = loss.item() / batch_data['target'].shape[0]
            print(batch_id, avg_loss)

   # model.eval()
    outputs = t.LongTensor(np.array([64, 80])).view(1,-1)

    outputs = t.max(model(outputs).view(1,-1), 1)[1]
    print(id2str[outputs.item()])