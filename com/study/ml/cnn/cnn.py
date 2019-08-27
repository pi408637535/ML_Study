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


file_path = '/Users/piguanghua/Downloads/GenPics'
BATCH_SIZE = 16
EPOCH = 10
device = t.device("cuda" if t.cuda.is_available() else "cpu")


# Load data
class dataset(Dataset):


    def __get_label(self,label_file):
        path = os.path.join(self.root_dir, "labels.csv")

        df = pd.read_csv(path, header=None)
        df.columns = ["name", "label"]
        df = df.set_index('name', True)
        return df

    def __init__(self, root_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.labels = self.__get_label(label_file)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, '%d.jpg' % idx)
        image = Image.open(img_name)

        #plt.imshow(image)
        #plt.show()

        label = self.labels.iloc[idx][0]

        #            sample = image

        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return (self.labels.shape[0])

data = dataset(file_path, "label.txt", transform=transforms.ToTensor())

dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

dataset_size = len(data)


# Conv network
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2,2),  #out 32@8*38
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2,2),  #16*3*18

        )
        self.out = nn.Linear(16 * 3 * 18, 40)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


# Train the net
class nCrossEntropyLoss(t.nn.Module):

    def __init__(self, n=4):
        super(nCrossEntropyLoss, self).__init__()
        self.n = n
        self.total_loss = 0
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, label):
        output_t = output[:, 0:10]
        label = t.tensor(t.LongTensor(label.data.cpu().numpy())).to(device)
        label_t = label[:, 0]

        for i in range(1, self.n):
            output_t = t.cat((output_t, output[:, 10 * i:10 * i + 10]),
                                 0)  # 损失的思路是将一张图平均剪切为4张小图即4个多分类，然后再用多分类交叉熵方损失
            label_t = t.cat((label_t, label[:, i]), 0)
            self.total_loss = self.loss(output_t, label_t)

        return self.total_loss


def equal(np1, np2):
    n = 0
    for i in range(np1.shape[0]):
        if (np1[i, :] == np2[i, :]).all():
            n += 1

    return n

net = CNN().to(device)
optimizer = t.optim.Adam(net.parameters(), lr=0.001)
#loss_func = nn.CrossEntropyLoss()
loss_func = nCrossEntropyLoss()

best_model_wts = copy.deepcopy(net.state_dict())
best_acc = 0.0

since = time.time()
for epoch in range(EPOCH):

    running_loss = 0.0
    running_corrects = 0

    for step, (inputs, label) in enumerate(dataloader):



        pred = t.LongTensor(BATCH_SIZE, 1).zero_()
        inputs = t.tensor(inputs).to(device)  # (bs, 3, 60, 240)
        label = t.tensor(label).to(device)  # (bs, 4)

        optimizer.zero_grad()

        output = net(inputs)  # (bs, 40)
        loss = loss_func(output, label)

        for i in range(4):
            pre = F.log_softmax(output[:, 10 * i:10 * i + 10], dim=1)  # (bs, 10)
            pred = t.cat((pred, pre.data.max(1, keepdim=True)[1].cpu()), dim=1)  #

        loss.backward()
        optimizer.step()

        running_loss += loss.data[0] * inputs.size()[0]
        running_corrects += equal(pred.numpy()[:, 1:], label.data.cpu().numpy().astype(int))

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(net.state_dict())

    if epoch == EPOCH - 1:
        t.save(best_model_wts, file_path + '/best_model_wts.pkl')

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Train Loss:{:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))