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


file_path = '/root/data/GenPics'
BATCH_SIZE = 2
EPOCH = 10
device = t.device("cuda" if t.cuda.is_available() else "cpu")

'''
参考博客：https://www.cnblogs.com/king-lps/p/8724361.html
https://www.jianshu.com/p/08e9d2669b42

'''


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
class nCrossEntropyLoss(nn.Module):

    def __init__(self, n=4):
        super(nCrossEntropyLoss, self).__init__()
        self.n = n
        self.total_loss = 0
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, label):

        batch_label = label.view(-1,1)
        batch_output = output.view(-1, 4, 10)
        self.total_loss = 0


        for i in range(BATCH_SIZE):
            for batch_label_item in batch_label[i]:
                label_batch_item = []

                str_element = str(batch_label_item.numpy().tolist())
                for label_item in range(len(str_element)):
                    label_batch_item.append(str_element[label_item])

                label_batch_item = list(map(int, label_batch_item))
                self.total_loss += self.loss(batch_output[i], t.LongTensor(label_batch_item))
        return self.total_loss


        '''
        output_t = output[:, 0:10]
        label = t.tensor(t.LongTensor(label.data.cpu().numpy())).to(device)
        label_t = label[:, 0]

        for i in range(1, self.n):
            output_t = t.cat((output_t, output[:, 10 * i:10 * i + 10]),0)  # 损失的思路是将一张图平均剪切为4张小图即4个多分类，然后再用多分类交叉熵方损失
            label_t = t.cat((label_t, label[:, i]), 0)
            self.total_loss = self.loss(output_t, label_t)

        return self.total_loss
        '''


def equal(np1, np2):
    n = 0
    for i in range(np1.shape[0]):
        if (np1[i, :] == np2[i, :]).all():
            n += 1

    return n



def train():
    net = CNN().to(device)

    optimizer = t.optim.SGD(net.parameters(), lr=0.001)
    # loss_func = nn.CrossEntropyLoss()
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
            print(label)
            loss = loss_func(output, label)

            '''
            for i in range(4):
                pre = F.log_softmax(output[:, 10 * i:10 * i + 10], dim=1)  # (bs, 10)
                pred = t.cat((pred, pre.data.max(1, keepdim=True)[1].cpu()), dim=1)  #
            '''
            loss.backward(retain_graph=True)
            optimizer.step()

            print("loss=", loss.data.numpy().tolist())
            running_loss += loss.data.numpy().tolist()
            # running_corrects += equal(pred.numpy()[:, 1:], label.data.cpu().numpy().astype(int))

        epoch_loss = running_loss / dataset_size
        # epoch_acc = running_corrects / dataset_size

        '''
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(net.state_dict())
        '''
        if epoch == EPOCH - 1:
            t.save(net.state_dict(), '/root/data/model/cnn.pkl')

        print("end")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Train Loss:{:.4f} Acc: {:.4f}'.format(epoch_loss, 4.5))


def test():
    net = CNN().to(device)
    net.load_state_dict(t.load('/root/data/model/cnn.pkl'))
    num = 0

    for step, (inputs, label) in enumerate(dataloader):
        inputs = t.tensor(inputs).to(device)
        label = t.tensor(label).to(device)
        output = net(inputs)

        for i in range(BATCH_SIZE):
            batch_item = output[i]
            c0 = np.argmax(batch_item.data.numpy()[0:10])
            c1 = np.argmax(batch_item.data.numpy()[10:20])
            c2 = np.argmax(batch_item.data.numpy()[20:30])
            c3 = np.argmax(batch_item.data.numpy()[30:40])
            c = '%s%s%s%s' % (c0, c1, c2, c3)
            print(c,label[i])

        num += 1
        if num > 5:
            break


if __name__ == '__main__':
    test()
