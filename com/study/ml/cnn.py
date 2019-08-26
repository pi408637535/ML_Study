import torch as t
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import csv
import cv2



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=(1, 1))
        )  # 39*9

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1)
        )

        '''
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        '''
        self.fc1 = nn.Linear(624, 40)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

cnn = CNN()
print(cnn)
EPOCH = 10   #遍历数据集次数
BATCH_SIE = 8
LR = 0.001
device = t.device("cuda" if t.cuda.is_available() else "cpu")
net = CNN().to(device)

criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)


if __name__ == '__main__':
    for epoch in range(EPOCH):
        losses = []
        iters = int(math.ceil(train_x.shape[0] / batsize))
        for i in range(iters):
            train_x_i = train_x[i * batsize: (i + 1) * batsize]
            train_y_i = train_y[i * batsize: (i + 1) * batsize]
            tx = Variable(train_x_i)
            ty = Variable(train_y_i)
            out = cnn(tx)
            loss = loss_func(out, ty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.data.mean())