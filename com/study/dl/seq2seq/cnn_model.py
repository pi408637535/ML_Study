# -*- coding: utf-8 -*-
# @Time    : 2020/5/12 18:01
# @Author  : piguanghua
# @FileName: cnn_model.py
# @Software: PyCharm

class LetNet(nn.Module):
    def __init__(self):
        super(LetNet, self).__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.s2 = nn.MaxPool2d(6, 6, kernel_size=14)
        self.c3 = nn.Conv2d(6, 16, kernel_size=)


    def forward(self, picture):
        pass
