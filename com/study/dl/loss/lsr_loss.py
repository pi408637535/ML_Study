# -*- coding: utf-8 -*-
# @Time    : 2020/8/7 14:33
# @Author  : piguanghua
# @FileName: lsr_loss.py
# @Software: PyCharm

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SmoothLabelCritierion(nn.Module):
    """
    TODO:
    1. Add label smoothing
    2. Calculate loss
    """

    def __init__(self, label_smoothing=0.0):
        super(SmoothLabelCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax()

        # When label smoothing is turned on, KL-divergence is minimized
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction='batchmean')
        else:
            self.criterion = nn.NLLLoss()
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):

        one_hot = t.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot



    def forward(self, dec_outs, labels):
        # Map the output to (0, 1)
        scores = F.log_softmax(dec_outs, dim = -1)
        # n_class
        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            gtruth = tmp_.detach()
        loss = self.criterion(scores, gtruth)
        return loss

class CrossEntropyLoss_LSR(nn.Module):
    def __init__(self, device, para_LSR=0.2):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.device = device
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def _toOneHot_smooth(self, label, batchsize, classes):
        prob = self.para_LSR * 1.0 / classes
        one_hot_label = t.zeros(batchsize, classes) + prob
        for i in range(batchsize):
            index = label[i]
            one_hot_label[i, index] += (1.0 - self.para_LSR)
        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)
        loss = t.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
        if size_average:
            return t.mean(loss)
        else:
            return t.sum(loss)

if __name__ == "__main__":
    LSR = SmoothLabelCritierion(label_smoothing=0.1)
    y_pred = t.randn(size=[8, 3], requires_grad=True)
    y_true = t.randint(low=0, high=2, size=(8, 1))

    loss = LSR(y_pred, y_true)
    print(loss)

    device = t.device('cuda' if False else 'cpu')
    lsr_model = CrossEntropyLoss_LSR(device, 0.1)
    loss = lsr_model(y_pred, y_true)
    print(loss)

