import torch as t
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import torchtext
import random

USE_CUDA = t.cuda.is_available()

#lstm完工
class LSTMModel(nn.Module):
    def __init__(self,  vocab_size, embed_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

        self.rnn_type = 'LSTM'
        self._layer = 1
        self._hidden_size = hidden_size

    #hidden (h_0, c_0)  h_0：batch, layer*direction, hidden_size
    #text batch,seq_len,input_size
    def forward(self, text, hidden):
        embedding = self.embedding(text)

        #embedding: batch,seq_len, embed_size
        #output: batch,seq_len,dicection*hidden_size
        #hidden:(h_0, c_0) batch,layer*diection, hidden_size
        output, hidden = self.lstm(embedding, hidden)

        output = output.contiguous().view(output.shape[0] * output.shape[1], output.shape[2])
        #out: seq_len * vocab_size
        out = self.out(output)
        return out.view(output.shape[0], output.shape[1], output.shape[1]),hidden


    def init_hidden(self, batch, requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad),
                    weight.new_zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad))
        else:
            return weight.new_zeros((batch, self._layer, self._hidden_size), requires_grad=requires_grad)

BATCH_SIZE = 32
EMBEDDING_SIZE = 100
MAX_VOCAB_SIZE = 50000

def repackage_hidden(h):
    if isinstance(h, t.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


if __name__ == '__main__':
    #保证实验结果可复现，需要把各种random seed固定在某只
    random.seed(0)
    np.random.seed(0)
    t.manual_seed(0)
    if USE_CUDA:
        t.cuda.random.seed(0)

    BATCH_SIZE = 32
    EMBEDDING_SIZE = 100
    MAX_VOCAB_SIZE = 50000

    #TorchText中的Field,它决定了数据如何处理
    TEXT = torchtext.data.Field(lower=True)
    path = "/Users/piguanghua/Downloads/text8"
    #LanguageModelingDataset 处理语言模型的数据集
    train,val,test = torchtext.datasets.LanguageModelingDataset.splits(
        path=path,
        train="text8.train.txt",
        validation="text8.dev.txt",
        test="text8.test.txt",
        text_field = TEXT
    )
    #构造 term-num 结构
    #build_vocab创建词频单词表， max_size限定单词总数
    TEXT.build_vocab(train, max_size = MAX_VOCAB_SIZE)
    #print(len(TEXT.vocab))
    #print(TEXT.vocab.itos[:10])
    #print(TEXT.vocab.stoi["and"])

    device = t.device("cuda" if USE_CUDA else "cpu" )

    '''
        BPTTIterator连续得到连贯的句子。
        BPTT back propagetion through time
        bptt_len bp回传长度
        repeat单文档不会重复
    '''
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator\
        .splits( (train, val, test), batch_size=BATCH_SIZE,
                 device=device, bptt_len = 50, repeat = False,
                 shuffle = True)

    it = iter(test_iter)
    batch = next(it)
    #print(batch)
    #batch 本身就两个维度
    #print(batch.text)
    #print(batch.target)
    #print(" ".join([ TEXT.vocab.itos[i] for i in batch.text[:, 0].data.cpu()] ))
    #print(" ".join([TEXT.vocab.itos[i] for i in batch.target[:, 0].data.cpu()]))

    VOCAB_SIZE = len(TEXT.vocab)
    model = LSTMModel(VOCAB_SIZE, EMBEDDING_SIZE, BATCH_SIZE )
    if USE_CUDA:
        model = model.cuda()
    optimizer = t.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        #hidden = model.init_hidden(BATCH_SIZE)
        if USE_CUDA:
            data, target = batch.text, batch.target
        data.t_()
        target.t_()
        repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())



