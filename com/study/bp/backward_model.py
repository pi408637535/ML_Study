import torch as t
import torch.nn as nn
import numpy as np


'''
    利用mode而没有利用optim
'''



if __name__ == '__main__':
    N,D_in,H,D_out = 64,1000,100,10

    model = nn.Sequential(
        nn.Linear(D_in, H),
        nn.ReLU(),
        nn.Linear(H, D_out)
    )
    nn.init.normal_(model[0].weight) #引入后效果更差了
    #nn.init.normal_(model[0].weight)
    #nn.init.normal_(model[2].weight)

    x = t.randn(N, D_in)
    y = t.randn(N, D_out)

    w1 = t.randn(D_in, H)
    w2 = t.randn(H, D_out)

    lr = 1e-3
    loss_fn = nn.MSELoss(reduction='sum')

    for i in range(500):
        # forward pass
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        #loss = (1/2 * t.pow((y_pred - y),2)).sum() .item()
        #loss = (1 / 2 * np.power((y_pred - y), 2)).sum()
        print(loss.item())
        loss.backward()

        with t.no_grad():
            for param in model.parameters():
                param -= lr * param.grad

        model.zero_grad()

