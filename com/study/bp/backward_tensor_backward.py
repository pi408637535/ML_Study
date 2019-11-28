import torch as t
import numpy as np

if __name__ == '__main__':
    N,D_in,H,D_out = 64,1000,100,10

    x = t.randn(N, D_in, requires_grad=True)
    y = t.randn(N, D_out, requires_grad=True)

    w1 = t.randn(D_in, H, requires_grad=True)
    w2 = t.randn(H, D_out, requires_grad=True)

    lr = 1e-6

    for i in range(500):
        # forward pass
        h = x@(w1)
        h_relu = t.clamp(h, min=0)
        y_pred = h_relu@(w2)

        loss = (1/2 * t.pow((y_pred - y),2)).sum()
        #loss = (1 / 2 * np.power((y_pred - y), 2)).sum()
        print(loss.item())

        loss.backward()

        with t.no_grad():
            w1 -= lr * w1.grad
            w2 -= lr * w2.grad
            w1.grad.zero_()
            w2.grad.zero_()

