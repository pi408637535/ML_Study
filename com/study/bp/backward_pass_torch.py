import torch as t
import numpy as np

if __name__ == '__main__':
    N,D_in,H,D_out = 64,1000,100,10

    x = t.randn(N, D_in)
    y = t.randn(N, D_out)

    w1 = t.randn(D_in, H)
    w2 = t.randn(H, D_out)

    lr = 1e-5

    for i in range(50):
        # forward pass
        h = x@(w1)
        h_relu = t.clamp(h, min=0)
        y_pred = h_relu@(w2)

        loss = (1/2 * t.pow((y_pred - y),2)).sum() .item()
        #loss = (1 / 2 * np.power((y_pred - y), 2)).sum()
        print(loss)

        y_grad = (y_pred - y)
        w2_grad = h_relu.t()@(y_grad)

        h_relu_grad = y_grad@(w2.t())
        h_relu[h_relu > 1e-4] = 1
        h_relu[h_relu < 0] = 0

        h_grad = h_relu_grad.clone()
        h_grad[h_grad > 1e-4] = 1
        h_grad[h_grad < 0] = 0
        w1_grad = x.t()@(h_grad)

        w1 -= lr * w1_grad
        w2 -= lr * w2_grad