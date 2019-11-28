import numpy as np
import math

if __name__ == '__main__':

    N, D_in, H, D_out = 64, 1000, 100, 10
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)

    lr = 1e-3

    for i in range(10):
        # forward pass
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        loss = (1 / 2 * np.power((y_pred - y), 2)).sum()
        print(loss)

        y_grad = (y_pred - y)
        w2_grad = h_relu.T.dot(y_grad)

        h_relu_grad = y_grad.dot(w2.T)
        h_relu[h_relu > 1e-4] = 1
        h_relu[h_relu < 0] = 0

        h_grad = h_relu_grad.copy()
        h_grad[h_grad > 1e-4] = 1
        h_grad[h_grad < 0] = 0
        w1_grad = x.T.dot(h_grad)

        w1 -= lr * w1_grad
        w2 -= lr * w2_grad

        # print(w1)
        # print(w2)





