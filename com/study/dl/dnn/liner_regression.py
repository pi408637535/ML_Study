import torch as t
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

class LinerRegression(t.nn.Module):
    def __init__(self):
        super(LinerRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

if __name__ == '__main__':
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

    model = LinerRegression()
    criterion = nn.MSELoss()
    optimizer = t.optim.SGD(model.parameters(), lr = 1e-4)

    x_train = t.from_numpy(x_train)
    y_train = t.from_numpy(y_train)

    num_epochs = 1000
    for epoch in range(num_epochs):
        out = model(x_train)
        loss = criterion(y_train, out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch[{epoch + 1}/{num_epochs}], loss: {loss.item():.6f}')

    predict = model(x_train)
    plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
    plt.plot(x_train.numpy(), predict.data.numpy(), label='Fitting Line')


    #plt.scatter(x_train, y_train)
    plt.show()

