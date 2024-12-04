
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

# function to create a list containing mini-batches
def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0

    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches

class MLP():
    def __init__(self, w_shape):
        self.w_shape = w_shape
        self.w = np.random.randn(w_shape,1)
        self.b = np.random.rand(1)
        self.alpha = 0.01
        self.cache_x = None

    def forward(self, x):
        x = x.reahpe(x.shape[0], self.w_shape)
        estimate_y = np.dot(self.w.T, x.T) + self.b
        self.cache_x =  x.T
        return estimate_y

    def bp(self, y, estimate_y):
        self.w = self.w - self.alpha * (y - estimate_y) * (-1) * self.cache_x
        self.b = self.b - self.alpha * (y - estimate_y) * (-1)

def loss_fun(estimate_y, y):
    loss = 1/2 * np.pow( estimate_y - y )
    return loss



if __name__ == '__main__':
    mean = np.array([5.0, 6.0])
    cov = np.array([[1.0, 0.95], [0.95, 1.2]])
    data = np.random.multivariate_normal(mean, cov, 8000)
    # visualising data

    # plt.scatter(data[:500, 0], data[:500, 1], marker='.')
    # plt.show()

    split_factor = 0.90
    split = int(split_factor * data.shape[0])
    X_train = data[:split, :-1]
    y_train = data[:split, -1].reshape((-1, 1))
    X_test = data[split:, :-1]
    y_test = data[split:, -1].reshape((-1, 1))

    batch_size = 32
    mini_batches = create_mini_batches(X_train, y_train, batch_size)
    mini_batches
        
