from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt


class MLP():
    def __init__(self, w_shape):
        self.w_shape = w_shape
        self.w = np.random.randn(w_shape,1)
        self.b = np.random.rand(1)
        self.alpha = 0.01
        self.cache_x = None

    def forward(self, x):
        estimate_y = np.dot(x, self.w) + self.b
        self.cache_x = x
        return estimate_y

    def bp(self, y, estimate_y):
        self.w = self.w - self.alpha * np.dot(self.cache_x.T, (y - estimate_y))
        self.b = self.b - self.alpha * np.sum(y - estimate_y) / y.shape[0]

def loss_function(estimate_y, y):
    batch = y.shape[0]
    loss = 0.5 * np.dot((y-estimate_y).T, (y-estimate_y)  )
    return loss / batch

if __name__ == '__main__':

    diabetes = load_diabetes()
    data = diabetes.data
    target = diabetes.target

    # 打乱数据
    X, y = shuffle(data, target, random_state=13)
    X = X.astype(np.float32)

    # 训练集与测试集的简单划分
    offset = int(X.shape[0] * 0.9)

    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    model = MLP(X.shape[1])
    epoch = 400
    loss_list = []
    for i in range(epoch):
       estimate_y = model.forward(X_train)
       loss = loss_function(estimate_y, y_train)
       loss_list.append(loss.tolist()[0])
       model.bp(y_train, estimate_y)

       print(loss)


    x = np.linspace(0, len(loss_list), num=len(loss_list))
    plt.plot( x,loss_list[::-1], color='blue')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()