import numpy as np

def linear_loss(X, y, w, b):
    num_train = X.shape[0]
    num_feature = X.shape[1]    
    # 模型公式
    y_hat = np.dot(X, w) + b    
    # 损失函数
    loss = np.sum((y_hat-y)**2)/num_train    
    # 参数的偏导
    dw = np.dot(X.T, (y_hat-y)) /num_train
    db = np.sum((y_hat-y)) /num_train    
    return y_hat, loss, dw, db

def initialize_params(dims):
    w = np.zeros((dims, 1))
    b = 0
    return w, b


def linar_train(X, y, learning_rate, epochs):
    w, b = initialize_params(X.shape[1])
    loss_list = []
    for i in range(1, epochs):
        # 计算当前预测值、损失和参数偏导
        y_hat, loss, dw, db = linear_loss(X, y, w, b)
        loss_list.append(loss)
        # 基于梯度下降的参数更新过程
        w += -learning_rate * dw
        b += -learning_rate * db
        # 打印迭代次数和损失

        if i % 10000 == 0:
            print('epoch %d loss %f' % (i, loss))

            # 保存参数
        params = {
            'w': w,
            'b': b
        }

        # 保存梯度
        grads = {
            'dw': dw,
            'db': db
        }

    return loss_list, loss, params, grads

from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle

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
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

print('X_train=', X_train.shape)
print('X_test=', X_test.shape)
print('y_train=', y_train.shape)
print('y_test=', y_test.shape)

loss_list, loss, params, grads = linar_train(X_train, y_train, 0.001, 100000)