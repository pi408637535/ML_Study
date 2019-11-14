import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math

# L = 1/2 * pow((f(x) - y),2*)
def update_weight_MSE(W, X, b, Y, alpha):

    epoch = X.shape[0]
    loss_list = []
    for i in range(epoch):
        epoch_X = X[i,:].reshape((1,-1))
        loss = 1/2 * math.pow( (Y[i] - np.dot(W, epoch_X.T)  + b),2)
        delta_w = np.dot( (np.dot(W, epoch_X.T) - Y[i]), epoch_X )
        delta_b = (np.dot(W, epoch_X.T) - Y[i])

        W = W - alpha * delta_w
        b = b - alpha * delta_b
        loss_list.append(loss)

    return loss_list



if __name__ == '__main__':
    data_path = "/Users/piguanghua/Downloads/data.csv"

    df = pd.read_csv(data_path)

    print(df)
    print(math.pow(2,3))

    X, Y = df[['Year', 'Bus']], df["PGDP"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    ss_x = StandardScaler()
    X_train = ss_x.fit_transform(X_train)

    ss_y = StandardScaler()
    y_train = ss_y.fit_transform(y_train.values.reshape((-1, 1)))

    b = np.random.uniform(0, 1, 1)
    W = np.random.uniform(-1, 1, (1,2))

    alpha = 0.1
    loss_list = update_weight_MSE(W, X_train, b, y_train, alpha=alpha)
    plt.figure()
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()

