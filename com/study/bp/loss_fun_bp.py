import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
#均方误差
def update_weight_MSE(W, b, X, Y, learning_rate):
    loss_list = []
    for i in range(len(Y)):
        loss = np.dot(W,X.T[:,i])
        loss_list.append(loss)
        if loss < 0.5:
            print()

        W = W - learning_rate *(X.T[:,i].reshape((-1,1)).T)
        b = b - learning_rate * loss
        print("loss=%f" % loss)
    return W,b,loss_list

if __name__ == '__main__':
    data_path = "/Users/piguanghua/Downloads/data.csv"

    df = pd.read_csv(data_path)
    X, Y = df[['Year', 'Bus']], df["PGDP"]

    learning_rate = 0.01
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    ss_x = StandardScaler()
    X_train = ss_x.fit_transform(X_train)

    ss_y = StandardScaler()
    y_train = ss_y.fit_transform(y_train.values.reshape((-1, 1)))

    W = np.random.uniform(low=0.0, high=1.0, size=(1,2))
    b = np.random.random_integers(low=2,size=1)
    W,b,loss_list = update_weight_MSE(W, b, X_train, y_train, learning_rate)
    plt.figure()
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()





