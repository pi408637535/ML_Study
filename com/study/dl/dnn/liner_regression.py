import torch as t
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



class LinerRegression(t.nn.Module):
    def __init__(self):
        super(LinerRegression, self).__init__()
        self.linear = nn.Linear(15, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

if __name__ == '__main__':

    path_train = "/Users/piguanghua/Downloads/house-prices-advanced-regression-techniques/train.csv"
    columns = ['MSSubClass', 'LotFrontage', 'LotArea', "OverallQual", "OverallCond", "YearBuilt",
               "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "TotalBsmtSF", "GrLivArea", "TotRmsAbvGrd",
               "GarageYrBlt", "GarageArea", "YrSold", "SalePrice"]

    data = pd.read_csv(path_train, usecols=columns)
    data["LotFrontage"] = data["LotFrontage"].fillna(np.mean(data["LotFrontage"]))
    data["GarageYrBlt"] = data["GarageYrBlt"].fillna(np.mean(data["GarageYrBlt"]))
    data = data.dropna()

    X = data.iloc[:, :-1].values
    y = data['SalePrice'].values  # 房间价格
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    bins = int(1 + 4 * math.log(len(y)))
    plt.hist(y, bins=bins, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("interval")
    # 显示纵轴标签
    plt.ylabel("SalePrice")
    # 显示图标题
    # plt.title("")
    plt.show()

    '''
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
    '''
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape((-1, 1)))

    x_train = X
    y_train = y


    model = LinerRegression()
    criterion = nn.MSELoss()
    optimizer = t.optim.SGD(model.parameters(), lr = 1e-4)

    x_train = t.FloatTensor(x_train)
    y_train = t.FloatTensor(y_train)

    num_epochs = 1000
    for epoch in range(num_epochs):
        out = model(x_train)
        loss = criterion(y_train, out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch[{epoch + 1}/{num_epochs}], loss: {loss.item():.6f}')





