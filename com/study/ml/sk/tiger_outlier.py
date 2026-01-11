import pandas as pd
from sklearn import preprocessing
import numpy as np

# normalization = scale

if __name__ == '__main__':
    path = '/Users/piguanghua/Downloads/sample.xlsx'
    df = pd.read_excel(path)
    print('数据基本信息：')
    # sample_type =  df['sample_type']
    column_name = 'sample_type'
    stats = df[column_name].value_counts()
    important_feature = ['isSameCompany', 'isSameCity', 'isSameIp', 'isSameChannel',
                         'sameSymbolTradeTimes_1h', 'sameSymbolTradeTimes_1d', 'sameSymbolQuantity_1h',
                         'tradeTimes_1h', 'tradeTimes_1d', 'sameSymbolQuantityRate_1h', 'sameSymbolQuantityRate',
                         'orderTimeDiff', 'isSameOrderIp', 'isSameOrderDevice', 'symbolVolumeMove', 'symbolPriceMove']

    print(stats)


    #训练过程框架
    X,y = date,target
    X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size = 0.3)

    scores = cross_val_score(model, )
    #validation_curve