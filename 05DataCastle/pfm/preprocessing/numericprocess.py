'''
对连续变量进行处理
'''
from util import dataset
from scipy import stats
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class NumProcess(object):
    def __init__(self, train, test):
        self.train = train.astype(float)
        self.test = test.astype(float)

    def boxcox(self):
        train = self.train.copy()
        test = self.test.copy()
        for col in train.columns:
            if stats.skew(train[col]) > 0.25:
                values, lam = stats.boxcox(train[col].values + 1)
                train[col] = values

            if stats.skew(test[col]) > 0.25:
                values, lam = stats.boxcox(test[col].values + 1)
                test[col] = values

        return train, test

    def log1p(self):
        train = self.train.copy()
        test = self.test.copy()

        for col in train.columns:
            train[col] = np.log1p(train[col])
            test[col] = np.log1p(test[col])

        return train, test

    def maxmin(self):
        scaler = MinMaxScaler()
        train = self.train.copy()
        test = self.test.copy()

        for col in train.columns:
            scaler.fit(train[col].values.reshape(-1, 1))
            train[col] = scaler.transform(train[col].values.reshape(-1, 1))
            test[col] = scaler.transform(test[col].values.reshape(-1, 1))

        return train, test

    def standar(self):
        scaler = StandardScaler()
        train = self.train.copy()
        test = self.test.copy()

        for col in train.columns:
            scaler.fit(train[col].values.reshape(-1, 1))
            train[col] = scaler.transform(train[col].values.reshape(-1, 1))
            test[col] = scaler.transform(test[col].values.reshape(-1, 1))

        return train, test

