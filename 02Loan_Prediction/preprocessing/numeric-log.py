'''
连续变量进行对数转换
'''
from util import dataset
import pandas as pd
import numpy as np

print('Loading data......')
train = dataset.load('numeric', 'train')
test = dataset.load('numeric', 'test')
num_col = dataset.load('numeric', 'feature')

for col in num_col:
    train[col] = np.log1p(train[col])
    test[col] = np.log1p(test[col])

print(train.head())
print('='*20)
print(test.head())
print('='*20)

print('Saving data......')
dataset(numeric_log1p=train).save('train')
dataset(numeric_log1p=test).save('test')

print('Done!')
