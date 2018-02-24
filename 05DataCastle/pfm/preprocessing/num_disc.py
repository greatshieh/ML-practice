'''
连续变量进行离散化
'''
from util import dataset
import pandas as pd
import numpy as np
import chimerge


def meger(x, inver_list):
    for i in range(len(inver_list)):
        if x <= inver_list[i]:
            return i + 1
    return i + 2


print('Loading data......')
train = dataset.load('numeric', 'train')
test = dataset.load('numeric', 'test')
num_col = dataset.load('numeric', 'feature')

target = dataset.load('target', 'train')

df = pd.concat([train, target], axis=1)

for col in num_col:
    _, interval_list = chimerge.ChiMerge(df, col, 'Attrition')
    train[col] = train[col].map(lambda x: meger(x, interval_list))
    test[col] = test[col].map(lambda x: meger(x, interval_list))

print(train.head())
print('=' * 20)
print(test.head())
print('=' * 20)

print('Saving data......')
dataset(numeric_disc=train).save('train')
dataset(numeric_disc=test).save('test')

train['source'] = 'train'
test['source'] = 'test'

df = pd.concat([train, test], axis=0)


for col in num_col:
    dummies = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummies], axis=1)
    df.drop([col], axis=1, inplace=True)

train = df[df['source']=='train'].copy()
test = df[df['source']=='test'].copy()

train.drop(['source'], axis=1, inplace=True)
test.drop(['source'], axis=1, inplace=True)

dataset(numeric_disc_dummy=train).save('train')
dataset(numeric_disc_dummy=test).save('test')

print('Done!')
