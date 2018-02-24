'''
分类变量创建虚拟变量
'''
from util import dataset
import pandas as pd

print('Loading data......')
train = dataset.load('categorical', 'train')
test = dataset.load('categorical', 'test')
cat_col = dataset.load('categorical', 'feature')

for col in cat_col:
    dummies = pd.get_dummies(train[col], prefix=col)
    train = pd.concat([train, dummies], axis=1)
    train.drop([col], axis=1, inplace=True)

    dummies = pd.get_dummies(test[col], prefix=col)
    test = pd.concat([test, dummies], axis=1)
    test.drop([col], axis=1, inplace=True)

print('Saving data......')
dataset(categorical_dummy=train).save('train')
dataset(categorical_dummy=test).save('test')

print('Done!')
