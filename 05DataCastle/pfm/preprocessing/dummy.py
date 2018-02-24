'''
为分类变量和有序变量创建虚拟变量
'''
from util import dataset
import pandas as pd

# 有序变量
ord_col = dataset.load('order', 'feature')

# 分类变量
cat_col = dataset.load('categorical', 'feature')

# 加载数据
print('LOADING......')
train = dataset.load('categorical', 'train')
test = dataset.load('categorical', 'test')

train['source'] = 'train'
test['source'] = 'test'
df = pd.concat([train, test], axis=0)

for col in cat_col:
    if len(df[col].unique()) > 2:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
        df.drop([col], axis=1, inplace=True)

train = df[df['source']=='train'].copy()
test = df[df['source']=='test'].copy()

train.drop(['source'], axis=1, inplace=True)
test.drop(['source'], axis=1, inplace=True)

print('Saving data......')
dataset(categorical_dummy=train).save('train')
dataset(categorical_dummy=test).save('test')

##############
# ##有序变量
##############

# 加载数据
print('LOADING......')
train = dataset.load('order', 'train')
test = dataset.load('order', 'test')

train['source'] = 'train'
test['source'] = 'test'
df = pd.concat([train, test], axis=0)

for col in ord_col:
    if len(df[col].unique()) > 2:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
        df.drop([col], axis=1, inplace=True)

train = df[df['source']=='train'].copy()
test = df[df['source']=='test'].copy()

train.drop(['source'], axis=1, inplace=True)
test.drop(['source'], axis=1, inplace=True)

print('Saving data......')
dataset(order_dummy=train).save('train')
dataset(order_dummy=test).save('test')

print('Done!')
