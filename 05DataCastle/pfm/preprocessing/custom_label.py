'''
自定义分类变量数值化
'''
from util import dataset
import pandas as pd
import numpy as np

# 分类变量
cat_col = dataset.load('categorical', 'feature')

# 加载数据
print('LOADING......')
train = dataset.load('categorical', 'train')
test = dataset.load('categorical', 'test')
y= dataset.load('target', 'train')

train = pd.concat([train, y], axis=1)

temp = pd.DataFrame({
    'Attriton':
    train.groupby('BusinessTravel')['Attrition'].mean().sort_values(),
    'ranking':
    np.arange(1, 4)
})
train['BusinessTravel'] = train['BusinessTravel'].map(
    lambda x: temp.loc[x, 'ranking'])
test['BusinessTravel'] = test['BusinessTravel'].map(
    lambda x: temp.loc[x, 'ranking'])

temp = pd.DataFrame({
    'Attriton':
    train.groupby('Department')['Attrition'].mean().sort_values(),
    'ranking':
    np.arange(1, 4)
})
train['Department'] = train['Department'].map(
    lambda x: temp.loc[x, 'ranking'])
test['Department'] = test['Department'].map(
    lambda x: temp.loc[x, 'ranking'])

temp = pd.DataFrame({
    'Attriton':
    train.groupby('EducationField')['Attrition'].mean().sort_values(),
    'ranking':
    np.arange(1, 7)
})

train['EducationField'] = train['EducationField'].map(
    lambda x: temp.loc[x, 'ranking'])
test['EducationField'] = test['EducationField'].map(
    lambda x: temp.loc[x, 'ranking'])

temp = pd.DataFrame({
    'Attriton':
    train.groupby('JobRole')['Attrition'].mean().sort_values(),
    'ranking':
    np.arange(1, 10)
})
train['JobRole'] = train['JobRole'].map(lambda x: temp.loc[x, 'ranking'])
test['JobRole'] = test['JobRole'].map(lambda x: temp.loc[x, 'ranking'])

temp = pd.DataFrame({
    'Attriton':
    train.groupby('MaritalStatus')['Attrition'].mean().sort_values(),
    'ranking':
    np.arange(1, 4)
})
train['MaritalStatus'] = train['MaritalStatus'].map(
    lambda x: temp.loc[x, 'ranking'])
test['MaritalStatus'] = test['MaritalStatus'].map(
    lambda x: temp.loc[x, 'ranking'])

train.drop(['Attrition'], axis=1, inplace=True)

print(train.head())

print('Saving data......')
dataset(custom_label=train).save('train')
dataset(custom_label=test).save('test')

print('Done!')
