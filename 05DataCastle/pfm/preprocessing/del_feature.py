'''
删除变量
'''
from util import dataset
import pandas as pd
import numpy as np

# 分类变量
cat_col = dataset.load('categorical', 'feature')
num_col = dataset.load('numeric', 'feature')
ord_col = dataset.load('order', 'feature')

# 加载数据
print('LOADING......')
train = dataset.load('train', 'all')
test = dataset.load('test', 'all')
y = dataset.load('target', 'train')

train.drop([
    'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsWithCurrManager', 'JobRole', 'StockOptionLevel', 'Gender',
    'DistanceFromHome', 'Education', 'PerformanceRating',
    'RelationshipSatisfaction', 'TrainingTimesLastYear'
], axis=1, inplace=True)
test.drop([
    'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsWithCurrManager', 'JobRole', 'StockOptionLevel', 'Gender',
    'DistanceFromHome', 'Education', 'PerformanceRating',
    'RelationshipSatisfaction', 'TrainingTimesLastYear'
], axis=1, inplace=True)

print(train.head())

new_cat = [x for x in train.columns if x in cat_col]
new_num = [x for x in train.columns if x in num_col]
new_ord = [x for x in train.columns if x in ord_col]

print('Saving data...')
dataset(numeric=new_num, categorical=new_cat, order=new_ord).save('feature')

dataset(train=train, test=test).save('all')

dataset(
    categorical=train[new_cat],
    numeric=train[new_num],
    order=train[new_ord]).save('train')
dataset(
    categorical=test[new_cat], numeric=test[new_num],
    order=test[new_ord]).save('test')

print('Done!')
