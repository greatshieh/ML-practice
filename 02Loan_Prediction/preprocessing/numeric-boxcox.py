'''
连续变量进行对数转换
'''
from util import dataset
from scipy import stats

print('Loading data......')
train = dataset.load('numeric', 'train').astype(float)
test = dataset.load('numeric', 'test').astype(float)
num_col = dataset.load('numeric', 'feature')

for col in num_col:
    if stats.skew(train[col]) > 0.25:
        values, lam = stats.boxcox(train[col].values+1)
        train[col] = values
        print(col)

    if stats.skew(test[col]) > 0.25:
        values, lam = stats.boxcox(test[col].values+1)
        test[col] = values

print(train.head())
print('='*20)
print(test.head())
print('='*20)

print('Saving data......')
dataset(numeric_boxcox=train).save('train')
dataset(numeric_boxcox=test).save('test')

print('Done!')
