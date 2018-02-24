'''
连续变量进行对数转换
'''
from util import dataset
from sklearn.preprocessing import StandardScaler

print('Loading data......')
train = dataset.load('numeric', 'train').astype(float)
test = dataset.load('numeric', 'test').astype(float)
num_col = dataset.load('numeric', 'feature')

scaler = StandardScaler()
for col in num_col:
    scaler.fit(train[col].values.reshape(-1,1))
    train[col] = scaler.transform(train[col].values.reshape(-1,1))
    test[col] = scaler.transform(test[col].values.reshape(-1,1))

print(train.head())
print('='*20)
print(test.head())
print('='*20)

print('Saving data......')
dataset(numeric_stdscale=train).save('train')
dataset(numeric_stdscale=test).save('test')

print('Done!')
