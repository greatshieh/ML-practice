from util import dataset

data = dataset.load('categorical_dummy', 'train')
print(data.shape)

data = dataset.load('numeric', 'feature')
print(data)