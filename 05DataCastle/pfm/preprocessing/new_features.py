'''
分类变量创建虚拟变量
'''
from util import dataset
import pandas as pd
from numericprocess import NumProcess

print('Loading data......')

cat_var = dataset.load('categorical', 'feature')
num_var = dataset.load('numeric', 'feature')

train = pd.concat(
    [dataset.load('numeric', 'train'),
     dataset.load('categorical', 'train')],
    axis=1)
test = pd.concat(
    [dataset.load('numeric', 'test'),
     dataset.load('categorical', 'test')],
    axis=1)

train['source'] = 'train'
test['source'] = 'test'

df = pd.concat([train, test], axis=0)

for x in range(len(cat_var)):
    for y in range(len(cat_var)):
        if x < y:
            name = '{}&{}'.format(cat_var[x], cat_var[y])
            df[name] = df[cat_var[x]].astype(
                str) + '&' + df[cat_var[y]].astype(str)

cat_var = [x for x in df.columns if x not in num_var and x != 'source']

for col in cat_var:
    dummies = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummies], axis=1)
    df.drop([col], axis=1, inplace=True)

cat_var = [x for x in df.columns if x not in num_var and x != 'source']

for x in [
        a for a in num_var
        if a not in
    ['MonthlyIncome', 'PercentSalaryHike', 'NumCompaniesWorked']
]:
    for y in [
            a for a in num_var
            if a not in
        ['MonthlyIncome', 'PercentSalaryHike', 'NumCompaniesWorked']
    ]:
        if x != y:
            name = '{}-{}'.format(x, y)
            df[name] = df[x] - df[y]

num_var = [x for x in df.columns if x not in cat_var]

print(num_var)

print(cat_var)

# df['WorkingYearsBefore'] = df['TotalWorkingYears'] - df['YearsAtCompany']

# def average_years(average_years):
#     x, y = average_years
#     if y == 0:
#         return 0
#     else:
#         return x / y

# df['AverageWorkingYears'] = df[['WorkingYearsBefore',
#                                 'NumCompaniesWorked']].apply(
#                                     average_years, axis=1)

X_train = df[df['source'] == 'train'].copy()
X_test = df[df['source'] == 'test'].copy()

X_train.drop(['source'], axis=1, inplace=True)
X_test.drop(['source'], axis=1, inplace=True)

dataset(new_feature=X_train).save('train')
dataset(new_feature=X_test).save('test')

# dataset(new_numeric=X_train).save('train')
# dataset(new_numeric=X_test).save('test')

# process = NumProcess(X_train, X_test)
# train, test = process.boxcox()
# dataset(new_numeric_boxcox=train).save('train')
# dataset(new_numeric_boxcox=test).save('test')

# train, test = process.log1p()
# dataset(new_numeric_log1p=train).save('train')
# dataset(new_numeric_log1p=test).save('test')

# train, test = process.maxmin()
# dataset(new_numeric_maxmin=train).save('train')
# dataset(new_numeric_maxmin=test).save('test')

# train, test = process.standar()
# dataset(new_numeric_stand=train).save('train')
# dataset(new_numeric_stand=test).save('test')

print('Done!')
