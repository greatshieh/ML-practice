'''
处理缺失值, 并将分类变量数值化
将处理后的数据按照分类变量和连续变量分开保存
'''
from util import dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


# 定义变量类型
# 目标变量
target_var = 'Loan_Status'

# 分类变量
cat_col = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'Credit_History', 'Property_Area'
]

# 连续变量
num_col = [
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'
]

# 加载数据
print('LOADING......')
train = pd.read_csv('./dataset/train.csv', header=0, index_col='Loan_ID')
test = pd.read_csv('./dataset/test.csv', header=0, index_col='Loan_ID')

train['source'] = 'train'
test['source'] = 'test'

df = pd.concat([train, test], axis=0)

# 填充缺失值
print('Imputing......')
df.loc[((df.Gender == "Male") & (df.Married.isnull())), "Married"] = "Yes"
df.loc[((df.Gender == "Female") & (df.Married.isnull())), "Married"] = "No"

df.loc[((df.Married == "Yes") & (df.Gender.isnull())), "Gender"] = "Male"
df.loc[((df.Married == "No") & (df.Gender.isnull())), "Gender"] = "Female"

df.Credit_History.fillna(1.0, inplace=True)
df["Dependents"] = df.groupby([
    "Married", "Gender", "Property_Area"
]).Dependents.transform(lambda x: x.fillna(x.value_counts().idxmax()))

df["LoanAmount"] = df.groupby("Education").LoanAmount.transform(
    lambda x: x.fillna(x.mean()))
df["Loan_Amount_Term"] = df.groupby("Married").Loan_Amount_Term.transform(
    lambda x: x.fillna(x.mean()))

train = df.dropna(axis=0)
test = df[df.Self_Employed.isnull()]

model = LinearDiscriminantAnalysis()
model.fit(train[["LoanAmount", "Loan_Amount_Term"]], train["Self_Employed"])
missing_se = model.predict(test[["LoanAmount", "Loan_Amount_Term"]])

df.loc[df.Self_Employed.isnull(), "Self_Employed"] = missing_se

df["Loan_Amount_Term"] = df.groupby("Married").Loan_Amount_Term.transform(
    lambda x: x.fillna(x.mean()))

df["inc/lAmount"] = df['ApplicantIncome'] / df['LoanAmount']
df["allIncome"] = df['ApplicantIncome'] + df['CoapplicantIncome']
df["netIncome"] = df["allIncome"] - (
    df['LoanAmount'] / df['Loan_Amount_Term'])

train = df[df['source'] == 'train'].copy()
train.drop(['source'], axis=1, inplace=True)

test = df[df['source'] == 'test'].copy()
test.drop(['source', target_var], axis=1, inplace=True)

# 检查缺失值
print('=' * 20)
print(train.isnull().sum())
print('=' * 20)
print(test.isnull().sum())
print('=' * 20)

# 将分类变量转化为数值
label_enc = LabelEncoder()
for x in [col for col in cat_col if train.dtypes[col] == 'object']:
    label_enc.fit(train[x])
    train[x] = label_enc.transform(train[x])
    test[x] = label_enc.transform(test[x])

num_col = [x for x in test.columns if x not in cat_col]

# 将数据保存为pickle文件
print('Saving feature name')
dataset(numeric=num_col).save('feature')
dataset(categorical=cat_col).save('feature')

print('Saving train set')
dataset(train=train).save('all')

print('Saving test set')
dataset(test=test).save('all')

print('Saving categorical data')
dataset(categorical=train[cat_col]).save('train')
dataset(categorical=test[cat_col]).save('test')
np.save('cat.npy', train[cat_col])

print('Saving numeric data')
dataset(numeric=train[num_col]).save('train')
dataset(numeric=test[num_col]).save('test')

print('Saving target data')
dataset(target=train[target_var]).save('train')

print('Done!')