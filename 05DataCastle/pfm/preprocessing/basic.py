'''
利用LabelEncoder将分类变量数值化，其他数据不做改变。
将处理后的数据按照分类变量和连续变量分开保存
'''
from util.util import dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# 定义变量类型
# 目标变量
target_var = 'Attrition'

# 连续变量
num_col = [
    'Age', 'MonthlyIncome', 'TotalWorkingYears', 'PercentSalaryHike',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager', 'NumCompaniesWorked'
]
# 有序变量
ord_col = [
    'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance',
    'TrainingTimesLastYear'
]

# 分类变量
cat_col = [
    'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
    'MaritalStatus', 'OverTime'
]

# 加载数据
print('LOADING......')
train = pd.read_csv('../dataset/pfm_train.csv')
test = pd.read_csv('../dataset/pfm_test.csv')

# train['source'] = 'train'
# test['source'] = 'test'

# df = pd.concat([train, test], axis=0)

# 将分类变量转化为数值
label_enc = LabelEncoder()
for x in cat_col:
    label_enc.fit(train[x])
    train[x] = label_enc.transform(train[x])
    test[x] = label_enc.transform(test[x])

# 将数据保存为pickle文件
print('Saving data...')
dataset(numeric=num_col, categorical=cat_col, order=ord_col).save('feature')

dataset(train=train, test=test).save('all')

dataset(
    categorical=train[cat_col],
    numeric=train[num_col],
    order=train[ord_col],
    target=train[target_var]).save('train')
dataset(
    categorical=test[cat_col], numeric=test[num_col],
    order=test[ord_col]).save('test')

print('Done!')