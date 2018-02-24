'''
数据的预处理和特征工程
'''

import pandas as pd
import numpy as np
import pprint

df_train = pd.read_csv('./dataset/pfm_train.csv')
df_test = pd.read_csv('./dataset/pfm_test.csv')

df_train['source'] = 1
df_test['source'] = 0

df = pd.concat([df_train, df_test], axis=0)

df.drop(['Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

# 预测变量
target_var = 'Attrition'

# 定义连续变量
cols_con = ['Age', 'DistanceFromHome', 'MonthlyIncome']

# 定义离散变量
cols_dis = [
    'BusinessTravel', 'Department', 'Education', 'EducationField',
    'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel',
    'JobRole', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked',
    'OverTime', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

# 连续变量离散化
N = 4  #分割的倍数

age_cut = pd.cut(df.Age, bins=range(18, 60, 3 * N))
age_dm = pd.get_dummies(age_cut, prefix='Age')

distance_cut = pd.cut(df.DistanceFromHome, bins=range(1, 29, 1 * N))
distance_dm = pd.get_dummies(distance_cut, prefix='Distance')

income_cut = pd.cut(df.MonthlyIncome, bins=range(1000, 20000, 1000 * N))
income_dm = pd.get_dummies(income_cut, prefix="income")

# 离散化变量的get_dummies
lst = []
for col in cols_dis:
    lst.append(pd.get_dummies(df[col], prefix=col))
df_dis = pd.concat(lst, axis=1)

# 整合起来
modelinput = pd.concat(
    [df_dis, age_dm, distance_dm, income_dm, df.Attrition, df.source], axis=1)

df_train = modelinput[modelinput.source == 1]
df_test = modelinput[modelinput.source == 0]
df_train.drop(['source'], axis=1, inplace=True)
df_test.drop(['source'], axis=1, inplace=True)
df_train.to_csv('./dataset/pfm_train_modifited.csv', index=False)
df_test.to_csv('./dataset/pfm_test_modifited.csv', index=False)
