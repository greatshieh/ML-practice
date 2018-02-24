import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./dataset/pfm_train.csv')

# 目标变量
target_var = 'Attrition'

# 连续变量
num_col = [
    'Age', 'MonthlyIncome', 'TotalWorkingYears', 'PercentSalaryHike', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'NumCompaniesWorked'
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


def cat_dummies(df):
    data = df.copy()

    data = pd.get_dummies(data, drop_first=True)

    return data


def ord_dummies(df):
    data = df.copy()

    for col in data.columns:
        dummies = pd.get_dummies(data[col], prefix=col)
        data = pd.concat([data, dummies], axis=1)
        data.drop([col], axis=1, inplace=True)

    return data


print(ord_dummies(df[ord_col]))
