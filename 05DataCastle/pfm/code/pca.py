from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

pca = PCA(n_components=2)

df_train = pd.read_csv('./dataset/pfm_train.csv')
df_test = pd.read_csv('./dataset/pfm_test.csv')

# 前文分析过，两个变量方差为0，可以删除。
# EmployeeNumber是唯一识别号码，删除
df_train.drop(
    ['Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)
df_test.drop(
    ['Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

# 预测变量
target_var = 'Attrition'

# 连续变量
continuous_var = [
    'Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

scaler = MinMaxScaler()

df_train[continuous_var] = scaler.fit_transform(np.log(df_train[continuous_var]+0.001))

pca_result = pca.fit_transform(df_train[continuous_var])
##print(pca.components_)
#print(pca_result)
label = df_train[target_var].map({0:'red', 1:'green'})

first_pc = pca.components_[0]
second_pc = pca.components_[1]

fig = plt.figure()
for ii, jj in pca_result:
    plt.scatter(first_pc[0]*ii[0], first_pc[1]*ii[0], c='r')
    plt.scatter(second_pc[0]*ii[1], second_pc[1]*ii[1], c='c')
    #plt.scatter(jj[0], jj[1], c='b')
    plt.scatter(ii[0], ii[1], c='b')

plt.show()
#fig = plt.figure()
#for x in range(len(pca_result)):
#    plt.scatter(pca_result[x][0], pca_result[x][1], alpha=0.7, c=label[x], edgecolors='white')
#plt.show()