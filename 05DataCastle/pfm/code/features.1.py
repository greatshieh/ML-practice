'''导入库'''
import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import pprint

from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# import validate as va

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
'''
# 连续变量
continuous_var = [
    'Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]
# 分类变量
categorical_var = [
    x for x in df_train.columns if x not in continuous_var and x != target_var
]

# 数据类型
# 数值型
numeric_var = [
    x for x in df_train.dtypes.index
    if df_train.dtypes[x] != 'object' and x != target_var and x not in continuous_var
]
# 字符型
character_var = [x for x in df_train.dtypes.index if df_train.dtypes[x] == 'object'
]
print(numeric_var)
print(character_var)
'''
# 字符型
character_var = [
    x for x in df_train.dtypes.index if df_train.dtypes[x] == 'object'
]
numeric_var = [
    x for x in df_train.dtypes.index
    if x != target_var and x not in character_var
]
# 将数值型变量标准化
scaler = MinMaxScaler()
pattern = scaler.fit(df_train[numeric_var])
df_train[numeric_var] = scaler.transform(df_train[numeric_var])
df_test[numeric_var] = scaler.transform(df_test[numeric_var])

df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

predictor = [x for x in df_train.columns if x != target_var]

validation_size = 0.3
seed = 7
scoring = 'accuracy'
X_train, X_test, y_train, y_test = train_test_split(
    df_train[predictor],
    df_train[target_var],
    test_size=validation_size,
    random_state=seed)
kfold = KFold(n_splits=10, random_state=seed)

model = LogisticRegression()
parameters = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1],
    'tol': [1e-6, 1e-5, 1e-4]
}

cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
msg = "原始模型交叉验证分数: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring=scoring, cv=kfold).fit(X_train, y_train)
print("优化模型交叉验证分数: %f" % (grid_search.best_score_))
pred_result = grid_search.best_estimator_.predict(X_test)
pred_score = accuracy_score(y_test, pred_result)
print('优化模型测试集分数： %.4f' % pred_score)
sub_result = grid_search.best_estimator_.predict(df_test[predictor])
submission = pd.DataFrame({'result': sub_result})
submission.to_csv('result.csv', index=False)
'''
# 将连续变量标准化
scaler = MinMaxScaler()
poly = PolynomialFeatures()
poly_features = poly.fit_transform(df_train[continuous_var])

poly_featuresdf = scaler.fit_transform(poly_features)
df = pd.DataFrame(poly_features)
df_train[continuous_var] = scaler.fit_transform(df_train[continuous_var])
df_train = pd.concat([df_train, df], axis=1)

poly_features = poly.fit_transform(df_test[continuous_var])
poly_features = scaler.fit_transform(poly_features)
df = pd.DataFrame(poly_features)
df_test[continuous_var] = scaler.fit_transform(df_test[continuous_var])
df_test = pd.concat([df_test, df], axis=1)

df_train[categorical_var] = df_train[categorical_var].astype(str)
df_test[categorical_var] = df_test[categorical_var].astype(str)

# 生成新的分类变量
var_temp = []
for x in range(len(categorical_var)):
    for y in range(len(categorical_var)):
        if x < y:
            df_train[
                categorical_var[x] + '_' +
                categorical_var[y]] = df_train[categorical_var[x]] + '-' + df_train[categorical_var[y]]
            df_test[
                categorical_var[x] + '_' +
                categorical_var[y]] = df_test[categorical_var[x]] + '-' + df_test[categorical_var[y]]
            var_temp.append(categorical_var[x] + '_' + categorical_var[y])

df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
predictor = [x for x in df_train.columns if x != target_var]
print(len(df_train.columns))
high_var = pd.DataFrame(sel.fit_transform(df_train[predictor]))
print(len(high_var.columns))
df_train = pd.concat([high_var, df_train[target_var]], axis=1)
df_test = pd.DataFrame(sel.fit_transform(df_test))

df_train['bias'] = -1
df_test['bias'] = -1

predictor = [x for x in df_train.columns if x in df_test.columns]

validation_size = 0.3
seed = 7
scoring = 'accuracy'
X_train, X_test, y_train, y_test = train_test_split(
    df_train[predictor],
    df_train[target_var],
    test_size=validation_size,
    random_state=seed)
kfold = KFold(n_splits=10, random_state=seed)

# clf = LogisticRegression()
# cv_results = cross_val_score(clf, X_train, y_train, cv=kfold, scoring=scoring)
# print('原始模型训练集分数： %.4f' % cv_results.mean())
# clf.fit(X_train, y_train)
# pred_result = clf.predict(X_test)
# pred_score = accuracy_score(y_test, pred_result)
# print('原始模型测试集分数： %.4f' % pred_score)
#
# parameters = {
#     'penalty': ['l1', 'l2'],
#     'C': [0.01, 0.1, 1],
#     'tol': [1e-6, 1e-5, 1e-4]
# }
# grid_search = GridSearchCV(
#     estimator=clf, param_grid=parameters, scoring=scoring, cv=kfold)
# grid_search = grid_search.fit(X_train, y_train)
# print('最优模型训练集得分是: %.4f' % grid_search.best_score_)
# pred_result = grid_search.best_estimator_.predict(X_test)
# pred_score = accuracy_score(y_test, pred_result)
# print('最优模型测试集分数： %.4f' % pred_score)

# predictions = grid_search.best_estimator_.predict(df_test[predictor])
# submission = pd.DataFrame({'result': predictions})
# submission.to_csv("result.csv", index=False)
results = []
names = []
models = []
models.append(('LR', LogisticRegression(), {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1],
    'tol': [1e-6, 1e-5, 1e-4]
}))
models.append(('RandomForest', RandomForestClassifier(max_features='sqrt'), {
    'max_depth': range(1, 11)
}))

for name, model, parameters in models:
    print('\n原始模型比较')
    print('=' * 40)
    cv_results = cross_val_score(
        model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

    print('\n最优模型比较')
    print('=' * 40)
    grid_search = GridSearchCV(
        estimator=model, param_grid=parameters, scoring=scoring, cv=kfold).fit(
            X_train, y_train)
    print("%s: %f" % (name, grid_search.best_score_))

    pred_result = grid_search.best_estimator_.predict(X_test)
    pred_score = accuracy_score(y_test, pred_result)
    print('最优模型测试集分数： %.4f' % pred_score)
'''
