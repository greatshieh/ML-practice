'''
xgboost调参过程
'''
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

import xgboost as xgb

import pandas as pd
import numpy as np
import pprint

import time

import warnings
warnings.filterwarnings('ignore')

# import validate as va

df_train = pd.read_csv('./dataset/pfm_train.csv')
df_test = pd.read_csv('./dataset/pfm_test.csv')

# 两个变量方差为0，可以删除。
# EmployeeNumber是唯一识别号码，删除
df_train.drop(
    ['Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)
df_test.drop(
    ['Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

# 预测变量
target_var = 'Attrition'

# 字符型
character_var = [
    x for x in df_train.dtypes.index if df_train.dtypes[x] == 'object'
]
numeric_var = [
    x for x in df_train.dtypes.index
    if x != target_var and x not in character_var
]

df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)
# 
# poly = PolynomialFeatures()
# poly_features = pd.DataFrame(poly.fit_transform(df_train[numeric_var]))
# df_train.drop(numeric_var, axis=1, inplace=True)
# df_train = pd.concat([df_train, poly_features], axis=1)

# poly_features = pd.DataFrame(poly.fit_transform(df_test[numeric_var]))
# df_test.drop(numeric_var, axis=1, inplace=True)
# df_test = pd.concat([df_test, poly_features], axis=1)

#df_train[numeric_var] = np.log10(df_train[numeric_var] + 0.001)
#df_test[numeric_var] = np.log10(df_test[numeric_var] + 0.001)

# 将数值型变量标准化
# scaler = StandardScaler()
# df_train[numeric_var] = scaler.fit_transform(df_train[numeric_var])
# df_test[numeric_var] = scaler.transform(df_test[numeric_var])

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

gbm = xgb.XGBClassifier(
    min_child_weight=1,
    max_depth=5,
    objective='reg:logistic',
    gamma=0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    learning_rate=0.1,
    colsample_bytree=1.0,
    colsample_bylevel=1.0,
    seed=seed,
    n_estimators=1000,
    subsample=1,
    verbose=True)

score = cross_val_score(gbm, X_train, y_train, cv=kfold)
print('cv-mean：%.4f, cv_std: %.4f' % (score.mean(), score.std()))
gbm.fit(X_train, y_train)
pred = gbm.predict(X_test)
score = accuracy_score(y_test, pred)
print('验证集得分: %.4f'%score)
'''
parameters = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}

grid_search = GridSearchCV(
    estimator=gbm, param_grid=parameters, scoring=scoring, cv=kfold).fit(
        X_train, y_train)

print("优化模型交叉验证分数: %f; 最优参数: %s" % (grid_search.best_score_,
                                    str(grid_search.best_params_)))

parameters = {'max_depth': [2, 3, 4], 'min_child_weight': [4, 5, 6]}

grid_search = GridSearchCV(
    estimator=gbm, param_grid=parameters, scoring=scoring, cv=kfold).fit(
        X_train, y_train)

print("优化模型交叉验证分数: %f; 最优参数: %s" % (grid_search.best_score_,
                                    str(grid_search.best_params_)))
'''
gbm = xgb.XGBClassifier(
    min_child_weight=5,
    max_depth=3,
    objective='reg:logistic',
    gamma=0.2,
    reg_alpha=1e-5,
    reg_lambda=1.0,
    learning_rate=0.05,
    colsample_bytree=1.0,
    colsample_bylevel=1.0,
    seed=seed,
    n_estimators=500,
    subsample=1,
    verbose=True)

# parameters = {'gamma':[i/10.0 for i in range(0,5)]}
# parameters = {
#  'subsample':[i/10.0 for i in range(6,11)],
#  'colsample_bytree':[i/10.0 for i in range(6,11)]
# }

score = cross_val_score(gbm, X_train, y_train, cv=kfold)
print('cv-mean：%.4f, cv_std: %.4f' % (score.mean(), score.std()))
gbm.fit(X_train, y_train)
pred = gbm.predict(X_test)
score = accuracy_score(y_test, pred)
print('验证集得分: %.4f'%score)
# parameters = {'learning_rate': [0.2, 0.05, 0.1], 'n_estimators': [500, 1000, 1500, 2000]}
# grid_search = GridSearchCV(
#     estimator=gbm, param_grid=parameters, scoring=scoring, cv=kfold).fit(
#         X_train, y_train)
# pprint.pprint(grid_search.grid_scores_)
# print("优化模型交叉验证分数: %.4f; 最优参数: %s" % (grid_search.best_score_,
#                                       str(grid_search.best_params_)))
sub_result = gbm.predict(df_test[predictor])
submission = pd.DataFrame({'result': sub_result})
submission.to_csv('result.csv', index=False)
