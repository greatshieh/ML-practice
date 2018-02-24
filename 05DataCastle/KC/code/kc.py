import pandas as pd
import numpy as np

from datetime import datetime

import time

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb

import sys
sys.path.append('.')

columns = [
    'date', 'price', 'bedroom', 'bathroom', 'roomarea', 'parkarea', 'floornum',
    'score', 'buildingarea', 'baseroomarea', 'buildingdate', 'fixdate', 'lat',
    'long'
]
df_train = pd.read_csv('./dataset/kc_train.csv', header=None, names=columns)
columns.remove('price')
df_test = pd.read_csv('./dataset/kc_test.csv', header=None, names=columns)

# 处理时间
df_train['sale_year'] = df_train['date'] // 10000
df_train[
    'sale_month'] = df_train['date'] // 100 - (df_train['sale_year'] * 100)

df_test['sale_year'] = df_test['date'] // 10000
df_test['sale_month'] = df_test['date'] // 100 - (df_test['sale_year'] * 100)

df_train['building_year'] = df_train['buildingdate'] // 10000
df_train['building_month'] = df_train['buildingdate'] // 100 - (
    df_train['building_year'] * 100)

df_test['building_year'] = df_test['buildingdate'] // 10000
df_test['building_month'] = df_test['buildingdate'] // 100 - (
    df_test['building_year'] * 100)

df_train['fix_year'] = df_train['fixdate'] // 10000
df_train[
    'fix_month'] = df_train['fixdate'] // 100 - (df_train['fix_year'] * 100)

df_test['fix_year'] = df_test['fixdate'] // 10000
df_test['fix_month'] = df_test['fixdate'] // 100 - (df_test['fix_year'] * 100)

df_train.drop(['date', 'buildingdate', 'fixdate'], axis=1, inplace=True)
df_test.drop(['date', 'buildingdate', 'fixdate'], axis=1, inplace=True)

#df_train['price'] = np.log(df_train['price'])

columns = df_test.columns.tolist()

scaler = StandardScaler()
df_train[columns] = scaler.fit_transform(df_train[columns])
df_test[columns] = scaler.transform(df_test[columns])

X_train, X_test, y_train, y_test = train_test_split(
    df_train[columns], df_train['price'], test_size=0.3, random_state=7)

kfold = KFold(n_splits=5, random_state=45)

model = []
# model.append(('LinearRegression', LinearRegression(
#     fit_intercept=True, normalize=True), {
#         'alpha': [.1, .3, .5, .9],
#         'tol': np.logspace(0.0001, 1, 4)
#     }))
model.append(('Lasso', Lasso(random_state=45), {
    'alpha': [.1, .3, .5, .9],
    'tol': [0.1, 0.01, 0.001, 0.0001]
}))
model.append(('Ridge', Ridge(), {
    'alpha': [.1, .3, .5, .9],
    'tol': [0.1, 0.01, 0.001, 0.0001]
}))
#model.append(('RandomForestRegressor', RandomForestRegressor(
#    max_features='sqrt'), {
#        'max_depth': range(1, 11),
#        'n_estimators': [100, 200, 500],
#        'min_samples_leaf': range(1, 11)
#    }))
#
#model.append(('GradientBoostingRegressor', GradientBoostingRegressor(), {
#    'max_depth': range(1, 11),
#    'n_estimators': [100, 200, 500]
#}))

for name, regressor, parameters in model:
    grid_search = GridSearchCV(
        estimator=regressor, param_grid=parameters, cv=kfold).fit(
            X_train, y_train)
    result = grid_search.best_estimator_.predict(X_test)
    rmse = mean_squared_error(y_test, result)
    score = r2_score(y_test, result)
    print('%s, best score = %.4f; R^2 = %.4f; RMSE=%.4f\n' %
          (name, grid_search.best_score_, score, rmse))


dtrain = xgb.DMatrix(X_train, label=y_train)

SEED = 45

gbm = xgb.XGBRegressor(
    min_child_weight=5,
    max_depth=3,
    objective='reg:linear',
    gamma=0,
    reg_alpha=0.6,
    reg_lambda=1,
    learning_rate=0.1, 
    colsample_bytree=1.0, 
    seed=SEED, 
    n_estimators=2375, 
    subsample=1,
    verbose=True
)

gbm.fit(X_train, y_train)

i = 0
gbm_time = 0
num_runs = 100
for i in range(num_runs):
    start = time.time()
    gbm_pred = gbm.predict(X_test)
    end = time.time()
    gbm_time += end-start


#unlog_pred = np.exp(gbm_pred)
#unlog_y_test = np.exp(y_test)

gbm_dict ={
    'name': 'xgboost',
    'avg_pred': gbm_time/num_runs,
    'rmse': mean_squared_error(y_test, gbm_pred),
    'r^2': r2_score(y_test, gbm_pred)
}

print("GBM Avg Predict Time(μs): ", gbm_dict['avg_pred'])
print("GBM RMSE: ", gbm_dict['rmse'])
print("GBM r^2: ", gbm_dict['r^2'])
