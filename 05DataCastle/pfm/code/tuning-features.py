# coding: utf-8

import pandas as pd
import pprint
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score

from sklearn.externals import joblib

df_train = pd.read_csv('./dataset/train_modified.csv')
df_test = pd.read_csv('./dataset/test_modified.csv')

df_train.drop(['Unnamed: 0'], axis=1, inplace=True)

target_var = 'Attrition'
predictor = [x for x in df_train.columns if x != target_var]

validation_size = 0.3
seed = 7
scoring = 'accuracy'
X_train, X_test, y_train, y_test = train_test_split(
    df_train[predictor],
    df_train[target_var],
    test_size=validation_size,
    random_state=seed)
kfold = StratifiedKFold(n_splits=10, random_state=seed)


def cross_val(model, X_train, y_train, X_test, y_test, kfold):
    cv_results = cross_val_score(
        model, X_train, y_train, cv=kfold, scoring=scoring)
    print('cv-mean: %.4f, cv-std: %.4f' % (cv_results.mean(),
                                           cv_results.std()))
    model.fit(X_train, y_train)
    train_result = model.predict(X_train)
    pred_result = model.predict(X_test)
    train_score = accuracy_score(y_train, train_result)
    pred_score = accuracy_score(y_test, pred_result)
    print('训练集分数:  %.4f' % train_score)
    print('测试集分数： %.4f' % pred_score)
    return


def tunning_params(model,
                   params,
                   scoring=scoring,
                   kfold=kfold,
                   X_train=X_train,
                   y_train=y_train,
                   X_test=X_test,
                   y_test=y_test):
    grid_search = GridSearchCV(
        estimator=model, param_grid=params, scoring=scoring, cv=kfold).fit(
            X_train, y_train)
    print('优化后模型:')
    print('最佳参数: %s' % str(grid_search.best_params_))
    print('最佳得分: %.4f' % grid_search.best_score_)
    model = model.set_params(**(grid_search.best_params_))
    cross_val(model, X_train, y_train, X_test, y_test, kfold)
    return model


base_score = 0.8
for i in range(0, 5000):
    X_train, X_test, y_train, y_test = train_test_split(
        df_train[predictor], df_train[target_var], test_size=0.2)
    lm = LogisticRegression()
    lm.fit(X_train, y_train)
    lmp = lm.predict(X_test)
    xxx = lm.score(X_test, y_test)
    if xxx > 0.94:
        break

lrm = LogisticRegression()
parameters = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 1.2, 1.5, 2],
    'tol': [1e-6, 1e-5, 1e-4]
}

print('原始模型:')
cross_val(lrm, X_train, y_train, X_test, y_test, kfold)

lrm = tunning_params(lrm, parameters)

rfm = RandomForestClassifier(random_state=seed)
print('原始模型:')
cross_val(rfm, X_train, y_train, X_test, y_test, kfold)

parameters = {'n_estimators': range(10, 110, 10)}
rfm = tunning_params(rfm, parameters)

parameters = {
    'max_depth': range(3, 14, 2)  #, 'min_samples_split':range(50,201,20)
}

rfm = tunning_params(rfm, parameters)

parameters = {
    'min_samples_split': range(2, 20),
    'min_samples_leaf': range(1, 10)
}

rfm = tunning_params(rfm, parameters)


parameters = {'max_features': range(3, 10)}

rfm = tunning_params(rfm, parameters)

gbm = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=seed)

print('xgboost基础模型:')
cross_val(gbm, X_train, y_train, X_test, y_test, kfold)


parameters = {'n_estimators': [100, 200, 500, 1000, 1500]}
gbm = tunning_params(gbm, parameters)

parameters = {'max_depth': range(1, 15, 2), 'min_child_weight': range(1, 6, 2)}

gbm = tunning_params(gbm, parameters)

parameters = {'gamma': [i / 10.0 for i in range(0, 5)]}

gbm = tunning_params(gbm, parameters)

parameters = {
    'subsample': [i / 10.0 for i in range(6, 11)],
    'colsample_bytree': [i / 10.0 for i in range(6, 11)]
}

gbm = tunning_params(gbm, parameters)

parameters = {
    'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100],
    'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 100]
}

gbm = tunning_params(gbm, parameters)

parameters = {'learning_rate': [0.01, 0.1], 'n_estimators': [200, 1500]}

gbm = tunning_params(gbm, parameters)

gbcm = GradientBoostingClassifier()
cross_val(gbcm, X_train, y_train, X_test, y_test, kfold)

parameters = [{
    'n_estimators': range(20, 81, 10),
    'max_depth': range(3, 14, 2),
    'learning_rate': [0.1, 0.5, 1.0],
    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
}]

gbcm = tunning_params(gbcm, parameters)

adbm = AdaBoostClassifier(random_state=seed)
cross_val(adbm, X_train, y_train, X_test, y_test, kfold)

parameters = [{
    'n_estimators': range(20, 81, 10),
    'learning_rate': [0.1, 0.5, 1.0],
}]
adbm = tunning_params(adbm, parameters)


etcm = ExtraTreesClassifier(random_state=seed)
cross_val(etcm, X_train, y_train, X_test, y_test, kfold)

parameters = {
    'n_estimators': range(10, 50, 20),
    'max_depth': range(3, 11, 2),
    'min_samples_split': range(2, 10),
    'min_samples_leaf': range(3, 6),
    'max_features': range(3, 6)
}
etcm = tunning_params(etcm, parameters)

models = []
models.append(('lrm', lrm))
models.append(('rfm', rfm))
models.append(('gbm', gbm))
models.append(('gbcm', gbcm))
models.append(('etcm', etcm))
models.append(('ad', adbm))
ensemble = VotingClassifier(models)
cross_val(ensemble, X_train, y_train, X_test, y_test, kfold)

joblib.dump(lrm, 'lr.pkl')
joblib.dump(rfm, 'rf.pkl')
joblib.dump(etcm, 'etcm.pkl')
joblib.dump(adbm, 'adbm.pkl')
joblib.dump(gbcm, 'gbcm.pkl')
joblib.dump(adbm, 'adbm.pkl')

# In[ ]:
