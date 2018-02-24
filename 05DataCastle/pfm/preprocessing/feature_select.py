from util import dataset

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.feature_selection import RFE

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from matplotlib import pyplot as plt

print('Loading Data...')

y = dataset.load('target', 'train')

X = dataset.load('new_feature', 'train')
# X.drop(['source'], axis=1, inplace=True)
# num_var = dataset.load('numeric_maxmin', 'train')
# cat_var = dataset.load('categorical_dummy', 'train')

# X = pd.concat([num_var, cat_var], axis=1)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=45)

lr = LogisticRegression(random_state=45)
kfold = StratifiedKFold(n_splits=10, random_state=45)
roc_base = 0.5
base_n = 1
result = pd.DataFrame(columns=['roc', 'acc', 'cv'])
print('Size:{}'.format(X.shape))
print('Start...')
for i in range(1, len(X.columns) + 1):

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.3, random_state=45)

    rfe = RFE(lr, i)

    acc_list = []
    roc_list = []

    for train, test in kfold.split(X, y):

        X_new = rfe.fit_transform(X.loc[train, :], y[train])

        feature_list = X.columns[rfe.support_].tolist()

        # cv_result = cross_validate(estimator=lr, X=X[feature_list], y=y, cv=kfold)

        lr.fit(X_new, y[train])
        acc = lr.score(X.loc[test, feature_list], y[test])
        acc_list.append(acc)
        y_score = lr.predict_proba(X.loc[test, feature_list])[:, 1]

        roc_score = roc_auc_score(y[test], y_score)
        roc_list.append(roc_score)
    print('n = {}, ROC: {:.4f}, ACC: {:.4f}'.format(
        i, np.mean(roc_list), np.mean(acc_list)))

    # result.loc[len(result)] = [roc_score, acc, cv_result['test_score'].mean()]

    # if roc_score > roc_base:
    #     roc_base = roc_score
    #     base_n = i
print('Done!')
# result.plot()
# plt.show()
# rfe = RFE(lr, i, 1)
# X_new = rfe.fit_transform(X, y)

# seed = 45
# models = []
# models.append(('LogisticRegression', LogisticRegression(random_state=seed)))
# models.append(('RandomForestClassifier', RandomForestClassifier(
#     random_state=45, n_estimators=20, max_depth=5)))
# models.append(('AdaBoostClassifier', AdaBoostClassifier(
#     random_state=45, n_estimators=60, learning_rate=.5)))
# models.append(('ExtraTreesClassifier', ExtraTreesClassifier(
#     random_state=seed, n_estimators=20)))
# models.append(('GradientBoostingClassifier', GradientBoostingClassifier(
#     random_state=seed, n_estimators=100, max_depth=3)))
# plt.figure(figsize=(15, 7))
# for name, clf in models:
#     clf.fit(X_train, y_train)
#     if hasattr(clf, 'decision_function'):
#         y_score = clf.decision_function(X_test)
#     else:
#         y_score = clf.predict_proba(X_test)[:, 1]

#     roc_score = roc_auc_score(y_test, y_score)
#     fpr, tpr, threshold = roc_curve(y_test, y_score)

#     plt.plot(fpr, tpr, label='{}({})'.format(name, roc_score))
# plt.legend(loc='best')
# plt.show()