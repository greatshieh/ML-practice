import sys
sys.path.append('./preprocessing')
from util import dataset

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

seed = 45
kfold = StratifiedKFold(n_splits=10, random_state=seed)

feature_list = []
feature_list.append(('categorical', 'numeric', 'order'))
feature_list.append(('categorical_dummy', 'numeric', 'order_dummy'))
feature_list.append(('categorical', 'numeric', 'order_dummy'))
feature_list.append(('categorical_dummy', 'numeric', 'order'))
feature_list.append(('custom_label', 'numeric', 'order_dummy'))
feature_list.append(('custom_label', 'numeric', 'order'))

lr = LogisticRegression(random_state=seed, C=30)

y = dataset.load('target', 'train')

params = {'penalty': ['l1', 'l2'], 'C': np.logspace(0.001, 100, 6)}

for cat_feature, num_feature, ord_feature in feature_list:
    print('{}+{}+{}==='.format(cat_feature, num_feature, ord_feature))
    ord_var = dataset.load(ord_feature, 'train')
    num_var = dataset.load(num_feature, 'train')
    cat_var = dataset.load(cat_feature, 'train')
    X = pd.concat([ord_var, num_var, cat_var], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed)

    # pos_train = X[X['Attrition']==1]

    cv_result = cross_validate(estimator=lr, X=X_train, y=y_train, cv=kfold)

    print('cv score: {:.4f}'.format(cv_result['test_score'].mean()))

    lr.fit(X_train, y_train)
    score = lr.score(X_test, y_test)
    print('score: {:.4f}'.format(score))

    # grid = GridSearchCV(estimator=lr, param_grid=params, cv=kfold)

    # grid.fit(X_train, y_train)
    # grid.best_estimator_.fit(X_train, y_train)
    # best_score = grid.best_estimator_.score(X_test, y_test)
    # print('best params: {}'.format(grid.best_params_))
    # print('best score: {:.4f}'.format(best_score))

# dataset(cv_result=result).save('all')

print('Done!')
