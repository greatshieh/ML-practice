import sys
sys.path.append('./preprocessing')
from util import dataset

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

seed = 45
kfold = StratifiedKFold(n_splits=10, random_state=seed)

feature_list = []
feature_list.append(('numeric', 'categorical'))
feature_list.append(('numeric_stdscale', 'categorical'))
feature_list.append(('numeric_maxmin', 'categorical'))
feature_list.append(('numeric_log1p', 'categorical'))
feature_list.append(('numeric_boxcox', 'categorical'))
feature_list.append(('numeric', 'categorical_dummy'))
feature_list.append(('numeric_stdscale', 'categorical_dummy'))
feature_list.append(('numeric_maxmin', 'categorical_dummy'))
feature_list.append(('numeric_log1p', 'categorical_dummy'))
feature_list.append(('numeric_boxcox', 'categorical_dummy'))

models = []
models.append(('LogisticRegression', LogisticRegression(random_state=seed)))
models.append(('RandomForestClassifier', RandomForestClassifier(
    random_state=45, n_estimators=20, max_depth=5)))
models.append(('AdaBoostClassifier', AdaBoostClassifier(
    random_state=45, n_estimators=60, learning_rate=.5)))
models.append(('ExtraTreesClassifier', ExtraTreesClassifier(
    random_state=seed, n_estimators=20)))
models.append(('GradientBoostingClassifier', GradientBoostingClassifier(
    random_state=seed, n_estimators=100, max_depth=3)))

rows = len(feature_list) * len(models)

result = pd.DataFrame(
    columns=['model', 'features', 'train_score', 'test_score'])

# result = pd.DataFrame(
#     np.zeros((rows, 4)),
#     columns=['model', 'features', 'train_score', 'test_score'])

y = dataset.load('target', 'train')

for num_feature, cat_feature in feature_list:
    print('{} + {}:'.format(num_feature, cat_feature))
    num_var = dataset.load(num_feature, 'train')
    cat_var = dataset.load(cat_feature, 'train')
    X = pd.concat([num_var, cat_var], axis=1)

    test_num = dataset.load(num_feature, 'test')
    test_cat = dataset.load(cat_feature, 'test')

    X_test = pd.concat([test_num, test_cat], axis=1)

    for name, clf in models:
        print(name + ':', end='')
        cv_result = cross_validate(estimator=clf, X=X, y=y, cv=kfold)

        train_score = cv_result['train_score'].mean()
        test_score = cv_result['test_score'].mean()

        result.loc[len(result)] = [
            name, '{} + {}:'.format(num_feature, cat_feature), train_score,
            test_score
        ]

        print('train score:{:.4f}, test score:{:.4f}'.format(
            train_score, test_score))

        clf.fit(X, y)
        y_pred = clf.predict(X_test)

        submission = pd.DataFrame({
            'Loan_Status': y_pred,
            'Loan_ID': X_test.index.tolist()
        })
        # submission['Loan_Status'] = submission['Loan_Status'].map({
        #     1: 'Y',
        #     0: 'N'
        # })
        filename = './result/{}_{}_{}.csv'.format(num_feature, cat_feature,
                                                  name)
        submission.to_csv(filename, index=False)

dataset(cv_result=result).save('all')

print('Done!')
