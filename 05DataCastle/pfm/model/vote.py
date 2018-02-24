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
from sklearn.ensemble import VotingClassifier

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
feature_list.append(('numeric_disc', 'categorical_dummy'))
feature_list.append(('numeric_disc', 'categorical'))
feature_list.append(('numeric_disc_dummy', 'categorical_dummy'))
feature_list.append(('numeric_disc_dummy', 'categorical'))

models = []
models.append(('LogisticRegression', LogisticRegression(random_state=seed)))
models.append(('RandomForestClassifier', RandomForestClassifier(
    random_state=seed)))
models.append(('AdaBoostClassifier', AdaBoostClassifier(random_state=45)))
models.append(('ExtraTreesClassifier',
               ExtraTreesClassifier(random_state=seed)))
models.append(('GradientBoostingClassifier', GradientBoostingClassifier(
    random_state=seed)))

rows = len(feature_list) * len(models)

result = pd.DataFrame(
    columns=['model', 'features', 'train_score', 'test_score'])

# result = pd.DataFrame(
#     np.zeros((rows, 4)),
#     columns=['model', 'features', 'train_score', 'test_score'])

y = dataset.load('target', 'train')

clf = VotingClassifier(models)

for num_feature, cat_feature in feature_list:
    print('{} + {}:'.format(num_feature, cat_feature), end='')
    num_var = dataset.load(num_feature, 'train')
    cat_var = dataset.load(cat_feature, 'train')
    X = pd.concat([num_var, cat_var], axis=1)

    cv_result = cross_validate(clf, X, y, cv=kfold)

    train_score = cv_result['train_score'].mean()
    test_score = cv_result['test_score'].mean()

    print('train score:{:.4f}, test score:{:.4f}'.format(
        train_score, test_score))

#dataset(cv_result=result).save('all')

print('Done!')
