from util import dataset

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

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


y = dataset.load('target', 'train')

rf = RandomForestClassifier(random_state=45)

#result = pd.DataFrame(columns=)

for num_feature, cat_feature in feature_list:
    num_var = dataset.load(num_feature, 'train')
    cat_var = dataset.load(cat_feature, 'train')
    X = pd.concat([num_var, cat_var], axis=1)

    rf.fit(X, y)
    print(rf.feature_importances_)


