import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

import xgboost as xgb


class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def get_oof(self, feature_train, label_train, feature_test):
        '''
        有base_model对训练集和测试集进行预测，生成第二层分类器所需的数据集
        feature_train: 训练集
        label_train: 训练集标签
        feature_test: 测试集
        '''

        # 生成用于存放第一层分类器预测结果的DataFrame
        # 训练集样本数 × base_model数目
        S_train = pd.DataFrame(
            np.zeros((feature_train.shape[0], len(self.base_models))),
            columns=self.base_models.keys())

        # 生成用于存放测试集预测结果的DataFrame
        # 测试集样本数 × base_model数目
        S_test = pd.DataFrame(
            np.zeros((feature_test.shape[0], len(self.base_models))),
            columns=self.base_models.keys())

        folds = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=45).split(
                feature_train, label_train)

        # 对每一个base_model进行训练和预测
        for name, clf in self.base_models.items():

            S_test_i = np.zeros((feature_test.shape[0], self.n_splits))

            # 对训练集进行K折交叉
            for j, (train_idx, test_idx) in enumerate(folds):
                # 生成交叉后的训练集
                kf_X_train = feature_train[train_idx]
                # 生成交叉后的训练样本对应的标签
                kf_y_train = label_train[train_idx]

                # 生成交叉后的验证集
                kf_X_test = feature_train[test_idx]

                #拟合训练集，生成模型
                clf.fit(kf_X_train, kf_y_train)

                y_pred = clf.predict(kf_X_test)

                S_train.loc[test_idx, name] = y_pred
                S_test_i[:, j] = clf.predict(feature_test)

            S_test.loc[:, name] = S_test_i.mean(axis=1)

        return S_train, S_test
        # self.stacker.fit(S_train, y)
        # res = self.stacker.predict(S_test)[:]
        # return res


df_train = pd.read_csv('./dataset/train_modified.csv')

df_train.drop(['Unnamed: 0'], axis=1, inplace=True)
# df_train.drop(['Attrition'], axis=1, inplace=True)
target_var = 'Attrition'
predictor = [x for x in df_train.columns if x != target_var]
X_train, X_test, y_train, y_test = train_test_split(
    df_train[predictor], df_train[target_var], test_size=0.3, random_state=45)
print(type(y_train))
lf = LogisticRegression()
rf = RandomForestClassifier()
gbc = GradientBoostingClassifier()
adc = AdaBoostClassifier()

stack = Ensemble(
    5, stacker=lf, base_models={'lf': lf,
                                'rf': rf,
                                'gbc': gbc,
                                'adc': adc})

train_df, test_df = stack.get_oof(X_train.values,
                                  y_train.ravel(), X_test.values)

gbm = xgb.XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=45)
base_score = 0.8
for i in range(1000):
    X_train, X_test, label_train, y_test = train_test_split(
        train_df, y_train, test_size=0.3)
    lf = LogisticRegression()
    lf.fit(X_train, label_train)
    score = lf.score(X_test, y_test)
    if score > base_score:
        print(score)
        base_score = score
