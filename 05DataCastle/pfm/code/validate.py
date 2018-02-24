'''
绘制学习曲线，roc曲线和PR曲线
'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


def plot_learning_curve(estimator,
                        title,
                        X,
                        y,
                        ylim=None,
                        cv=None,
                        n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20),
                        verbose=0):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，
         其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(15, 7))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r")
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g")
    plt.plot(
        train_sizes,
        train_scores_mean,
        'o-',
        color="r",
        label="Training score")
    plt.plot(
        train_sizes,
        test_scores_mean,
        'o-',
        color="g",
        label="Cross-validation score")
    plt.legend(loc="best")

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) +
                (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (
        test_scores_mean[-1] - test_scores_std[-1])

    return midpoint, diff


def plot_pr_curve(estimator, X, y, test_size=0.3):
    """
    画出data在某模型上的P-R曲线.
    参数解释
    ----------
    estimator : 你用的分类器。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=test_size, random_state=0)
    estimator.fit(X_train, Y_train)
    if hasattr(estimator, 'decision_function'):
        y_score = estimator.decision_function(X_test)
    else:
        y_score = estimator.predict_proba(X_test)[:, 1]
    average_precision = average_precision_score(Y_test, y_score)
    precision, recall, threshold = precision_recall_curve(Y_test, y_score)

    plt.figure(figsize=(15, 7))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.step(recall, precision, color='b', alpha=0.5, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.title('Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))

    return


def plot_roc_curve(estimator, X, y, test_size=0.3):
    """
    画出data在某模型上的ROC曲线.
    参数解释
    ----------
    estimator : 你用的分类器。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，
         其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=test_size, random_state=0)
    estimator.fit(X_train, Y_train)
    if hasattr(estimator, 'decision_function'):
        y_score = estimator.decision_function(X_test)
    else:
        y_score = estimator.predict_proba(X_test)[:, 1]
    roc_score = roc_auc_score(Y_test, y_score)
    fpr, tpr, threshold = roc_curve(Y_test, y_score)

    plt.figure(figsize=(15, 7))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.step(fpr, tpr, color='b', alpha=0.5, where='post')
    plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('ROC curve: AUC={0:0.2f}'.format(roc_score))

    return
