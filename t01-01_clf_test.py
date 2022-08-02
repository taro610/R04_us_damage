import scipy.io
import random
import numpy as np
import torch

import os
import warnings
import numpy as np
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM

from scipy.io import loadmat
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import optuna

# supress warnings for clean output
warnings.filterwarnings("ignore")


# シードの固定
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


SEED = 42
# シード固定
fix_seed(SEED)

feature_file_tuple = [
    (
        '//features_train_normal_n_1st-vs-us_1st_fl.mat',
        '//features_test_normal_n_1st-vs-us_1st_fl.mat',
        '//features_test_anomaly_n_1st-vs-us_1st_fl.mat',
    ),
    (
        '//features_train_normal_n_1st-vs-us_4th_fl.mat',
        '//features_test_normal_n_1st-vs-us_4th_fl.mat',
        '//features_test_anomaly_n_1st-vs-us_4th_fl.mat',
    ),
    (
        '//features_train_normal_n_1st-vs-us_11th_fl.mat',
        '//features_test_normal_n_1st-vs-us_11th_fl.mat',
        '//features_test_anomaly_n_1st-vs-us_11th_fl.mat',
    ),
]

NORMAL = 0
ANOMALY = 1

# split ratio of test data
TEST_SIZE = 0.4

# for Optuna
SPLIT_RATIO = 0.1

# Number of search trials for Optuna
N_TRIALS = 50

# Define the number of iterations for evaluation
N_ITER = 10


def benchmark_base(clf, use_scaler: bool = False):
    '''
    'in_fl', 'sq_in_fl', 'm_in', 're_fl', 'sq_re_fl', 'm_rc',
    'dif_fl', 'sq_dif_fl', 'm_dif', 'std_dif', 'in_rdc_fl',
    'sq_in_rdc_fl', 're_rdc_fl', 'sq_re_rdc_fl', 'dif_rdc_fl',
    'sq_dif_rdc_fl', 'dif_hz_fl', 'sq_dif_hz_fl', 'dif_tm_fl',
    'sq_dif_tm_fl', 'lat'
    '''
    tr_norm, tes_norm, tes_anom = feature_file_tuple[1]
    path_head = './/inputs'
    tr_norm_dic = scipy.io.loadmat(path_head + tr_norm)
    tes_norm_dic = scipy.io.loadmat(path_head + tes_norm)
    tes_anom_dic = scipy.io.loadmat(path_head + tes_anom)

    print(tr_norm_dic.keys())

    # key_list = ['re_fl', 'in_fl', 'dif_fl', ]
    # key_list = ['sq_in_fl', 'sq_re_fl']
    # key_list = ['re_fl', 'dif_fl',]
    # key_list = ['m_in', 'm_dif', 'm_rc']
    # key_list = ['lat',] # no_scaler
    key_list = ['dif_fl', ] # no_scaler
    # key_list = ['re_fl']  # no_scaler
    # key_list = ['in_fl']  # no_scaler
    # key_list = ['in_fl', 're_fl']  # no_scaler
    # key_list = ['lat', 'dif_rdc_fl'] # scaler

    tr_norm_list = []
    tes_norm_list = []
    tes_anom_list = []
    for k in key_list:
        tr_norm_list.append(tr_norm_dic[k])
        tes_norm_list.append(tes_norm_dic[k])
        tes_anom_list.append(tes_anom_dic[k])

    tr_norm = np.hstack(tr_norm_list)
    tes_norm = np.hstack(tes_norm_list)
    tes_anom = np.hstack(tes_anom_list)

    # train and valid data
    X_train = tr_norm
    X_valid = np.concatenate([tes_norm, tes_anom])
    y_valid = np.concatenate([np.zeros(len(tes_norm)), np.ones(len(tes_anom))])

    print(X_train.shape)
    print(X_valid.shape)
    print(y_valid.shape)

    if use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)

    clf.fit(X_train)
    anomaly_scores = clf.decision_function(X_valid)
    roc = roc_auc_score(y_valid, anomaly_scores)
    print(roc)


def knn_clf():
    # 普通のKNN
    clf_name = 'KNN'
    n_neighbors = 10
    algorithm = 'auto'
    # algorithm = 'ball_tree'
    # algorithm = 'kd_tree'
    # algorithm = 'brute'
    contamination = 0.01
    leaf_size = 30
    clf = KNN(
        algorithm=algorithm,
        n_neighbors=n_neighbors,
        leaf_size=leaf_size,
        contamination=contamination,
    )
    # print(clf)

    # マハラノビス距離を用いる場合
    ''' np.cov
    rowvar
    ここがTrueだと各々の行が１つのデータの組だとみなされます。
    共分散を求める際は行間で求めます。
    Falseの場合、各々の列が１つのデータの組となります。
    '''
    # X_train_cov = np.cov(X_train, rowvar=False)
    # print(X_train_cov.shape)
    # n_neighbors = 10
    # algorithm = 'auto'
    # contamination = 0.01
    # leaf_size = 30
    # metric = 'mahalanobis'
    # metric_params = {'V': X_train_cov}
    #
    # clf = KNN(
    #     metric=metric,
    #     metric_params=metric_params,
    #     n_neighbors=n_neighbors,
    # )
    # # print(clf)

    return clf


def lof_clf():
    '''
    https://pyod.readthedocs.io/en/latest/pyod.models.html?highlight=lof#pyod.models.lof.LOF
    https://pyod.readthedocs.io/en/latest/_modules/pyod/models/lof.html#LOF
    '''

    # 普通のKNN
    n_neighbors = 50
    algorithm = 'auto'
    contamination = 0.01
    leaf_size = 30
    clf = LOF(
        algorithm=algorithm,
        n_neighbors=n_neighbors,
        leaf_size=leaf_size,
        contamination=contamination,
    )
    # print(clf)

    # マハラノビス距離を用いる場合
    ''' np.cov
    rowvar
    ここがTrueだと各々の行が１つのデータの組だとみなされます。
    共分散を求める際は行間で求めます。
    Falseの場合、各々の列が１つのデータの組となります。
    '''
    # X_train_cov = np.cov(X_train, rowvar=False)
    # print(X_train_cov.shape)
    # n_neighbors = 20
    # algorithm = 'auto'
    # contamination = 0.01
    # leaf_size = 30
    # metric = 'mahalanobis'
    # metric_params = {'V': X_train_cov}

    # clf = LOF(
    #     metric=metric,
    #     metric_params=metric_params,
    #     n_neighbors=n_neighbors,
    # )
    # # print(clf)
    return clf


def ifor_clf():
    '''
    https://pyod.readthedocs.io/en/latest/pyod.models.html?highlight=forest#pyod.models.iforest.IForest
    https://pyod.readthedocs.io/en/latest/_modules/pyod/models/iforest.html#IForest
    '''

    n_estimators = 100
    max_samples = 'auto'
    contamination = 0.01
    max_features = 1.0
    clf = IForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
    )
    print(clf)

    return clf


def osvm_clf():
    '''
    https://pyod.readthedocs.io/en/latest/pyod.models.html?highlight=svm#module-pyod.models.ocsvm
    https://pyod.readthedocs.io/en/latest/_modules/pyod/models/ocsvm.html#OCSVM
    '''

    kernel = 'rbf'
    gamma = 'auto'
    contamination = 0.01
    clf = OCSVM(
        kernel=kernel,
        gamma=gamma,
        contamination=contamination,
    )
    print(clf)

    return clf


if __name__ == "__main__":
    # clf = knn_clf()
    # clf = lof_clf()
    clf = ifor_clf()
    # clf = osvm_clf()

    # use_scaler = True
    use_scaler = False
    benchmark_base(clf, use_scaler=use_scaler)
