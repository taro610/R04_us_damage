from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn import datasets
from sklearn import metrics
import random
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM

import math

import cv2
from scipy import ndimage

import scipy
from tqdm import tqdm

import functools
from collections import OrderedDict

from typing import Dict, Optional, List, Tuple
# from util.auto_encoder_model_tools import get_stacked_data, compute_recons_data, create_data_set_dict
import gc
from functools import partial

import pandas as pd


def get_stacked_data(dataset_dict: Dict, key_list: List) -> np.ndarray:
    """
    create_data_set_dict(five_elems: np.ndarray)で生成した辞書を渡して、key_listに含まれる要素をhstackする
    """
    data_list = [dataset_dict.get(k) for k in key_list if k in dataset_dict.keys()]
    return np.hstack(data_list)


# 標準化を適用する関数
def stand_scaler(tr_normal, test_normal, test_anomaly):
    scaler = StandardScaler()
    scaler.fit(tr_normal)

    tr_normal_sd = scaler.transform(tr_normal)
    test_normal_sd = scaler.transform(test_normal)
    test_anomaly_sd = scaler.transform(test_anomaly)

    return tr_normal_sd, test_normal_sd, test_anomaly_sd


# pcaを適用する関数
def pca_reduce(tr_normal, test_normal, test_anomaly, dims=1700):
    pca = PCA(n_components=dims)  # 0.8016444
    pca.fit(tr_normal)

    tr_normal_pca = pca.transform(tr_normal)
    test_normal_pca = pca.transform(test_normal)
    test_anomaly_pca = pca.transform(test_anomaly)

    # print('pca parcentage')
    # print(np.cumsum(pca.explained_variance_ratio_))

    return tr_normal_pca, test_normal_pca, test_anomaly_pca


# 入力したテストデータをそのまま評価値として使う
def donothing_pred(tr_normal, test_normal, test_anomaly):
    _ = tr_normal

    pred_test = test_normal
    pred_outliers = test_anomaly

    return pred_test, pred_outliers


def od_lof_pred(tr_normal, test_normal, test_anomaly, n_neighbors=50):
    '''
    https://pyod.readthedocs.io/en/latest/pyod.models.html?highlight=lof#pyod.models.lof.LOF
    https://pyod.readthedocs.io/en/latest/_modules/pyod/models/lof.html#LOF
    '''
    algorithm = 'auto'
    contamination = 0.01
    leaf_size = 30
    clf = LOF(
        algorithm=algorithm,
        n_neighbors=n_neighbors,
        leaf_size=leaf_size,
        contamination=contamination,
    )
    clf.fit(tr_normal)
    pred_test = clf.decision_function(test_normal) if test_normal is not None else None
    pred_outliers = clf.decision_function(test_anomaly) if test_anomaly is not None else None

    return pred_test, pred_outliers


def od_knn_pred(tr_normal, test_normal, test_anomaly, n_neighbors=10):
    '''
    https://pyod.readthedocs.io/en/latest/pyod.models.html?highlight=knn#pyod.models.knn.KNN
    https://pyod.readthedocs.io/en/latest/_modules/pyod/models/knn.html#KNN
    '''
    algorithm = 'auto'
    contamination = 0.01
    leaf_size = 30

    clf = KNN(
        algorithm=algorithm,
        n_neighbors=n_neighbors,
        leaf_size=leaf_size,
        contamination=contamination,
    )

    clf.fit(tr_normal)
    pred_test = clf.decision_function(test_normal) if test_normal is not None else None
    pred_outliers = clf.decision_function(test_anomaly) if test_anomaly is not None else None

    return pred_test, pred_outliers


def od_knn_maha_pred(tr_normal, test_normal, test_anomaly, n_neighbors=10):
    '''
    https://pyod.readthedocs.io/en/latest/pyod.models.html?highlight=knn#pyod.models.knn.KNN
    https://pyod.readthedocs.io/en/latest/_modules/pyod/models/knn.html#KNN
    '''

    # マハラノビス距離を用いる場合
    ''' np.cov
    rowvar
    ここがTrueだと各々の行が１つのデータの組だとみなされます。
    共分散を求める際は行間で求めます。
    Falseの場合、各々の列が１つのデータの組となります。
    '''
    X_train_cov = np.cov(tr_normal, rowvar=False)
    # print(X_train_cov.shape)
    algorithm = 'auto'
    contamination = 0.01
    leaf_size = 30
    metric = 'mahalanobis'
    metric_params = {'V': X_train_cov}

    clf = KNN(
        metric=metric,
        metric_params=metric_params,
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        leaf_size=leaf_size,
        contamination=contamination,
    )

    clf.fit(tr_normal)
    pred_test = clf.decision_function(test_normal) if test_normal is not None else None
    pred_outliers = clf.decision_function(test_anomaly) if test_anomaly is not None else None

    return pred_test, pred_outliers


# LOFで評価する関数
def lof_pred(tr_normal, test_normal, test_anomaly, n_neighbors=100):
    '''
    トレーニングデータから外れサンプルを探すときには (anomaly detection)、
    デフォルトの設定 (novelty=False) でよいが 、トレーニングデータが 存在するときの、
    テストデータのデータ密度を計算するときは(novelty detection)、novelty=True とする

    contamination で外れサンプルの割合を設定する必要がある
    '''

    # clf = LocalOutlierFactor(n_neighbors=50, novelty=True, contamination=0.0001)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    clf.fit(tr_normal)

    # pred_test = clf.predict(test_normal)
    # pred_outliers = clf.predict(test_anomaly)

    # score_samples低いほど異常である。
    # pred_test = clf.score_samples(test_normal)
    # pred_outliers = clf.score_samples(test_anomaly)
    pred_test = clf.score_samples(test_normal) if test_normal is not None else None
    pred_outliers = clf.score_samples(test_anomaly) if test_anomaly is not None else None

    #     # decision_functionの結果は負の値は外れ値、正の値は外れ値を表す。こっち使うとAUCの評価のときまずいことになりそう。
    #     pred_test = clf._decision_function(test_normal)
    #     pred_outliers = clf._decision_function(test_anomaly)

    return pred_test, pred_outliers


# OneClassSVMで評価する関数
def onesvm_pred(tr_normal, test_normal, test_anomaly, nu=0.01, gamma='auto', kernel='rbf'):
    # clf = OneClassSVM(nu=0.01, gamma=0.1, kernel='rbf')
    clf = OneClassSVM(nu=nu, gamma=gamma, kernel=kernel)

    clf.fit(tr_normal)

    # pred_test = clf.predict(all_test_normal)
    # pred_outliers = clf.predict(all_test_anomaly)

    # pred_test = clf.score_samples(test_normal)
    # pred_outliers = clf.score_samples(test_anomaly)
    pred_test = clf.score_samples(test_normal) if test_normal is not None else None
    pred_outliers = clf.score_samples(test_anomaly) if test_anomaly is not None else None

    return pred_test, pred_outliers


def isofore_pred(tr_normal, test_normal, test_anomaly, max_features=4, contamination=0.05):
    clf = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,
        max_features=max_features,
        bootstrap=False,
        n_jobs=-1,
        random_state=1
    )
    clf.fit(tr_normal)

    # 低い程異常値が大きい
    # pred_test = clf.score_samples(test_normal)
    # pred_outliers = clf.score_samples(test_anomaly)
    pred_test = clf.score_samples(test_normal) if test_normal is not None else None
    pred_outliers = clf.score_samples(test_anomaly) if test_anomaly is not None else None

    return pred_test, pred_outliers


def mahalanobis_list(train_data: np.ndarray):
    '''
    1変数当たりのマハラノビス距離計算
    '''
    data_num = train_data.shape[0]
    param_num = train_data.shape[1]

    # 平均値
    average = np.mean(train_data, axis=0)
    # 偏差
    deviation = train_data - average
    # 分散共分散行列を計算
    cov = np.cov(train_data.T)
    # 分散共分散行列の逆行列を計算
    cov_i = np.linalg.pinv(cov)

    # distance.mahalanobis
    # データ: train_data[i], 平均値: mean, 共分散行列の逆行列: np.linalg.pinv(cov) から距離を計算
    print('Calculation of Mahalanobis Distance')
    m_d = [scipy.spatial.distance.mahalanobis(train_data[i], average, cov_i) / param_num for i in tqdm(range(data_num))]

    # SN比解析
    print('Calculation of SN Ratio')
    sn = [10 * np.log10((deviation[i] ** 2) / np.diag(cov)) for i in tqdm(range(data_num))]

    return np.array(m_d), average, cov, cov_i, np.array(sn)


# すでに計算した平均や共分散を使ってマハラノビス距離を計算
def mahalanobis_apply(
        test_data: np.ndarray,
        average: np.ndarray,
        cov: np.ndarray,
        cov_i: np.ndarray,
):
    '''
    すでに計算した平均や共分散を使ってマハラノビス距離を計算

    mean,cov,cov_iは正常データで求めたもの

    '''
    data_num = test_data.shape[0]
    param_num = test_data.shape[1]

    # 偏差
    deviation = test_data - average

    # distance.mahalanobis
    # データ: train_data[i], 平均値: mean, 共分散行列の逆行列: np.linalg.pinv(cov) から距離を計算
    print('Calculation of Mahalanobis Distance')
    m_d = [scipy.spatial.distance.mahalanobis(test_data[i], average, cov_i) / param_num for i in tqdm(range(data_num))]

    # SN比解析
    print('Calculation of SN Ratio')
    sn = [10 * np.log10((deviation[i] ** 2) / np.diag(cov)) for i in tqdm(range(data_num))]

    return np.array(m_d), np.array(sn)


def mahalanobis_pred(tr_normal, test_normal, test_anomaly):
    m_d, mean, cov, cov_i, SN1 = mahalanobis_list(tr_normal)
    print('MD shape', m_d.shape)
    print('param mean:', mean)
    print('param cov:\n', cov)
    print('SN ratio shape', SN1.shape)

    # 計算済みの共分散をもちいて算出
    if test_normal is not None:
        pred_test, pred_test_sn = mahalanobis_apply(test_normal, mean, cov, cov_i)
    else:
        pred_test, pred_test_sn = None, None

    if test_anomaly is not None:
        pred_outliers, pred_outliers_sn = mahalanobis_apply(test_anomaly, mean, cov, cov_i)
    else:
        pred_outliers, pred_outliers_sn = None, None

    return pred_test, pred_outliers


# 2次元
def calc_mahara_2dim(tr_normal_2d, test_normal_2d, test_anomaly_2d):
    hz_num = tr_normal_2d.shape[1]
    md_train = []
    md_test_normal = []
    md_test_anomaly = []
    for h_i in range(hz_num):
        print(f"hz_index: {h_i} / {hz_num}")
        tr_normal = tr_normal_2d[:, h_i, :]
        test_normal = test_normal_2d[:, h_i, :]
        test_anomaly = test_anomaly_2d[:, h_i, :]  # (1000, 128)
        #         #これは時間軸と思う
        #         tr_normal = tr_normal_2d[:,:,h_i]
        #         test_normal = test_normal_2d[:,:,h_i]
        #         test_anomaly = test_anomaly_2d[:,:,h_i]

        # 周波数軸におけるマハラノビスの計算
        m_d, mean, cov, cov_i, SN1 = mahalanobis_list(tr_normal)
        # 計算済みの共分散で算出
        pred_test, pred_test_sn = mahalanobis_apply(test_normal, mean, cov, cov_i)
        pred_outliers, pred_outliers_sn = mahalanobis_apply(test_anomaly, mean, cov, cov_i)

        md_train.append(m_d)
        md_test_normal.append(pred_test)
        md_test_anomaly.append(pred_outliers)

    return np.array(md_train).transpose(), np.array(md_test_normal).transpose(), np.array(md_test_anomaly).transpose()


def compute_feature_2dim(tr_normal_2d, test_normal_2d, test_anomaly_2d, pred_function, is_slice_on_hz=True):
    '''
    tr_normal_2d.shape->(*, 128, 128):(sample, hz, time)とする

    出力は
    np.array(pred_test_list).shape->(128, *) : (hz or time, sample)

    '''
    hz_dim_num = tr_normal_2d.shape[1]
    tm_dim_num = tr_normal_2d.shape[2]

    dim_num = tr_normal_2d.shape[1] if is_slice_on_hz else tr_normal_2d.shape[2]

    pred_test_list = []
    test_anomaly_list = []
    for i in tqdm(range(dim_num)):
        #         print(f"index: {i} / {dim_num}")
        if is_slice_on_hz:  # 周波数軸でスライス
            tr_normal = tr_normal_2d[:, i, :]
            test_normal = test_normal_2d[:, i, :]
            test_anomaly = test_anomaly_2d[:, i, :]  # (1000, 128)
        else:  # 時間軸でスライス
            tr_normal = tr_normal_2d[:, :, i]
            test_normal = test_normal_2d[:, :, i]
            test_anomaly = test_anomaly_2d[:, :, i]  # (1000, 128)

        pred_test, pred_outliers = pred_function(tr_normal, test_normal, test_anomaly)

        pred_test_list.append(pred_test)
        test_anomaly_list.append(pred_outliers)

    #     return np.array(pred_test_list).transpose(), np.array(test_anomaly_list).transpose()
    return np.array(pred_test_list), np.array(test_anomaly_list)


def calc_aucs(pred_normal_fl, pred_anomaly_fl, is_pred_invert=False):
    # 評価値の正負を逆にする
    pred_normal_fl = -pred_normal_fl if is_pred_invert else pred_normal_fl
    pred_anomaly_fl = -pred_anomaly_fl if is_pred_invert else pred_anomaly_fl

    aucs = []
    for n, a in zip(pred_normal_fl, pred_anomaly_fl):
        y_true = np.concatenate([np.zeros(len(n)), np.ones(len(a))])
        y_score = np.concatenate([n, a])

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        #         print(auc)
        aucs.append(auc)
    return np.array(aucs)


def auc_plot(pred_test, pred_outliers, is_pred_invert=False, auc_img_dir=None, img_name=''):
    # plot
    plt.plot(pred_test, label='norm', lw=0.5)
    plt.plot(pred_outliers, label='anom', lw=0.5)
    plt.legend()
    if auc_img_dir is not None:
        plt.savefig(f"{auc_img_dir}/{img_name}_plot.png")
    # plt.show()
    plt.close()

    # ROC曲線の描画
    y_true = np.concatenate([np.zeros(len(pred_test)), np.ones(len(pred_outliers))])
    # 評価値の正負を逆にする
    pred_test = -pred_test if is_pred_invert else pred_test
    pred_outliers = -pred_outliers if is_pred_invert else pred_outliers

    y_score = np.concatenate([pred_test, pred_outliers])
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %.4f)' % auc)
    plt.legend()
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    if auc_img_dir is not None:
        plt.savefig(f"{auc_img_dir}/{img_name}_roc.png")
    # plt.show()
    plt.close()
    # print(auc)

    # F値の計算
    # FPRを0.1に抑えた時の閾値を採用する
    upper_fpr = 0.1
    fpr_index = np.argmin(np.abs(fpr - upper_fpr))
    gap = y_score - thresholds[fpr_index]
    y_pred = (gap > 0) * 1
    f_score = metrics.f1_score(y_true, y_pred)
    thresh = thresholds[fpr_index]
    # print("F-score under fpr=0.1:" + str(f_score))
    # print("thresholds under fpr=0.1:" + str(thresh))

    print(f"{img_name} - auc : {auc}")

    return auc, fpr, tpr


def eval_from_setting_dict(
        setting_dict: Dict,
        clf_method_dict: Dict,
        train_normal_dict: Dict,
        test_normal_dict: Dict,
        test_anomaly_dict: Dict,
        auc_img_dir: None,
        add_img_name: str = '',
):
    """
    setting_dictの例

        test_set_dict = {
            'used_data': ['dif_fl', 'm_sq'],
            'clf_method': 'lof',
            'clf_params': {'n_neighbors':100},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None, # None or int
        }

    """
    stack_list = setting_dict['used_data']
    clf_method = setting_dict['clf_method']
    clf_params = setting_dict['clf_params']
    is_pred_inv = setting_dict['is_pred_invert']
    is_stdz = setting_dict['is_stdz']
    pca_comps = setting_dict['pca_comps']

    method_func = clf_method_dict.get(clf_method)
    if clf_params is not None:
        method_func = functools.partial(method_func, **clf_params)

    all_tr_normal = get_stacked_data(train_normal_dict, stack_list)
    all_test_normal = get_stacked_data(test_normal_dict, stack_list)
    all_test_anomaly = get_stacked_data(test_anomaly_dict, stack_list)

    if is_stdz:
        all_tr_normal, all_test_normal, all_test_anomaly = stand_scaler(all_tr_normal, all_test_normal,
                                                                        all_test_anomaly)

    if pca_comps is not None:
        all_tr_normal, all_test_normal, all_test_anomaly = pca_reduce(all_tr_normal, all_test_normal, all_test_anomaly,
                                                                      dims=pca_comps)

    pred_test, pred_outliers = method_func(all_tr_normal, all_test_normal, all_test_anomaly)

    auc_img_name = add_img_name + f"{'-'.join(stack_list)}_{clf_method}_sdtz{is_stdz}_pca{'-' if pca_comps is None else str(pca_comps)}"
    auc, fpt, tpr = auc_plot(pred_test, pred_outliers, is_pred_invert=is_pred_inv, auc_img_dir=auc_img_dir,
                             img_name=auc_img_name)

    od = OrderedDict()
    od['auc'] = auc
    od['used_data'] = '-'.join(stack_list)
    od['clf_method'] = clf_method
    od['is_pred_invert'] = is_pred_inv
    od['is_stdz'] = is_stdz
    od['pca_comps'] = '-' if pca_comps is None else str(pca_comps)
    od['normal_ave'] = pred_test.mean()
    od['anomaly_ave'] = pred_outliers.mean()

    return od


def get_clf_map_dict() -> Dict:
    return {
        'nothing': donothing_pred,
        'lof': lof_pred,
        'osvm': onesvm_pred,
        'ifore': isofore_pred,
        'md': mahalanobis_pred,
        'od_lof': od_lof_pred,
        'od_knn': od_knn_pred,
        'od_knn_maha': od_knn_maha_pred,
    }


'--------------------------------------------------------------------'


def get_setting_list_od() -> List:
    eval_setting_list = [
        # {
        #     'used_data': ['in_hz_fl_r','in_hz_fl_m', 'in_hz_fl_s',],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['re_hz_fl_r','re_hz_fl_m', 're_hz_fl_s',],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['dif_hz_fl_r','dif_hz_fl_m', 'dif_hz_fl_s',],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['re_hz_fl_m', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['dif_hz_fl_r',],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['re_hz_fl_m', 'dif_hz_fl_r', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        ## Nothing
        {
            'used_data': ['m_in_r'],
            'clf_method': 'nothing',
            'clf_params': None,  # None or Dict
            'is_pred_invert': False,  # mseは大きい方が異常値
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_rc_r'],
            'clf_method': 'nothing',
            'clf_params': None,  # None or Dict
            'is_pred_invert': False,  # mseは大きい方が異常値
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_dif_r'],
            'clf_method': 'nothing',
            'clf_params': None,  # None or Dict
            'is_pred_invert': False,  # mseは大きい方が異常値
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        ## Local Outlier Factor (LOF)
        # single
        # {
        #     'used_data': ['m_in_m', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': False,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['m_rc_m', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': False,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['m_dif_m', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': False,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['m_in_s', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': False,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['m_rc_s', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': False,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['m_dif_s', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': False,
        #     'pca_comps': None,  # None or int
        # },
        {
            'used_data': ['m_in_r', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_rc_r', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_dif_r', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        # sn
        {
            'used_data': ['m_in_sn', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_rc_sn', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_dif_sn', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        ## multi
        {
            'used_data': ['m_dif_r', 'm_dif_s', 'm_rc_r', 'm_rc_s', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_dif_r', 'm_rc_r', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_dif_s', 'm_rc_s', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_in_r', 'm_in_s', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_rc_r', 'm_rc_s', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_dif_r', 'm_dif_s', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['lat', 'm_dif_r', 'm_dif_s', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_in_r', 'm_in_s', 'm_rc_r', 'm_rc_s', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        ## Knn
        # single
        {
            'used_data': ['m_in_r', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_rc_r', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_dif_r', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        # sn
        {
            'used_data': ['m_in_sn', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_rc_sn', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_dif_sn', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        ## multi
        {
            'used_data': ['m_dif_r', 'm_dif_s', 'm_rc_r', 'm_rc_s', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_dif_r', 'm_rc_r', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_dif_s', 'm_rc_s', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_in_r', 'm_in_s', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_rc_r', 'm_rc_s', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_dif_r', 'm_dif_s', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['lat', 'm_dif_r', 'm_dif_s', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        {
            'used_data': ['m_in_r', 'm_in_s', 'm_rc_r', 'm_rc_s', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        # {
        #     'used_data': ['m_in_r', 'm_in_m', 'm_in_s', 'm_rc_r'],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['m_in_r', 'm_in_m', 'm_in_s', 'm_dif_r'],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['m_in_r', 'm_in_m', 'm_in_s', 'm_dif_r'],
        #     'clf_method': 'od_knn',
        #     'clf_params': {'n_neighbors': 10},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['m_rc_m', 'm_dif_r', 'm_dif_m', 'm_dif_s', ],
        #     'clf_method': 'od_knn',
        #     'clf_params': {'n_neighbors': 10},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['lat', 'm_dif_r', 'm_dif_m', 'm_dif_s', ],
        #     'clf_method': 'od_knn',
        #     'clf_params': {'n_neighbors': 10},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['m_rc_m', 'm_rc_s', 'm_dif_m', 'm_dif_s', ],
        #     'clf_method': 'od_knn',
        #     'clf_params': {'n_neighbors': 10},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # # One-clas-svm 　フラット
        # {
        #     'used_data': ['m_in_r', 'm_in_m', 'm_in_s', 'm_dif_r'],
        #     'clf_method': 'osvm',
        #     'clf_params': {'nu': 0.01, 'gamma': 'auto', 'kernel': 'rbf', },
        #     'is_pred_invert': True,  # osvmは小さい方(0に近い方)が異常値
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # # IsolationForest　差分画像　フラット　＋　潜在空間 標準化なし
        # {
        #     'used_data': ['m_in_r', 'm_in_m', 'm_in_s', 'm_dif_r'],
        #     'clf_method': 'ifore',
        #     'clf_params': {'max_features': 4, 'contamination': 0.05, },
        #     'is_pred_invert': True,  # iforeは小さい方が異常値
        #     'is_stdz':  True,
        #     'pca_comps': None,  # None or int
        # },
        # # Local Outlier Factor (LOF)　フラット 標準化なし
        # {
        #     'used_data': ['m_in', 'm_dif', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['m_in', 'm_dif', ],
        #     'clf_method': 'od_knn',
        #     'clf_params': {'n_neighbors': 10},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['sq_in_stdrz_fl', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': False,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['m_in_rate', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': False,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['sq_in_stdrz_fl', 'm_dif_r', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['sq_in_stdrz_fl', 'sq_dif_fl', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['m_in_rate', 'm_dif_r', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['sq_in_fl', 'm_dif_r', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['sq_in_fl', 'sq_dif_fl', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # Local Outlier Factor (LOF) 差分画像　フラット 標準化なし

        # # Local Outlier Factor (LOF) 差分画像　フラット 標準化なし
        # {
        #     'used_data': ['in_fl', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': False,
        #     'pca_comps': None,  # None or int
        # },
        # # Local Outlier Factor (LOF) 差分画像　フラット 標準化なし
        # {
        #     'used_data': ['re_fl', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': False,
        #     'pca_comps': None,  # None or int
        # },
        # # Local Outlier Factor (LOF)　フラット 標準化なし
        # {
        #     'used_data': ['dif_fl', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': False,
        #     'pca_comps': None,  # None or int
        # },
        # # Local Outlier Factor (LOF) 差分画像　フラット 標準化なし
        # {
        #     'used_data': ['in_fl', ],
        #     'clf_method': 'od_knn',
        #     'clf_params': {'n_neighbors': 10},
        #     'is_pred_invert': False,
        #     'is_stdz': False,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['re_fl', ],
        #     'clf_method': 'od_knn',
        #     'clf_params': {'n_neighbors': 10},
        #     'is_pred_invert': False,
        #     'is_stdz': False,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['dif_fl', ],
        #     'clf_method': 'od_knn',
        #     'clf_params': {'n_neighbors': 10},
        #     'is_pred_invert': False,
        #     'is_stdz': False,
        #     'pca_comps': None,  # None or int
        # },
        # # {
        # #     'used_data': ['in_fl', ],
        # #     'clf_method': 'od_knn_maha',
        # #     'clf_params': {'n_neighbors': 10},
        # #     'is_pred_invert': False,
        # #     'is_stdz': False,
        # #     'pca_comps': None,  # None or int
        # # },
        # # {
        # #     'used_data': ['re_fl', ],
        # #     'clf_method': 'od_knn_maha',
        # #     'clf_params': {'n_neighbors': 10},
        # #     'is_pred_invert': False,
        # #     'is_stdz': False,
        # #     'pca_comps': None,  # None or int
        # # },
        # # {
        # #     'used_data': ['dif_fl', ],
        # #     'clf_method': 'od_knn_maha',
        # #     'clf_params': {'n_neighbors': 10},
        # #     'is_pred_invert': False,
        # #     'is_stdz': False,
        # #     'pca_comps': None,  # None or int
        # # },
        # {
        #     'used_data': ['in_fl', 're_fl', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # # Local Outlier Factor (LOF)　フラット 標準化なし
        # {
        #     'used_data': ['in_fl', 'dif_fl', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # # Local Outlier Factor (LOF)　フラット 標準化なし
        # {
        #     'used_data': ['re_fl', 'dif_fl', ],
        #     'clf_method': 'od_lof',
        #     'clf_params': {'n_neighbors': 50},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['in_fl', 're_fl', ],
        #     'clf_method': 'od_knn',
        #     'clf_params': {'n_neighbors': 10},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['in_fl', 'dif_fl', ],
        #     'clf_method': 'od_knn',
        #     'clf_params': {'n_neighbors': 10},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        # {
        #     'used_data': ['re_fl', 'dif_fl', ],
        #     'clf_method': 'od_knn',
        #     'clf_params': {'n_neighbors': 10},
        #     'is_pred_invert': False,
        #     'is_stdz': True,
        #     'pca_comps': None,  # None or int
        # },
        {
            'used_data': ['lat', ],
            'clf_method': 'od_lof',
            'clf_params': {'n_neighbors': 50},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        # knn 差分画像　フラット 標準化なし
        {
            'used_data': ['lat', ],
            'clf_method': 'od_knn',
            'clf_params': {'n_neighbors': 10},
            'is_pred_invert': False,
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
        # One-clas-svm 　フラット
        {
            'used_data': ['lat', ],
            'clf_method': 'osvm',
            'clf_params': {'nu': 0.01, 'gamma': 'auto', 'kernel': 'rbf', },
            'is_pred_invert': True,  # osvmは小さい方(0に近い方)が異常値
            'is_stdz': True,
            'pca_comps': None,  # None or int
        },
        # IsolationForest　差分画像　フラット　＋　潜在空間 標準化なし
        {
            'used_data': ['lat', ],
            'clf_method': 'ifore',
            'clf_params': {'max_features': 4, 'contamination': 0.05, },
            'is_pred_invert': True,  # iforeは小さい方が異常値
            'is_stdz': False,
            'pca_comps': None,  # None or int
        },
    ]
    return eval_setting_list


''' -------------------------------------------------------------------------------------- '''


def create_data_set_dict(five_elems: np.ndarray) -> Tuple[Dict]:
    """
        def compute_recons_data(model, dataloader, mid_features: List = [], pkl_file_path: str = None) の出力結果を前処理する


    ↓これがrecons_pkl()の返り値として出力されるデータ（サンプル数×5要素）
    print(outputs.shape) -> (1000, 5) # サンプルが1000個の時

    5つの要素の内容(AEの場合）
    print(outputs[0,0].shape) # input_img -> (1, 1, 128, 128)
    print(outputs[0,1].shape) # output_img -> (1, 1, 128, 128)
    print(outputs[0,2].shape) # diff_img -> (1, 1, 128, 128)
    print(outputs[0,3].shape) # mse -> () 1個のデータという意味
    print(outputs[0,4].shape) # Latent space -> (1, 1, 1024)

    5つの要素の内容(VAEの場合）
    4つめ5つめの要素の意味が違うはずなので注意


    """
    # 入力　2次元　元データ
    inputs = five_elems[:, 0]
    # 再構成　2次元　元データ
    recons = five_elems[:, 1]
    # 差分画像　2次元　元データ
    diff_outputs = five_elems[:, 2]
    # 潜在空間　1次元　元データ
    latent_outputs = five_elems[:, 4]
    #     print(np.squeeze(diff_outputs[0]).shape)
    #     print(np.squeeze(latent_outputs[0]).shape)

    # 差分画像を縮小して小さくしたもの
    # 画像サイズを変更してみる（パディングのイメージ）
    # size = (64, 64)
    size = (32, 32)
    diff_outputs_reduc = np.asarray([cv2.resize(np.squeeze(out), size) for out in diff_outputs])
    ## 遅い
    # diff_outputs_reduc = np.asarray([ndimage.zoom(np.squeeze(out), size) for out in diff_outputs])
    #     print(diff_outputs_reduc[0].shape)
    recons_reduc = np.asarray([cv2.resize(np.squeeze(out), size) for out in recons])
    inputs_reduc = np.asarray([cv2.resize(np.squeeze(out), size) for out in inputs])

    # 入力画像フラット （128x128=16384, 1次元）
    fl_inputs = np.asarray([out.ravel() for out in inputs])
    sq_fl_inputs = np.asarray([out ** 2 for out in fl_inputs])
    m_sq_inputs = np.asarray([np.mean(out) for out in sq_fl_inputs])
    m_inputs_mean = np.asarray([np.mean(out) for out in fl_inputs])
    m_inputs_std = np.asarray([np.std(out) for out in fl_inputs])
    m_inputs_rate = np.asarray([np.mean(out) / np.std(out) for out in fl_inputs])
    m_inputs_stdrzd_fl = np.asarray([(out - np.mean(out)) / np.std(out) for out in fl_inputs])
    sq_inputs_stdrzd = np.asarray([out ** 2 for out in m_inputs_stdrzd_fl])
    m_inputs_sn = np.asarray([math.log10(((np.mean(out) ** 2) / (np.std(out) ** 2 + 1e-8))) for out in fl_inputs])

    # 再構成フラット （128x128=16384, 1次元）
    fl_recons = np.asarray([out.ravel() for out in recons])
    sq_fl_recons = np.asarray([out ** 2 for out in fl_recons])
    m_sq_recons = np.asarray([np.mean(out) for out in sq_fl_recons])
    m_recons_mean = np.asarray([np.mean(out) for out in fl_recons])
    m_recons_std = np.asarray([np.std(out) for out in fl_recons])
    m_recons_sn = np.asarray([math.log10(((np.mean(out) ** 2) / (np.std(out) ** 2 + 1e-8))) for out in fl_recons])

    # 差分画像フラット（128x128=16384, 1次元）
    fl_diff_outputs = np.asarray([out.ravel() for out in diff_outputs])
    sq_fl_diff_outputs = np.asarray([out ** 2 for out in fl_diff_outputs])
    m_sq_diff = np.asarray([np.mean(out) for out in sq_fl_diff_outputs])
    m_diff_mean = np.asarray([np.mean(out) for out in fl_diff_outputs])
    m_diff_std = np.asarray([np.std(out) for out in fl_diff_outputs])
    std_sq_diff_outputs = np.asarray([np.std(out) for out in sq_fl_diff_outputs])
    m_diff_sn = np.asarray([math.log10(((np.mean(out) ** 2) / (np.std(out) ** 2 + 1e-8))) for out in fl_diff_outputs])

    # 入力画像フラット 縮小（ * x * = *, 1次元）
    fl_inputs_reduc = np.asarray([out.ravel() for out in inputs_reduc])
    sq_fl_inputs_reduc = np.asarray([out ** 2 for out in fl_inputs_reduc])

    # 再構成フラット 縮小（ * x * = *, 1次元）
    fl_recons_reduc = np.asarray([out.ravel() for out in recons_reduc])
    sq_fl_recons_reduc = np.asarray([out ** 2 for out in fl_recons_reduc])

    # 差分画像フラット 縮小（ * x * = *, 1次元）
    fl_diff_outputs_reduc = np.asarray([out.ravel() for out in diff_outputs_reduc])
    #     print(fl_diff_outputs_reduc.shape)
    #     print(fl_diff_outputs_reduc.min(), fl_diff_outputs_reduc.max())
    sq_fl_diff_outputs_reduc = np.asarray([out ** 2 for out in fl_diff_outputs_reduc])
    # m_sq_diff_outputs_reduc = np.asarray([np.mean(out) for out in sq_fl_diff_outputs_reduc])

    #
    fl_in_recons_mean = np.asarray([(i + o) / 2.0 for o, i in zip(fl_recons, fl_inputs)])

    # # 再構成と入力の比率
    # fl_recons_rate = np.asarray([o / i for o, i in zip(fl_recons, fl_inputs)])
    # sq_lf_recons_rate = np.asarray([out ** 2 for out in fl_recons_rate])
    # m_sq_recons_rate = np.asarray([np.mean(out) for out in sq_lf_recons_rate])

    # # 差分画像と入力画像の比率
    # fl_diff_outputs_rate = np.asarray([o / i for o, i in zip(sq_fl_diff_outputs, fl_inputs)])
    # sq_fl_diff_outputs_rate = np.asarray([out ** 2 for out in fl_diff_outputs_rate])
    # m_sq_diff_outputs_rate = np.asarray([np.mean(out) for out in sq_fl_diff_outputs_rate])

    # # 差分2乗画像と再構成2乗画像の比率
    # fl_diff_vs_recons_rate = np.asarray([np.log10(d/(r + 1e-6)) for d, r in zip(sq_fl_diff_outputs, sq_fl_recons)])
    # # fl_diff_vs_recons_rate = np.asarray([(d/(r + 1e-12)) for d, r in zip(sq_fl_diff_outputs, sq_fl_recons)])
    # fl_diff_vs_recons_rate = np.asarray([(d / (r + 1e-12)) for d, r in zip(fl_diff_outputs, fl_recons)])
    fl_diff_vs_recons_rate = np.asarray([(r / (d + 1e-12)) for d, r in zip(fl_diff_outputs, fl_recons)])
    # m_diff_vs_recons_rate = np.asarray([np.mean(out) for out in fl_diff_vs_recons_rate])

    # 潜在空間フラット（1024, 1次元）
    fl_latent_outputs = np.asarray([out.ravel() for out in latent_outputs])
    #     print(fl_latent_outputs.shape)
    #     print(fl_latent_outputs.min(), fl_latent_outputs.max())

    # # 入力　2次元　元データ
    # inputs = five_elems[:, 0]
    # # 再構成　2次元　元データ
    # recons = five_elems[:, 1]
    # # 差分画像　2次元　元データ
    # diff_outputs = five_elems[:, 2]

    # （128x128, 2次元）
    arr_inputs = np.asarray([np.squeeze(out) for out in inputs])
    arr_recons = np.asarray([np.squeeze(out) for out in recons])
    arr_diff = np.asarray([np.squeeze(out) for out in diff_outputs])

    # 差分二乗画像（128x128, 2次元）
    sq_arr_inputs = np.asarray([out ** 2 for out in arr_inputs])
    sq_arr_recons = np.asarray([out ** 2 for out in arr_recons])
    sq_arr_diff = np.asarray([out ** 2 for out in arr_diff])

    # 周波数軸を残して平均（128, 1次元）
    hz_arr_inputs_mean = np.asarray([np.mean(out, axis=1) for out in arr_inputs])
    hz_arr_recons_mean = np.asarray([np.mean(out, axis=1) for out in arr_recons])
    hz_arr_diff_mean = np.asarray([np.mean(out, axis=1) for out in arr_diff])

    # 周波数軸を残してstd（128, 1次元）
    hz_arr_inputs_std = np.asarray([np.std(out, axis=1) for out in arr_inputs])
    hz_arr_recons_std = np.asarray([np.std(out, axis=1) for out in arr_recons])
    hz_arr_diff_std = np.asarray([np.std(out, axis=1) for out in arr_diff])

    # 周波数軸を残してrms（128, 1次元）
    hz_arr_inputs_rms = np.asarray([np.sqrt(np.mean(out, axis=1)) for out in sq_arr_inputs])
    hz_arr_recons_rms = np.asarray([np.sqrt(np.mean(out, axis=1)) for out in sq_arr_recons])
    hz_arr_diff_rms = np.asarray([np.sqrt(np.mean(out, axis=1)) for out in sq_arr_diff])

    # 時間軸を残して平均（128, 1次元）
    tm_arr_inputs_mean = np.asarray([np.mean(out, axis=0) for out in arr_inputs])
    tm_arr_recons_mean = np.asarray([np.mean(out, axis=0) for out in arr_recons])
    tm_arr_diff_mean = np.asarray([np.mean(out, axis=0) for out in arr_diff])

    # 時間軸を残してrms（128, 1次元）
    tm_arr_inputs_rms = np.asarray([np.sqrt(np.mean(out, axis=0)) for out in sq_arr_inputs])
    tm_arr_recons_rms = np.asarray([np.sqrt(np.mean(out, axis=0)) for out in sq_arr_recons])
    tm_arr_diff_rms = np.asarray([np.sqrt(np.mean(out, axis=0)) for out in sq_arr_diff])

    #
    # # 差分画像（128x128, 2次元）
    # arr_diff_outputs = np.asarray([np.squeeze(out) for out in diff_outputs])
    # #     print(arr_diff_outputs.shape)
    #
    # # 差分二乗画像（128x128, 2次元）
    # sq_arr_diff_outputs = np.asarray([out ** 2 for out in arr_diff_outputs])
    # #     print(sq_arr_diff_outputs.shape)
    #
    # # 再構成画像　周波数軸を残して平均（128, 1次元）
    # hz_arr_diff_outputs = np.asarray([np.mean(out, axis=1) for out in arr_diff_outputs])
    #
    # # 差分画像　周波数軸を残して平均（128, 1次元）
    # hz_arr_diff_outputs = np.asarray([np.mean(out, axis=1) for out in arr_diff_outputs])
    # #     print(hz_arr_diff_outputs.shape)
    #
    # # 差分画像　時間軸を残して平均（128, 1次元）
    # tm_arr_diff_outputs = np.asarray([np.mean(out, axis=0) for out in arr_diff_outputs])
    # #     print(tm_arr_diff_outputs.shape)
    #
    # # 差分二乗画像　周波数軸を残して平均（128, 1次元）
    # hz_sq_arr_diff_outputs = np.asarray([np.mean(out, axis=1) for out in sq_arr_diff_outputs])
    # #     print(hz_sq_arr_diff_outputs.shape)
    #
    # # 差分二乗画像　時間軸を残して平均（128, 1次元）
    # tm_sq_arr_diff_outputs = np.asarray([np.mean(out, axis=0) for out in sq_arr_diff_outputs])
    # #     print(tm_sq_arr_diff_outputs.shape)

    # 2次元のデータををまとめた辞書
    dict_2d = {
        'dif_arr': arr_diff,  # (*,128,128)  差分画像
        'sq_arr': sq_arr_diff,  # (*,128,128)　差分二乗画像
    }

    # 2次元のデータををまとめた辞書
    dict_fl = {
        # 入力
        'in_fl': fl_inputs,  # 入力フラット（*, 16384）,
        'sq_in_fl': sq_fl_inputs,  # 入力二乗画像フラット（*, 16384）
        'sq_in_stdrz_fl': sq_inputs_stdrzd,
        'm_in_r': np.expand_dims(m_sq_inputs, 1),  # rms
        'm_in_m': np.expand_dims(m_inputs_mean, 1),  # mean
        'm_in_s': np.expand_dims(m_inputs_std, 1),  # mean
        'm_in_rate': np.expand_dims(m_inputs_rate, 1),  # mean
        'm_in_sn': np.expand_dims(m_inputs_sn, 1),  # mean
        # 再構成
        're_fl': fl_recons,  # 再構成フラット（*, 16384）,
        'sq_re_fl': sq_fl_recons,  # 入力二乗再構成フラット（*, 16384）,
        'm_rc_r': np.expand_dims(m_sq_recons, 1),  # rms
        'm_rc_m': np.expand_dims(m_recons_mean, 1),  # mean
        'm_rc_s': np.expand_dims(m_recons_std, 1),  # mean
        'm_rc_sn': np.expand_dims(m_recons_sn, 1),  # mean
        # 差分
        'dif_fl': fl_diff_outputs,  # 差分画像フラット（*, 16384）,
        'sq_dif_fl': sq_fl_diff_outputs,  # 差分二乗画像フラット（*, 16384）
        'm_dif_r': np.expand_dims(m_sq_diff, 1),  # rms
        'm_dif_m': np.expand_dims(m_diff_mean, 1),  # mean
        'm_dif_s': np.expand_dims(m_diff_std, 1),  # mean
        'std_dif': np.expand_dims(std_sq_diff_outputs, 1),
        'm_dif_sn': np.expand_dims(m_diff_sn, 1),  # mean
        # 入力(画像縮小)
        'in_rdc_fl': fl_inputs_reduc,  # 入力画像フラット 縮小（*, *）
        'sq_in_rdc_fl': sq_fl_inputs_reduc,  # 入力二乗画像フラット 縮小（*, *）
        # 再構成(画像縮小)
        're_rdc_fl': fl_recons_reduc,  # 再構成画像フラット 縮小（*, *）
        'sq_re_rdc_fl': sq_fl_recons_reduc,  # 再構成二乗画像フラット 縮小（*, *）
        # 差分(画像縮小)
        'dif_rdc_fl': fl_diff_outputs_reduc,  # 差分画像フラット 縮小（*, *）
        'sq_dif_rdc_fl': sq_fl_diff_outputs_reduc,  # 差分二乗画像フラット 縮小（*, *）
        # 軸平均(周波数)
        'in_hz_fl_m': hz_arr_inputs_mean,
        're_hz_fl_m': hz_arr_recons_mean,
        'dif_hz_fl_m': hz_arr_diff_mean,  # 差分画像　周波数軸を残して平均（*, 128）
        'in_hz_fl_s': hz_arr_inputs_std,
        're_hz_fl_s': hz_arr_recons_std,
        'dif_hz_fl_s': hz_arr_diff_std,  # 差分画像　周波数軸を残して平均（*, 128）
        'in_hz_fl_r': hz_arr_inputs_rms,
        're_hz_fl_r': hz_arr_recons_rms,
        'dif_hz_fl_r': hz_arr_diff_rms,  # 差分画像　周波数軸を残してRMS (*,128)
        # 軸平均(時間)
        'dif_tm_fl_m': tm_arr_diff_mean,  # 差分画像　時間軸を残して平均（*, 128）
        'dif_tm_fl_r': tm_arr_diff_rms,  # 差分二乗画像　時間軸を残してRMS  (*,128)
        # 潜在空間
        'lat': fl_latent_outputs,  # 潜在空間フラット（*, 1024）
        # 're_rate_fl': fl_recons_rate,
        # 'sq_re_rate_fl': sq_lf_recons_rate,
        'dvr_rate_fl': fl_diff_vs_recons_rate,
        # 'dif_rate_fl': fl_diff_outputs_rate,
        # 'm_sq_rate': np.expand_dims(m_sq_diff_outputs_rate, 1),
        # 'm_rc_rate': np.expand_dims(m_sq_recons_rate, 1),
        # 'm_dvr_rate': np.expand_dims(m_diff_vs_recons_rate, 1),
        'in_re_mean_fl': fl_in_recons_mean,
    }

    return dict_fl, dict_2d
