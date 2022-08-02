from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

import functools
from collections import OrderedDict

from typing import Dict, Optional, List
from functools import partial

import pandas as pd

from nn_model.clf_tools import donothing_pred, lof_pred, onesvm_pred, isofore_pred, mahalanobis_pred, \
    od_lof_pred, od_knn_pred, od_knn_maha_pred
from nn_model.clf_tools import compute_feature_2dim, calc_aucs

from nn_model.clf_tools import get_setting_list_od, eval_from_setting_dict

from nn_model.ae_model_tools import ae_compute_recons_data

from nn_model.clf_tools import get_clf_map_dict, create_data_set_dict, get_stacked_data

class AeEvalManager(object):
    def __init__(self, ae_model, normal_key, anomaly_key, output_dir, ):
        """
        EvaManagerに渡す normal_kやanomaly_kは、ただの文字情報としてつかうだけだと思う

        Parameters
        ----------
        setting_list
        normal_key
        anomaly_key
        output_img_dir
        """

        # クラス分類器　辞書
        self._clf_method_dict = {
            'nothing': donothing_pred,
            'lof': lof_pred,
            'osvm': onesvm_pred,
            'ifore': isofore_pred,
            'md': mahalanobis_pred,
            'od_lof': od_lof_pred,
            'od_knn': od_knn_pred,
            'od_knn_maha': od_knn_maha_pred,
        }
        self._eval_setting_list = get_setting_list_od()
        # self._eval_setting_list = get_setting_list()
        # self._eval_setting_list = get_setting_list_long()

        self._ae_model = ae_model
        self._normal_key = normal_key
        self._anomaly_key = anomaly_key
        self._output_dir = output_dir
        self._auc_result_2d_dict = OrderedDict()  # 周波数帯調査用
        self._auc_result_fl_list = []  # 分類器精度比較用

    def __call__(self, crf_tr_dataloader, tset_normal_dataloader, test_anomaly_dataloader):
        # _, aucs_hz = self.compute_eval_result(crf_tr_dataloader, tset_normal_dataloader, test_anomaly_dataloader)
        _, aucs_hz, features_tuple = self.compute_eval_result(crf_tr_dataloader, tset_normal_dataloader,
                                                              test_anomaly_dataloader)
        # return self._auc_result_fl_list, aucs_hz
        return self._auc_result_fl_list, aucs_hz, features_tuple

    def compute_eval_result(self, crf_tr_dataloader, tset_normal_dataloader, test_anomaly_dataloader):
        mid_features = []
        tr_normal_outputs = ae_compute_recons_data(self._ae_model, crf_tr_dataloader, mid_features=mid_features)
        mid_features = []
        normal_outputs = ae_compute_recons_data(self._ae_model, tset_normal_dataloader, mid_features=mid_features)
        mid_features = []
        anomaly_outputs = ae_compute_recons_data(self._ae_model, test_anomaly_dataloader, mid_features=mid_features)

        # データを成形した辞書型を生成
        train_normal_dict_fl, train_normal_dict_2d = create_data_set_dict(tr_normal_outputs)
        test_normal_dict_fl, test_normal_dict_2d = create_data_set_dict(normal_outputs)
        test_anomaly_dict_fl, test_anomaly_dict_2d = create_data_set_dict(anomaly_outputs)

        # フラットな1次元データを用いた評価
        # 評価用の設定リスト（eval_setting_list）の内容に関して評価を実施
        for e_dict in self._eval_setting_list:
            auc_result = eval_from_setting_dict(
                e_dict,
                self._clf_method_dict,
                train_normal_dict_fl,
                test_normal_dict_fl,
                test_anomaly_dict_fl,
                auc_img_dir=self._output_dir,
                add_img_name=f"{self._normal_key}-{self._anomaly_key}_"
            )
            auc_result['normal_key'] = self._normal_key
            auc_result['anomaly_key'] = self._anomaly_key
            self._auc_result_fl_list.append(auc_result)

        # 2次元データを用いた評価
        tr_normal_2d = train_normal_dict_2d['dif_arr']
        test_normal_2d = test_normal_dict_2d['dif_arr']
        test_anomaly_2d = test_anomaly_dict_2d['dif_arr']
        pred_function_for_2d = partial(lof_pred, n_neighbors=100)

        pred_normal_fl_2d, pred_anomaly_fl_2d = compute_feature_2dim(
            tr_normal_2d,
            test_normal_2d,
            test_anomaly_2d,
            pred_function_for_2d,
            is_slice_on_hz=True
        )
        aucs = calc_aucs(pred_normal_fl_2d, pred_anomaly_fl_2d, is_pred_invert=True)
        auc_2d_key = f"{self._normal_key}-{self._anomaly_key}"
        self._auc_result_2d_dict[auc_2d_key] = aucs

        # return auc_result, aucs
        return auc_result, aucs, (train_normal_dict_fl, test_normal_dict_fl, test_anomaly_dict_fl)

        # gc.collect()

    def save_result_csv(self):
        auc_df = pd.DataFrame(self._auc_result_fl_list)
        auc_csv_name = f"{self._normal_key}-{self._anomaly_key}_result_auc.csv"
        auc_df.to_csv(self._output_dir + '/' + auc_csv_name)

        auc_2d_df = pd.DataFrame(self._auc_result_2d_dict)
        auc_2d_csv_name = f"{self._normal_key}-{self._anomaly_key}_result_2d_auc.csv"
        auc_2d_df.to_csv(self._output_dir + '/' + auc_2d_csv_name)

    def save_result_2d_img(self):
        auc_2d_df = pd.DataFrame(self._auc_result_2d_dict)
        auc_2d_df.plot()
        plt.savefig(f"{self._output_dir}/{self._normal_key}-{self._anomaly_key}_result_2d.png")
        plt.close()

        # auc_2d_df.mean(axis=1).plot()
        # plt.savefig(f"{self._output_dir}/result_2d_mean.png")
        # plt.close()


class AeReconstructTest(object):
    def __init__(
            self,
            ae_model,
            output_dir,
            sample_idx: int = 10,
            add_name: str = '',
            plt_text:bool = False,  # 画像に数値テキストを出力するかどうか
    ):
        """

        指定したインデックス番号の再構成画像を参考データとして保存する

        Parameters
        ----------
        ae_model
        output_dir
        """

        self._ae_model = ae_model
        self._output_dir = output_dir
        self._sample_idx = sample_idx
        self._batch = 5  # 表示する画像の数
        self._add = add_name
        self._plt_text = plt_text

    def __call__(self, crf_tr_dataloader, tset_normal_dataloader, test_anomaly_dataloader, idx: int = 100):
        mid_features = []
        tr_normal_outputs = ae_compute_recons_data(self._ae_model, crf_tr_dataloader, mid_features=mid_features)
        mid_features = []
        normal_outputs = ae_compute_recons_data(self._ae_model, tset_normal_dataloader, mid_features=mid_features)
        mid_features = []
        anomaly_outputs = ae_compute_recons_data(self._ae_model, test_anomaly_dataloader, mid_features=mid_features)

        idx = self._sample_idx

        '''確認用'''
        normal_mse = normal_outputs[:, 3]  # compute_recons_data() -> aeから直接出力されたmse
        anomaly_mse = anomaly_outputs[:, 3]
        plt.plot(normal_mse, label='norm', lw=0.5)
        plt.plot(anomaly_mse, label='anorm', lw=0.5)
        plt.legend()
        # plt.savefig(f"{self._output_dir}/mse_tes_idx{idx}.png")
        plt.savefig(f"{self._output_dir}/mse_recons_tes{self._add}.png")
        plt.close()

        '''正常データ'''
        # idx = self._sample_idx
        idx = random.choice(range(0, len(normal_outputs) - self._batch))
        with sns.axes_style("white"):  # imshowのときに白いグリッドが入るのを避ける
            fig = plt.figure(figsize=(15, 9))
            for i in range(0, self._batch):
                in_img = np.squeeze(normal_outputs[idx + i, 0])
                out_img = np.squeeze(normal_outputs[idx + i, 1])
                diff_img = np.squeeze(normal_outputs[idx + i, 2])
                mse = normal_outputs[idx + i, 3]

                # 上の段に訓練データ
                plt.subplot(3, 5, i + 1)
                plt.imshow(in_img, 'gray')
                if self._plt_text:
                    plt.text(0, 0, f"RMS: {np.square(in_img).mean():12.8f}")
                    plt.text(0, 10, f"mean: {in_img.mean():12.8f}")
                    plt.text(0, 20, f"std: {in_img.std():12.8f}")
                    plt.text(0, 30, f"min: {in_img.min():12.8f}")
                    plt.text(0, 40, f"max: {in_img.max():12.8f}")
                    # plt.text(0, 50, f"SD : {np.square(in_img - in_img.mean()).mean():12.8f}")
                # 段に生成データ
                plt.subplot(3, 5, 5 + i + 1)
                plt.imshow(out_img, 'gray')
                if self._plt_text:
                    plt.text(0, 0, f"RMS: {np.square(out_img).mean():12.8f}")
                    plt.text(0, 10, f"mean: {out_img.mean():12.8f}")
                    plt.text(0, 20, f"std: {out_img.std():12.8f}")
                    plt.text(0, 30, f"min: {out_img.min():12.8f}")
                    plt.text(0, 40, f"max: {out_img.max():12.8f}")
                    # plt.text(0, 50, f"SD : {np.square(out_img - out_img.mean()).mean():12.8f}")
                # 下の段に生成データ
                plt.subplot(3, 5, 10 + i + 1)
                plt.imshow(diff_img, 'gray')
                if self._plt_text:
                    plt.text(0, 0, f"RMS: {np.square(diff_img).mean():12.8f}")
                    plt.text(0, 10, f"mean: {diff_img.mean():12.8f}")
                    plt.text(0, 20, f"std: {diff_img.std():12.8f}")
                    plt.text(0, 30, f"min: {diff_img.min():12.8f}")
                    plt.text(0, 40, f"max: {diff_img.max():12.8f}")
                    # plt.text(0, 50, f"SD : {np.square(diff_img - diff_img.mean()).mean():12.8f}")
                    # plt.text(0.01, 100, f"{mse:12.8f}")

            plt.savefig(f"{self._output_dir}/normal_recons_tes_idx{idx}{self._add}.png")
            plt.close()

        # joblib で保存
        joblib.dump(np.squeeze(normal_outputs[idx:idx + self._batch, 0]),
                    f"{self._output_dir}/normal_in_idx{idx}{self._add}.joblib", compress=3)
        joblib.dump(np.squeeze(normal_outputs[idx:idx + self._batch, 1]),
                    f"{self._output_dir}/normal_recons_idx{idx}{self._add}.joblib", compress=3)
        joblib.dump(np.squeeze(normal_outputs[idx:idx + self._batch, 2]),
                    f"{self._output_dir}/normal_diff_idx{idx}{self._add}.joblib", compress=3)

        '''異常データ'''
        idx = self._sample_idx
        with sns.axes_style("white"):  # imshowのときに白いグリッドが入るのを避ける
            fig = plt.figure(figsize=(15, 9))
            for i in range(0, self._batch):
                in_img = np.squeeze(anomaly_outputs[idx + i, 0])
                out_img = np.squeeze(anomaly_outputs[idx + i, 1])
                diff_img = np.squeeze(anomaly_outputs[idx + i, 2])
                mse = anomaly_outputs[idx + i, 3]

                # 上の段に訓練データ
                plt.subplot(3, 5, i + 1)
                plt.imshow(in_img, 'gray')
                if self._plt_text:
                    plt.text(0, 0, f"RMS: {np.square(in_img).mean():12.8f}")
                    plt.text(0, 10, f"mean: {in_img.mean():12.8f}")
                    plt.text(0, 20, f"std: {in_img.std():12.8f}")
                    plt.text(0, 30, f"min: {in_img.min():12.8f}")
                    plt.text(0, 40, f"max: {in_img.max():12.8f}")
                    # plt.text(0, 50, f"SD : {np.square(in_img - in_img.mean()).mean():12.8f}")
                # 段に生成データ
                plt.subplot(3, 5, 5 + i + 1)
                plt.imshow(out_img, 'gray')
                if self._plt_text:
                    plt.text(0, 0, f"RMS: {np.square(out_img).mean():12.8f}")
                    plt.text(0, 10, f"mean: {out_img.mean():12.8f}")
                    plt.text(0, 20, f"std: {out_img.std():12.8f}")
                    plt.text(0, 30, f"min: {out_img.min():12.8f}")
                    plt.text(0, 40, f"max: {out_img.max():12.8f}")
                    # plt.text(0, 50, f"SD : {np.square(out_img - out_img.mean()).mean():12.8f}")
                # 下の段に生成データ
                plt.subplot(3, 5, 10 + i + 1)
                plt.imshow(diff_img, 'gray')
                if self._plt_text:
                    plt.text(0, 0, f"RMS: {np.square(diff_img).mean():12.8f}")
                    plt.text(0, 10, f"mean: {diff_img.mean():12.8f}")
                    plt.text(0, 20, f"std: {diff_img.std():12.8f}")
                    plt.text(0, 30, f"min: {diff_img.min():12.8f}")
                    plt.text(0, 40, f"max: {diff_img.max():12.8f}")
                    # plt.text(0, 50, f"SD : {np.square(diff_img - diff_img.mean()).mean():12.8f}")
                    # plt.text(0, 100, f"{mse:12.8f}")

            plt.savefig(f"{self._output_dir}/anomaly_recons_tes_idx{idx}{self._add}.png")
            plt.close()

        # joblib で保存
        joblib.dump(np.squeeze(anomaly_outputs[idx:idx + self._batch, 0]),
                    f"{self._output_dir}/anomaly_in_idx{idx}{self._add}.joblib", compress=3)
        joblib.dump(np.squeeze(anomaly_outputs[idx:idx + self._batch, 1]),
                    f"{self._output_dir}/anomaly_recons_idx{idx}{self._add}.joblib", compress=3)
        joblib.dump(np.squeeze(anomaly_outputs[idx:idx + self._batch, 2]),
                    f"{self._output_dir}/anomaly_diff_idx{idx}{self._add}.joblib", compress=3)


class AeTransitCheckManager(object):
    def __init__(
            self,
            ae_model,
            normal_key,
            output_dir,
            eval_setting: Optional[List],
    ):
        """
        Parameters
        ----------
        setting_list
        normal_key
        anomaly_key
        output_img_dir
        """

        # クラス分類器　辞書
        # self._clf_method_dict = {
        #     'nothing': donothing_pred,
        #     'lof': lof_pred,
        #     'osvm': onesvm_pred,
        #     'ifore': isofore_pred,
        #     'md': mahalanobis_pred,
        #     'od_lof': od_lof_pred,
        #     'od_knn': od_knn_pred,
        #     'od_knn_maha': od_knn_maha_pred,
        # }
        self._clf_method_dict = get_clf_map_dict()

        sample_dic_list = [
            # Local Outlier Factor (LOF) 差分画像　フラット 標準化なし
            {
                'used_data': ['lat', ],
                'clf_method': 'lof',
                'clf_params': {'n_neighbors': 100},
                'is_pred_invert': True,  # lofは小さい方が異常値
                'is_stdz': True,
                'pca_comps': None,  # None or int
            },
        ]
        self._eval_setting_list = sample_dic_list if eval_setting is None else eval_setting
        self._ae_model = ae_model
        self._normal_key = normal_key
        self._output_dir = output_dir
        self._target_result_list = []

    def __call__(self, crf_tr_dataloader, target_dataloader):
        _ = self.compute_result(crf_tr_dataloader, target_dataloader)
        return self._target_result_list

    def compute_result(self, crf_tr_dataloader, target_dataloader):
        """
        test_narmal_dataloaderは無くてもよいダミーデータ

        Parameters
        ----------
        crf_tr_dataloader
        tset_normal_dataloader
        test_anomaly_dataloader

        Returns
        -------

        """
        mid_features = []
        tr_normal_outputs = ae_compute_recons_data(self._ae_model, crf_tr_dataloader, mid_features=mid_features)
        mid_features = []
        target_outputs = ae_compute_recons_data(self._ae_model, target_dataloader, mid_features=mid_features)

        # データを成形した辞書型を生成
        train_normal_dict_fl, train_normal_dict_2d = create_data_set_dict(tr_normal_outputs)
        target_dict_fl, target_dict_2d = create_data_set_dict(target_outputs)

        # フラットな1次元データを用いた評価
        # 評価用の設定リスト（eval_setting_list）の内容に関して評価を実施
        for e_dict in self._eval_setting_list:
            stack_list = e_dict['used_data']
            clf_method = e_dict['clf_method']
            clf_params = e_dict['clf_params']
            is_pred_inv = e_dict['is_pred_invert']
            is_stdz = e_dict['is_stdz']
            pca_comps = e_dict['pca_comps']

            method_func = self._clf_method_dict.get(clf_method)
            if clf_params is not None:
                method_func = functools.partial(method_func, **clf_params)

            all_tr_normal = get_stacked_data(train_normal_dict_fl, stack_list)
            all_target = get_stacked_data(target_dict_fl, stack_list)

            if is_stdz:
                scaler = StandardScaler()
                scaler.fit(all_tr_normal)
                all_tr_normal = scaler.transform(all_tr_normal)
                all_target = scaler.transform(all_target)

            if pca_comps is not None:
                pca = PCA(n_components=pca_comps)
                pca.fit(all_tr_normal)
                all_tr_normal = pca.transform(all_tr_normal)
                all_target = pca.transform(all_target)

            _, pred_target = method_func(all_tr_normal, None, all_target)

            img_name = f"{self._normal_key}_" \
                       + f"{'-'.join(stack_list)}_{clf_method}_sdtz{is_stdz}_pca{'-' if pca_comps is None else str(pca_comps)}"

            # plot
            plt.plot(pred_target, label='transit', lw=0.5)
            plt.legend()
            if self._output_dir is not None:
                plt.savefig(f"{self._output_dir}/{img_name}_plot.png")
            # plt.show()
            plt.close()

            self._target_result_list.append(pred_target)

        return pred_target

        # gc.collect()

    def save_result_csv(self):
        auc_df = pd.DataFrame(self._target_result_list)
        auc_csv_name = f"{self._normal_key}_transit_check_result.csv"
        auc_df.to_csv(self._output_dir + '/' + auc_csv_name)
