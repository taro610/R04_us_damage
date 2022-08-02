import seaborn as sns
import random
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional
import torch
from torchsummary import summary
import torch.nn as nn
from sklearn.model_selection import train_test_split

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import gc
from collections import OrderedDict
import pandas as pd
from util.path_tools import GetFileListBySuffix
from util.path_tools import include_list_picker_strs, ana_count_select_picker
# from util.dataset_tools import SpecDatasetForPytorch, DataTransform
from util.dataset_tools import gen_train_dataloader, gen_test_dataloader_dict
# # from util.this_exp_model import AutoEncoder, compute_recons_data, train_model, create_data_set_dict
from collections import OrderedDict

from nn_model.ae_model import AutoEncoderBN, AutoEncoderDrop, AutoEncoderDrop64
from nn_model.ae_model_tools import ae_compute_recons_data, ae_train_model
from nn_model.ae_clf_tools import AeReconstructTest, AeEvalManager

from nn_model.vae_model import VAE
from nn_model.vae_model_tools import vae_train_model
from nn_model.vae_clf_tools import VaeEvalManager, VaeReconstructTest

from util.dataset_dict_tools import get_file_list_from_dict, train_test_split_from_dict

from util.image_data import ArrayTransform, ArrayCompose, NdimageZoom, NdimageResize \
    , Normalize_0to1, ToTorchTensor1ch, GlobalConstractNormalization, Standardize, GCN_simple, GCN_simple_2

# font settings etc.
sns.set(font_scale=1.4, font="Times New Roman")
sns.set_style("darkgrid", {'font.family': 'serif', 'font.serif': 'Times New Roman'})
'''
fig.save_fig('test.png', bbox_inches="tight", pad_inches=0.05) # 'test.png'はpath+保存名です
plt.save_fig('test.png', bbox_inches="tight", pad_inches=0.05) # 'test.png'はpath+保存名です

これらは、plt.show()より前で実施しないと真っ白い画像が保存される
'''


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

# 定数の設定
IMG_SIZE = 128
# IMG_SIZE = 64
AE_MIN_BATCH = 64

# AE_NUM_EPOCH = 501
AE_NUM_EPOCH = 101
# AE_NUM_EPOCH = 301
# AE_NUM_EPOCH = 31

# TEST_SAMPLE_RATE = 0.1
TEST_SAMPLE_RATE = 0.2
# CLF_SAMPLE_RATE = 0.1
CLF_SAMPLE_RATE = 0.15

# AE_PICK_NUM = 7000
# AE_PICK_NUM = None  # AEトレーニングに使用するデータを絞るためにランダムピックする場合
# TEST_PICK_NUM = None  # テスト用データを絞るためにランダムピックする場合
TEST_PICK_NUM = 300  # テスト用データを絞るためにランダムピックする場合
# TEST_PICK_NUM = 200  # テスト用データを絞るためにランダムピックする場合
# TEST_PICK_NUM = 10  # テスト用データを絞るためにランダムピックする場合

# 解析対象データ
# DATA_TYPE = 'CWTx128_1rot_df10kh_olog'
# DATA_TYPE = 'CWTx128_1rot_df100k-1m_flog_olog_aaf_zc'
DATA_TYPE = 'CWTx128_1rot_df10kh_olog_aaf_zc'

SRC_DIR_HEAD = 'I:\\experimental_data\\110_processed_data'
# USE_STEPS = ['_st3.0', '_st4.0', '_st5.0']
USE_STEPS = ['_st3.0', '_st4.0', '_st5.0', '_st6.0', '_st7.0']  # これを使うときは折損直前の穴の6～7step目を要確認
# USE_STEPS = ['_st3.0', '_st4.0', '_st5.0', '_st6.0', ]  # これを使うときは折損直前の穴の6～7step目を要確認
OUTPUT_DIR = './outputs'

''' do train '''
IS_TRAIN = True
# IS_TRAIN = False

''' save extracted features '''
SAVE_FEATURES = False
# SAVE_FEATURES = True

''' stdrz '''
STDRZ = None

''' 0-1 clip '''
# CLIP = None
# CLIP = (-2.7536, 5.2146)  # 'CWTx128_1rot_df100k-1m_flog_olog_aaf_zc' raw -> 0-1clip
CLIP = (-2.5393, 5.3828)  # 'CWTx128_1rot_df10kh_olog_aff_zc' waxmaunt raw -> 0-1clip

''' pre processing '''
pre_pro_dic = {
    'gcn_l1': GlobalConstractNormalization('l1'),
    # 'gcn_l2': GlobalConstractNormalization('l2'),
    # 'gn_l2': GCN_simple(),
    'gn_l2': GCN_simple_2(),
    # 'stdrz': Standardize(STDRZ[0], STDRZ[1]),
    'clip': Normalize_0to1(CLIP[0], CLIP[1])
}

DATA_PREPRO_SETTING = [
    # 'stdrz',
    # 'gn_l2',
    'clip',
]

pre_pro = []
for k in DATA_PREPRO_SETTING:
    if k is not None:
        pre_pro.append(pre_pro_dic[k])
print(pre_pro)

DATA_LOADER_SETTING = {
    'img_size': IMG_SIZE,
    'pre_process': pre_pro,
}

''' AE model '''
ae_map = {
    'AE-DO': [AutoEncoderDrop, (ae_train_model, AeReconstructTest, AeEvalManager)],
    'AE-DO64': [AutoEncoderDrop64, (ae_train_model, AeReconstructTest, AeEvalManager)],
    'AE-BN': [AutoEncoderBN, (ae_train_model, AeReconstructTest, AeEvalManager)],
    'VAE': [VAE, (vae_train_model, VaeReconstructTest, VaeEvalManager)],
}

# AE_TYPE = 'AE-DO'
AE_TYPE = 'AE-BN'
# AE_TYPE = 'VAE'

AE_CLASS = ae_map.get(AE_TYPE)[0]
AE_TR_TOOL, AE_RECONS_CLASS, AE_EVAL_CLASS = ae_map.get(AE_TYPE)[1]

''' AE Setting '''
Z_DIM = 1024
# Z_DIM = 128
# Z_DIM = 32

decoder_out_map = {
    '-': None,
    'sig': nn.Sigmoid(),
    'relu': nn.ReLU(),
}

# DECODER_OUT = '-'
DECODER_OUT = 'sig'
# DECODER_OUT = 'relu'

AE_SETTING = {
    'z_dim': Z_DIM,
    'decoder_output_activation': decoder_out_map.get(DECODER_OUT)
}

'''
## SKD61
normal_dirname_dic = {
    'n_1st': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\02_norm_1st',  # * bat_02
    'n_2nd': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\04_norm_2nd',  # * bat_04
    'n_3rd': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\06_norm_3rd',  # * bat_01
    'n_4th': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\08_norm_4th',  # * bat_03
    # 'n_5th': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\10_norm_5th',  # bat_05
    'n_6th': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\12_norm_6th',  # * bat_02
    # 'n_7th': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\14_norm_7th',  # bat_04
}

## SKD61_CUTTER
anomaly_dirname_dic = {
    'us_1st': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\01_us_1st',  # * 途中で折れ bat_01
    'us_2nd': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\03_us_2nd',  # * bat_03
    'us_3rd': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\05_us_3rd',  # bat_05
    'us_4th': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\07_us_4th',  # * 途中で折れ bat_02
    'us_5th': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\09_us_5th',  # bat_04
    'us_6th': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\11_us_6th',  # * bat_01
    'us_7th': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\13_us_7th',  # bat_03
    # 'us_8th': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\15_us_8th',  # bat_01 ５穴目までなので使うな
    'us_9th': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\16_us_9th',  # bat_01
    'us_10th': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\17_us_10th',  # bat_01
    'us_11th': '\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\18_us_11th',  # * 途中で折れ bat_01

}

'''

## SKD61
normal_dirname_dic = {
    # early_0to61
    'n1': ['\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\02_norm_1st', (0, 61)],
    'n2': ['\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\04_norm_2nd', (0, 61)],
    'n3': ['\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\06_norm_3rd', (0, 61)],
    'n4': ['\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\08_norm_4th', (0, 61)],
    'n6': ['\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\12_norm_6th', (0, 61)],
}

# SKD61_CUTTER
anomaly_dirname_dic = {
    'u1': ['\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\01_us_1st', (0, 61)],
    'u2': ['\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\03_us_2nd', (0, 61)],
    'u4': ['\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\07_us_4th', (0, 61)],
    'u6': ['\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\11_us_6th', (0, 61)],
    'u10': ['\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\17_us_10th', (0, 61)],
    'u11': ['\\09_dip_DIR_30db_20V\\058__us_damage_60arr_SKD61\\18_us_11th', (0, 61)],
}

# 使用するファイルのパスをまとめた辞書
NORMAL_FILES_DIC = get_file_list_from_dict(
    normal_dirname_dic,
    path_head=SRC_DIR_HEAD,
    add_dir='/' + DATA_TYPE,
    u_steps=USE_STEPS,
)
ANOMALY_FILES_DIC = get_file_list_from_dict(
    anomaly_dirname_dic,
    path_head=SRC_DIR_HEAD,
    add_dir='/' + DATA_TYPE,
    u_steps=USE_STEPS,
)

print('anomaly num')
for k, v in ANOMALY_FILES_DIC.items():
    print(k, ':', len(v))

# トレーニング用とテスト用を最初に切り分け（このテスト用データは最後の検証以外どのようなトレーニングにも使わない）
NORMAL_FILES_TR_DIC, NORMAL_FILES_TES_DIC = train_test_split_from_dict(
    NORMAL_FILES_DIC,
    test_rate=TEST_SAMPLE_RATE,
    test_picknum=TEST_PICK_NUM,
)

for k, v in NORMAL_FILES_TR_DIC.items():
    print(k, ':', len(v))

# # トレーニング用のデータを減らす
# pick = 2000
# NORMAL_FILES_TR_DIC = {k: random.sample(v, pick) for k, v in NORMAL_FILES_TR_DIC.items()}

for k, v in NORMAL_FILES_TR_DIC.items():
    print(k, ':', len(v))

# グループk-holdの組み合わせ
# normal_files_dic　のkeyを使ってk-holdする
kfold_normal_key_set_dic = {}

for k in NORMAL_FILES_DIC.keys():
    key_list = list(NORMAL_FILES_DIC.keys())
    kfold_normal_key_set_dic[k] = [s for s in key_list if s != k]


def do_train_eval_from_keys(
        train_keys: List[str],
        test_normal_keys: List[str],
        test_anomaly_keys: List[str],
        output_dir_ext: str = '',
        is_train: bool = False,
        save_features: bool = False,
        model_class=AE_CLASS,  # ae class
        model_setting=AE_SETTING,
        model_trainer=AE_TR_TOOL,  # ae model trainer
        model_recons_class=AE_RECONS_CLASS,
        model_eval_class=AE_EVAL_CLASS,
):
    train_name = '-'.join(train_keys)
    model_name = f"{DATA_TYPE}-{train_name}"
    name_params = f'128xcwt_z{Z_DIM}_{AE_NUM_EPOCH}epo_1rot'

    print(model_name)
    print(name_params)

    this_exp_output_dir = f"{OUTPUT_DIR}/{DATA_TYPE}{output_dir_ext}/{train_name}"
    print(this_exp_output_dir)

    # この実験の出力用のディレクトリを生成しておく(AE_model関係)
    os.makedirs(this_exp_output_dir, exist_ok=True)

    output_img_dir = this_exp_output_dir + '/img/'
    os.makedirs(output_img_dir, exist_ok=True)

    output_params_dir = this_exp_output_dir + '/params/'
    os.makedirs(output_params_dir, exist_ok=True)

    '''---------------------------
    dataloader 生成
    ---------------------------'''

    # トレーニング用のファイルリスト
    print('train_keys_list : ', train_keys)
    train_file_list = []
    for k in train_keys:
        train_file_list = train_file_list + NORMAL_FILES_TR_DIC[k]
    print(len(train_file_list))

    ae_tr_dataloader, crf_tr_dataloader = gen_train_dataloader(
        train_file_list,
        clf_rate=CLF_SAMPLE_RATE,
        ae_batch_size=AE_MIN_BATCH,
        setting=DATA_LOADER_SETTING,
    )

    # テスト用の正常データ
    for k in test_normal_keys:
        assert not (k in train_keys), f"train keys should not include test key {k}!"
    normal_test_dict = NORMAL_FILES_TES_DIC
    # テスト用の異常データ
    anomaly_test_dict = ANOMALY_FILES_DIC

    normal_test_dataloader_dict = gen_test_dataloader_dict(
        normal_test_dict,
        setting=DATA_LOADER_SETTING,
        test_picknum=TEST_PICK_NUM
    )
    anomaly_test_dataloader_dict = gen_test_dataloader_dict(
        anomaly_test_dict,
        setting=DATA_LOADER_SETTING,
        test_picknum=TEST_PICK_NUM
    )

    # AEトレーニング
    print(f'ae_train x ({AE_MIN_BATCH}batch)', len(ae_tr_dataloader))  # 個数×バッチサイズ
    print('-------------')

    # 分類器トレーニング
    print('crf_train', len(crf_tr_dataloader))  # 個数×1
    print('-------------')

    # テストデータ
    for k, v in normal_test_dataloader_dict.items():
        print(k, len(v))

    for k, v in anomaly_test_dataloader_dict.items():
        print(k, len(v))

    '''---------------------------
    モデルの動作確認
    ---------------------------'''

    ''' AutoEncoderModel '''
    ae = model_class(**model_setting)

    # summary
    summary(ae.to('cuda:0'), (1, IMG_SIZE, IMG_SIZE))  # summary(model,(channels,H,W))
    ae.to('cpu')  # cpuに戻しておく

    # out_params_dir = 'params'
    # name_loss = "train_loss_" + str(model_name) + f"_{AE_NUM_EPOCH}epo" + ".npy"
    # name_path = "params_" + str(model_name) + f"_{AE_NUM_EPOCH}epo" + ".pth"
    name_loss = f"train_loss_{DATA_TYPE}_{AE_NUM_EPOCH}epo" + ".npy"
    name_path = f"params_{DATA_TYPE}_{AE_NUM_EPOCH}epo" + ".pth"
    print(name_loss)
    print(name_path)

    loss_list = []
    if is_train:
        # モデルのファイルが既に存在するかどうか
        model_is_exist = os.path.exists(output_params_dir + name_path)
        # assert model_is_exist is False, f"model file ({output_params_dir + name_path}) is already exist."
        if model_is_exist:
            msg = f''' --- canceled model training ! ---
            model file ({output_params_dir + name_path}) is already exist.
            if you want to train new model you should delete exist one.
            '''
            print(msg)
        if not model_is_exist:
            # トレーニング実行
            model_trainer(
                ae,
                dataloader=ae_tr_dataloader,
                num_epochs=AE_NUM_EPOCH,
                output_img_dir=output_img_dir,
                loss_list=loss_list,
            )

            # print(loss_list)

            # 学習曲線
            plt.figure()
            # plt.plot(loss_list[10:], 'r-', label='train_loss')
            plt.plot(loss_list[5:], 'r-', label='train_loss')
            # plt.plot(loss_list[1:], 'r-', label='train_loss')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.grid()
            plt.savefig(f"{output_img_dir}/leaning_loss_curve.png")
            plt.close()
            # plt.show()

            # モデルを保存
            np.save(output_params_dir + name_loss, np.array(loss_list))
            torch.save(ae.state_dict(), output_params_dir + name_path)

    print('-------------')

    # print(output_params_dir)
    # print(name_path)

    # モデルの読み込み
    ae.load_state_dict(torch.load(output_params_dir + name_path))
    # for child_name, child_module in ae.named_children():
    #     print(child_name, child_module)

    for normal_k in test_normal_keys:
        output_results_dir = this_exp_output_dir + f"/test_{normal_k}/"
        os.makedirs(output_results_dir, exist_ok=True)
        auc_result_2d_dict = OrderedDict()  # 周波数帯調査用
        auc_result_fl_list = []  # 分類器精度比較用

        for anomaly_k in test_anomaly_keys:
            # 学習済みモデルを用いた検証
            t_nomal_dataloader = normal_test_dataloader_dict[normal_k]
            t_anomaly_dataloader = anomaly_test_dataloader_dict[anomaly_k]

            # 再構成画像の生成テスト(参考資料として再構成データのサンプルを保存しておく)
            # rt = ReconstructTest(ae, output_results_dir, sample_idx=10, add_name=f"_{normal_k}-{anomaly_k}")
            rt = model_recons_class(ae, output_results_dir, sample_idx=TEST_PICK_NUM - 20,
                                    add_name=f"_{normal_k}-{anomaly_k}")
            rt(crf_tr_dataloader, t_nomal_dataloader, t_anomaly_dataloader)

            # 分類器による評価
            em = model_eval_class(ae, normal_k, anomaly_k, output_results_dir)
            # auc_result_list, aucs_hz = em(crf_tr_dataloader, t_nomal_dataloader, t_anomaly_dataloader)
            auc_result_list, aucs_hz, feature_tuple = em(crf_tr_dataloader, t_nomal_dataloader, t_anomaly_dataloader)

            if save_features:
                # save features with .mat file
                train_normal_dict_fl, test_normal_dict_fl, test_anomaly_dict_fl = feature_tuple
                tr_normal_mat_name = f"features_train_normal_{normal_k}-vs-{anomaly_k}_fl.mat"
                tes_normal_mat_name = f"features_test_normal_{normal_k}-vs-{anomaly_k}_fl.mat"
                tes_anomaly_mat_name = f"features_test_anomaly_{normal_k}-vs-{anomaly_k}_fl.mat"
                scipy.io.savemat(output_results_dir + '/' + tr_normal_mat_name, train_normal_dict_fl)
                scipy.io.savemat(output_results_dir + '/' + tes_normal_mat_name, test_normal_dict_fl)
                scipy.io.savemat(output_results_dir + '/' + tes_anomaly_mat_name, test_anomaly_dict_fl)

            # auc_result_fl_list.append(auc_result_dict)
            auc_result_fl_list.extend(auc_result_list)
            auc_result_2d_dict[anomaly_k] = aucs_hz

        # save csv
        auc_df = pd.DataFrame(auc_result_fl_list)
        auc_csv_name = f"{normal_k}_result_auc.csv"
        auc_df.to_csv(output_results_dir + '/' + auc_csv_name)

        auc_2d_df = pd.DataFrame(auc_result_2d_dict)
        auc_2d_csv_name = f"{normal_k}_result_2d_auc.csv"
        auc_2d_df.to_csv(output_results_dir + '/' + auc_2d_csv_name)

        # save png
        auc_2d_df.plot()
        plt.savefig(f"{output_results_dir}/result_2d.png")
        plt.close()

        auc_2d_df.mean(axis=1).plot()
        plt.savefig(f"{output_results_dir}/result_2d_mean.png")
        plt.close()

        gc.collect()

    # 最後に
    del ae
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # この実験結果を出力するディレクトリに文字を追加する
    ae_mode = f"_{AE_TYPE}_{Z_DIM}_{DECODER_OUT}_"
    pre_pro = "_".join(DATA_PREPRO_SETTING)

    this_exp_add = '_e01-02_wax_us_cwt'

    exp_add_name = this_exp_add + ae_mode + f"{AE_NUM_EPOCH}epo_" + pre_pro

    # シード固定
    fix_seed(SEED)

    print('---- k-fold settings ----')
    for k, v in kfold_normal_key_set_dic.items():
        print(k, v)

    kfold_keys = list(kfold_normal_key_set_dic.keys())
    # kfold_keys = ['B_047', ]
    print(kfold_keys)

    kfold_anomaly_keys = list(anomaly_dirname_dic.keys())

    for i, khold_key in enumerate(kfold_keys):
        print(f"----- exp{i} : hold key {khold_key} -----")
        train_keys = kfold_normal_key_set_dic[khold_key]
        test_normal_keys = [khold_key]

        # test_normal_keys = ['n01']
        # test_anomaly_keys = ['us_4th', ]
        # test_anomaly_keys = ['a01', 'a02','a03','a04', 'a05', ]
        # test_anomaly_keys = ['a01', 'a03','a04', 'a05', 'a06']
        # print(f"anomaly_key: {kfold_anomaly_keys[i]} ")
        # test_anomaly_keys = [kfold_anomaly_keys[i]]
        test_anomaly_keys = ['u1', 'u2', 'u4', 'u6', 'u10', 'u11']

        do_train_eval_from_keys(
            train_keys,
            test_normal_keys,
            test_anomaly_keys,
            exp_add_name,
            is_train=IS_TRAIN,
            save_features=SAVE_FEATURES,
        )
