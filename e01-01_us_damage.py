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
# DATA_TYPE = 'CWTx128_1rot_df10kh_olog_aaf_zc'  # 0-60
# DATA_TYPE = 'CWTx128_1rot_df100k-1m_olog_aaf'  # ujbB
DATA_TYPE = 'CWTx128_1rot_df100k-1m_flog_olog_aaf'  # ujbB

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
# STDRZ = (1.849262, 0.4512)  # 'CWTx128_1rot_df10kh_olog' 60 ana <- matigattene?
# STDRZ = (1.7529, 0.4258)  # 'CWTx128_1rot_df10kh_olog_aaf_zc' 60 ana
# STDRZ = (1.6890, 0.4047)  # 'CWTx128_1rot_df10kh_olog_aaf_zc'
# STDRZ = (1.7082, 0.4172)  # 'CWTx128_1rot_df10kh_olog_aaf_zc' waxmaunt
# STDRZ = (1.6890, 0.4047)  # 'CWTx128_1rot_df10kh_olog_aaf_zc'
# STDRZ = (1.5626, 0.3005)  # 'CWTx128_1rot_df200kh_flog_olog_aaf_zc'
# STDRZ = (1.7697, 0.3710)  # 'flog_CWTx128_1rot_df100k-1m_flog_olog_aaf_zc' 60 ana

''' 0-1 clip '''
# CLIP = None
# CLIP = (-2.2657, 5.2463)  # 'CWTx128_1rot_df10kh_olog' raw -> 60 ana raw -> 0-1clip
# CLIP = (-2.6219, 5.1340)  # 'CWTx128_1rot_df100k-1m_olog_aaf' ujbB raw -> 0-1clip
CLIP = (-2.8680, 5.1548)  # 'CWTx128_1rot_df100k-1m_flog_olog_aaf' ujbB raw -> 0-1clip
# CLIP = (-2.2904, 5.18252)  # 'CWTx128_1rot_df10kh_olog_aff_zc' raw -> 60 ana raw -> 0-1clip
# CLIP = (-2.53928, 5.3828)  # 'CWTx128_1rot_df10kh_olog_aaf_zc' raw -> 0-1clip
# CLIP = (-2.7536, 5.2146)  # 'CWTx128_1rot_df100k-1m_flog_olog_aaf_zc' raw -> 0-1clip
# CLIP = (-2.2687, 4.8723)  # 'CWTx128_1rot_df100k-1m_flog_olog_aaf_zc' raw -> 0-1clip 60ana
# CLIP = (-2.5225, 4.8540)  # 'CWTx128_1rot_df200kh_flog_olog_aaf_zc' raw -> 0-1clip
# CLIP = (-2.8408, 4.7027)  # 'CWTx128_1rot_df300k-1m_olog_aaf_zc' raw -> 0-1clip
# CLIP = (-2.7746, 5.1962)  # 'CWTx128_1rot_df100k-1m_olog_aaf_zc' raw -> 0-1clip
# CLIP = (-2.7536, 5.2146)  # 'CWTx128_1rot_df100k-1m_flog_olog_aaf_zc' raw -> 0-1clip

# CLIP = (-10.4485, 9.1268)  # 'CWTx128_1rot_df10kh_olog_aaf_zc' stdrz -> 0-1clip
# CLIP = (-13.5939, 9.4151)  # 'CWTx128_1rot_df200kh_flog_olog_aaf_zc' stdrz -> 0-1clip

# CLIP = (-9.9982, 6.7883)  # 'CWTx128_1rot_df10kh_olog' 60ana GCNL2 -> 0-1clip
# CLIP = (-10.609, 6.6211)  # 'CWTx128_1rot_df10kh_olog_aaf_zc' waxmaunt GCNL2 -> 0-1clip
# CLIP = (-11.779, 6.1128)  # 'CWTx128_1rot_df100k-1m_flog_olog_aaf_zc' GCNL2 -> 0-1clip 60ana

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
# Z_DIM = 1024
Z_DIM = 128
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
# 加工初期
early_first_target_setting = {
    'n01': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (0, 61)],
    'n02': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (0, 61)],
    'n03': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (0, 61)],
    'n04': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (0, 61)],
    'n05': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (0, 61)],
    'n06': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61', (0, 61)],
}

early_60_target_setting = {
    # early_60to120
    'n01_046': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (60, 121)],
    'n02_047': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (60, 121)],
    'n03_050': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (60, 121)],
    'n04_052': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (60, 121)],
    'n05_053': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (60, 121)],
    'n06_051': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61',(60, 121)],
}

early_120_target_setting = {
    'n01_046': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (120, 181)],
    'n02_047': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (120, 181)],
    'n03_050': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (120, 181)],
    'n04_052': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (120, 181)],
    'n05_053': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (120, 181)],
    'n06_051': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61',(120, 181)],
}

early_180_target_setting = {
    'n01_046': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (180, 241)],
    'n02_047': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (180, 241)],
    'n03_050': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (180, 241)],
    'n04_052': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (180, 241)],
    'n05_053': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (180, 241)],
    'n06_051': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61',(180, 241)],
}

middle_1_target_setting = {
    'n01_046': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (300, 361)], 
    'n02_047': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (600, 661)],
    'n03_050': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (2100, 2161)],
    'n04_052': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (240, 301)],
    'n05_053': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (240, 301)],
    'n06_051': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61',(480, 561)],
}

middle_2_target_setting = {
    'n01_046': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (840, 901)], 
    'n02_047': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (1800, 1861)],
    'n03_050': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (3960, 4021)],
    'n04_052': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (720, 781)],
    'n05_053': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (600, 661)],
    'n06_051': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61',(960, 1021)],
}

late_before_10_target_setting = {
    # late_before_10_target_setting
    'n01_046': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (1210, 1271)], 
    'n02_047': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (2832, 2893)],
    'n03_050': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (5827, 5888)],
    'n04_052': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (946, 1007)],
    'n05_053': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (911, 972)],
    'n06_051': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61',(1633, 1694)],
}


late_before_5_target_setting = {
    'n01_046': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (1276, 1281)], 
    'n02_047': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (2898, 2903)],
    'n03_050': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (5893, 5898)],
    'n04_052': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (1012, 1017)],
    'n05_053': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (977, 982)],
    'n06_051': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61',(1699, 1704)],
}


    # early_60to180
    'n01': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (61, 181)],
    'n02': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (61, 181)],
    'n03': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (61, 181)],
    'n04': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (61, 181)],
    'n05': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (61, 181)],
    # 'n06': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61', (61, 181)],
    
    # late_before_-60to-10_target_setting
    'a01': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (1210, 1271)],
    'a02': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (2832, 2893)],
    'a03': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (5827, 5888)],
    'a04': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (946, 1007)],
    'a05': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (911, 972)],
    # 'a06': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61', (1633, 1694)], 
    
    
    # late_before_60_target_setting
    'a01': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (1220, 1281)],
    'a02': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (2842, 2903)],
    'a03': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (5837, 5898)],
    'a04': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (956, 1017)],
    'a05': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (921, 982)],
    'a06': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61',(1643, 1704)],
    
    # late_before_-5to-0_target_setting
    'a01': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (1276, 1281)], 
    'a02': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (2898, 2903)],
    'a03': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (5893, 5898)],
    'a04': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (1012, 1017)],
    'a05': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (977, 982)],
    'a06': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61',(1699, 1704)],
    
    # mid_target_setting
    'a01': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (241, 361,)], 
#     'a02': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (421, 541,)],
    'a03': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (3201, 3321)],
    'a04': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (301, 421)],
    'a05': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (301, 421)],
    'a06': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61',(841, 961)],

 # last_target_setting(test=30)
    'a03': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (5897, 5899)],
    'a01': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (1281, 1283,)],
    #     'a02': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (2903, 2904,)],
    'a04': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (1016, 1018)],
    'a05': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (981, 983)],
    'a06': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61', (1703, 1705)],

'''

## SKD61
normal_dirname_dic = {
    # early_0to61
    'n03': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (0, 61)],
    'n01': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (0, 61)],
    # 'n02': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (0, 61)],
    'n04': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (0, 61)],
    'n05': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (0, 61)],
    'n06': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61', (0, 61)],
}

# SKD61_CUTTER
anomaly_dirname_dic = {
    # mid_target_setting
    # 'dam': ['\\07_dip_DIR_30dB_20V\\999_dummy', (0, 61)],
    'a03': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (3201, 3321)],
    'a01': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (241, 361,)],
    #     'a02': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (421, 541,)],
    'a04': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (301, 421)],
    'a05': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (301, 421)],
    'a06': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61', (841, 961)],
}

# ## SKD61_CUTTER
# anomaly_dirname_dic = {
#     # late_before_5(test=30)
#     'a03': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (5894, 5899)],
#     'a01': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (1278, 1283,)],
#     #     'a02': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (2899, 2904,)],
#     'a04': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (1013, 1018)],
#     'a05': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (978, 983)],
#     'a06': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61', (1700, 1705)],
# }


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

    # this_exp_add = '_e01-01_cwt_60to180vs-60to-10'
    # this_exp_add = '_e01-01_cwt_0to60vs-60to-0'
    # this_exp_add = '_e01-01_cwt_0to60vs-60to-10'
    # this_exp_add = '_e01-01_cwt_0to60vs-5to-0'
    this_exp_add = '_e01-01_cwt_0to60vsMid'

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
        print(f"anomaly_key: {kfold_anomaly_keys[i]} ")
        test_anomaly_keys = [kfold_anomaly_keys[i]]
        # test_anomaly_keys = ['dam']

        do_train_eval_from_keys(
            train_keys,
            test_normal_keys,
            test_anomaly_keys,
            exp_add_name,
            is_train=IS_TRAIN,
            save_features=SAVE_FEATURES,
        )
