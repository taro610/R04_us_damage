import seaborn as sns
import random
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split

from util.path_tools import GetFileListBySuffix
from util.path_tools import include_list_picker_strs, ana_count_select_picker


def gen_dir_path(dir_name: str, path_head: str, add_dir: str):
    src_dir = path_head + dir_name + add_dir
    return src_dir


def joblib_file_list(
        src_dir: str,
        is_recursive: bool = False,
):
    get_fl = GetFileListBySuffix(['.joblib'], recursive=is_recursive)
    file_list = get_fl(src_dir)
    return file_list


def list_filter(
        file_list: List[str],
        u_steps: List[str],  # e.g. ['_st3.0', '_st4.0', '_st5.0']
        ana_range: Tuple[int, int] = (0, 9999),  # 穴数の範囲を指定(min, max)
):
    fl_file_list = include_list_picker_strs(file_list, u_steps)
    fl_file_list = ana_count_select_picker(fl_file_list, ana_range[0], ana_range[1])
    return fl_file_list


def gen_file_list(
        dir_name: str,
        path_head: str,  # e.g. 'I:\\experimental_data\\110_processed_data'
        add_dir: str,  # e.g.  '/' + 'CWTx128_1rot_df50kh_olog_aaf_zc'
        u_steps: List[str],  # e.g. ['_st3.0', '_st4.0', '_st5.0', '_st6.0', '_st7.0']
        ana_range: Tuple[int, int] = (0, 9999),  # 穴数の範囲を指定(min, max)
):
    dir_path = gen_dir_path(dir_name, path_head, add_dir)
    jl_files = joblib_file_list(dir_path, is_recursive=False)
    files = list_filter(jl_files, u_steps, ana_range)
    return files


def get_file_list_from_dict(
        src_setting_dic: Dict,
        path_head: str,  # e.g. 'I:\\experimental_data\\110_processed_data'
        add_dir: str,  # e.g.  '/' + 'CWTx128_1rot_df50kh_olog_aaf_zc'
        u_steps: List[str],  # e.g. ['_st3.0', '_st4.0', '_st5.0', '_st6.0', '_st7.0']
):
    """
    src_setting_dic = {
        'n01_046': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (0, 61)],
        'n02_047': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (0, 61)],
        'n03_050': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (0, 61)],
        'n04_052': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (0, 61)],
        'n05_053': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (0, 61)],
        'n06_051': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61', (0, 61)],
    }

    :param src_setting_dic:
    :param path_head:
    :param add_dir:
    :param u_steps:
    :return:
    """
    files_dic = {}
    for k, v in src_setting_dic.items():
        src_dir, ana_range = v
        files = gen_file_list(
            dir_name=src_dir,
            path_head=path_head,
            add_dir=add_dir,
            u_steps=u_steps,
            ana_range=ana_range,
        )
        files_dic[k] = files
    return files_dic


'''-------------------------------------'''

# def dir_path_from_dict(dirname_dic: Dict, path_head: str, add_dir: str):
#     dir_path_dic = {}
#     for k, v in dirname_dic.items():
#         src_dir = path_head + v + add_dir
#         dir_path_dic[k] = src_dir
#     return dir_path_dic
#
#
# def joblib_file_list_from_dict(
#         dir_dic: Dict,
# ):
#     get_fl_recursive = GetFileListBySuffix(['.joblib'], recursive=True)
#     file_list_dic = {k: get_fl_recursive(v) for k, v in dir_dic.items()}
#     return file_list_dic
#
#
# def list_filter_from_dict(
#         file_list_dic: Dict,
#         u_steps: List[str],
#         ana_range: Tuple[int, int],  # 穴数の範囲を指定(min, max)
# ):
#     filtered_dic = {}
#     for k, v in file_list_dic.items():
#         # stepで選別しておく
#         file_list = include_list_picker_strs(v, u_steps)
#         # 穴数の範囲で選別しておく
#         filtered_dic[k] = ana_count_select_picker(file_list, ana_range[0], ana_range[1])
#     return filtered_dic
#
#
# def get_file_list_from_dict(
#         dirname_dic: Dict,
#         path_head: str,  # e.g. 'I:\\experimental_data\\110_processed_data'
#         add_dir: str,  # e.g.  '/' + 'CWTx128_1rot_df50kh_olog_aaf_zc'
#         u_steps: List[str],  # e.g. ['_st3.0', '_st4.0', '_st5.0', '_st6.0', '_st7.0']
#         ana_range: Tuple[int, int] = (0, 60),  # 穴数の範囲を指定(min, max)
# ):
#     dir_dic = dir_path_from_dict(dirname_dic, path_head, add_dir)
#     file_dic = joblib_file_list_from_dict(dir_dic)
#     return list_filter_from_dict(file_dic, u_steps, ana_range)


'''-------------------------------------'''


def train_test_split_from_dict(
        file_dic: Dict,
        test_rate: float,  # e.g. 0.1
        test_picknum: Optional[int] = None,  # e.g. 300
):
    train_list_dic = {}
    test_list_dict = {}
    for k, v in file_dic.items():
        tr_list, tes_list = train_test_split(v, test_size=test_rate)
        if test_picknum is not None:
            assert len(tes_list) >= test_picknum, f'test data pick errer: {len(tes_list)} >= {test_picknum} '
            tes_list = random.sample(tes_list, test_picknum)
        train_list_dic[k] = tr_list
        test_list_dict[k] = tes_list
    return train_list_dic, test_list_dict


if __name__ == '__main__':

    ## SKD61
    normal_dirname_dic = {
        'n01_046': ['\\07_dip_DIR_30dB_20V\\046_20211202_1282deore_60arr_SKD61', (0, 61)],
        'n02_047': ['\\07_dip_DIR_30dB_20V\\047_20211202_2903deore_60arr_SKD61', (0, 61)],
        'n03_050': ['\\07_dip_DIR_30dB_20V\\050_20211209_5898deire_60arr_SKD61', (0, 61)],
        'n04_052': ['\\07_dip_DIR_30dB_20V\\052_20220111_1021deore_60arr_SKD61', (0, 61)],
        'n05_053': ['\\07_dip_DIR_30dB_20V\\053_20220117_982deore_60arr_SKD61', (0, 61)],
        'n06_051': ['\\07_dip_DIR_30dB_20V\\051_20211221_1704deore_60arr_SKD61', (0, 61)],
    }

    # 解析対象データ
    DATA_TYPE = 'CWTx128_1rot_df10kh_olog'
    SRC_DIR_HEAD = 'I:\\experimental_data\\110_processed_data'
    USE_STEPS = ['_st3.0', '_st4.0', '_st5.0', '_st6.0', '_st7.0']  # これを使うときは折損直前の穴の6～7step目を要確認
    OUTPUT_DIR = './outputs'

    # 使用するファイルのパスをまとめた辞書
    NORMAL_FILES_DIC = get_file_list_from_dict(
        normal_dirname_dic,
        path_head=SRC_DIR_HEAD,
        add_dir='/' + DATA_TYPE,
        u_steps=USE_STEPS,
    )

    TEST_SAMPLE_RATE = 0.1
    TEST_PICK_NUM = 300  # テスト用データを絞るためにランダムピックする場合

    # トレーニング用とテスト用を最初に切り分け（このテスト用データは最後の検証以外どのようなトレーニングにも使わない）
    NORMAL_FILES_TR_DIC, NORMAL_FILES_TES_DIC = train_test_split_from_dict(
        NORMAL_FILES_DIC,
        test_rate=TEST_SAMPLE_RATE,
        test_picknum=TEST_PICK_NUM,
    )

    for k, v in NORMAL_FILES_TR_DIC.items():
        print(k, ':', len(v))

    # トレーニング用のデータを減らす
    pick = 2000
    NORMAL_FILES_TR_DIC = {k: random.sample(v, pick) for k, v in NORMAL_FILES_TR_DIC.items()}

    for k, v in NORMAL_FILES_TR_DIC.items():
        print(k, ':', len(v))

    # グループk-holdの組み合わせ
    # normal_files_dic　のkeyを使ってk-holdする
    kfold_normal_key_set_dic = {}

    for k in NORMAL_FILES_DIC.keys():
        key_list = list(NORMAL_FILES_DIC.keys())
        kfold_normal_key_set_dic[k] = [s for s in key_list if s != k]
