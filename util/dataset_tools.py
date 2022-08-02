import joblib
from typing import Dict, List, Tuple, Optional
import torch
from scipy import ndimage
from torchvision import transforms

import random
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional
import torch
from torchsummary import summary
from sklearn.model_selection import train_test_split

from util.image_data import ArrayTransform, ArrayCompose, NdimageZoom, NdimageResize \
    , Normalize_0to1, ToTorchTensor1ch, GlobalConstractNormalization, Standardize, GCN_simple


class DataTransform2():
    def __init__(
            self,
            img_size: int,
            pre_process: Optional[List] = None,
    ):
        self._img_size = img_size

        self._trf = [NdimageResize([img_size, img_size])]
        # self._trf = []

        if pre_process is not None:
            for m in pre_process:
                self._trf.append(m)

        # # np.ndarrayのサイズを指定した大きさにリサイズ
        # self._trf.append(NdimageResize([img_size, img_size]))

        # torch
        self._trf.append(ToTorchTensor1ch(is_image=True))

        self._transforms = transforms.Compose(self._trf)


    def __call__(self, img):
        return self._transforms(img)


# class DataTransform():
#     """
#     dataの加工
#
#         mean=1.626
#         std=0.362
#     """
#
#     def __init__(
#             self,
#             img_size: int,
#             stdrz: Optional[Tuple[float]] = None,  # (1.626, 0.362) (mean, std)
#             clip: Optional[Tuple[float]] = None,  # (min, max)
#             gcn: Optional[str] = None,
#     ):
#         self._gcn_dict = {
#             'gcn_l1': GlobalConstractNormalization('l1'),
#             'gcn_l2': GlobalConstractNormalization('l2'),
#             'gn_l2': GCN_simple(),
#         }
#         self._img_size = img_size
#         self._stdrz = stdrz
#         self._clip = clip
#         self._gcn = gcn
#         self._gcn_trf = self._gcn_dict.get(self._gcn, None)
#
#         tfs = []
#
#         # standardize
#         if self._stdrz is not None:
#             mean = self._stdrz[0]
#             std = self._stdrz[1]
#             tfs.append(Standardize(mean, std))
#
#         # 指定した数値でクリップして正規化（0~1の値に）
#         if self._clip is not None:
#             n_min = self._clip[0]
#             n_max = self._clip[1]
#             tfs.append(Normalize_0to1(n_min, n_max))
#
#         # GCN or CN
#         if self._gcn_trf is not None:
#             tfs.append(self._gcn_trf)
#
#         # np.ndarrayのサイズを指定した大きさにリサイズ
#         tfs.append(NdimageResize([img_size, img_size]))
#
#         # torch
#         tfs.append(ToTorchTensor1ch(is_image=True))
#
#         self._transforms = transforms.Compose(tfs)
#
#     def __call__(self, img):
#         return self._transforms(img)


def gen_train_dataloader(
        file_list: List[str],
        clf_rate: float,
        ae_batch_size: int = 64,
        setting: Optional[Dict] = None,
):
    """
    トレーニング用のファイルを格納したリストを渡して、AEトレーニングおよび分類器トレーニング用のdataloaderを生成する

    Parameters
    ----------
    file_list
    ae_batch_size
    setting = {
        'img_size': 128,
        'stdrz': (1.626, 0.362),
        'clip' : None,
        'gcn' : None, # 'gcn_l1', 'gn_l2'
    }
    """
    ae_tr_list, clf_tr_list = train_test_split(file_list, test_size=clf_rate)

    print('ae_tr_list', len(ae_tr_list))
    print('crf_tr_list', len(clf_tr_list))

    # dataset
    ae_tr_dataset = SpecDatasetForPytorch(
        file_list=ae_tr_list,
        # transform=DataTransform(**setting)
        transform = DataTransform2(**setting)
    )
    clf_tr_dataset = SpecDatasetForPytorch(
        file_list=clf_tr_list,
        # transform=DataTransform(**setting)
        transform = DataTransform2(**setting)
    )

    # dataloader
    # ae_tr_dataloader は batch_size=AE_MIN_BATCH とする
    ae_tr_dataloader = torch.utils.data.DataLoader(
        ae_tr_dataset, batch_size=ae_batch_size, shuffle=True,
    )
    # 分類器の学習やテストデータの batch_size は 1 とする
    clf_tr_dataloader = torch.utils.data.DataLoader(
        clf_tr_dataset, batch_size=1, shuffle=False,
    )

    return ae_tr_dataloader, clf_tr_dataloader


def gen_test_dataloader_dict(
        test_dict: Dict,
        test_picknum: Optional[int] = None,
        setting: Optional[Dict] = None,
):
    """
    テスト用のファイルを格納したリストを、キーでまとめた辞書型を渡す
    dataloaderはキーでまとめたかたちで辞書型で返す
    Parameters
    ----------
    test_dict
    test_picknum  : データ数を絞るためにランダムピックする場合
    setting = {
        'img_size': 128,
        'stdrz': (1.626, 0.362),
        'clip' : None,
        'gcn' : None, # 'gcn_l1', 'gn_l2'
    }

    """

    if test_picknum is not None:
        test_dict = {k: random.sample(v, test_picknum) for k, v in test_dict.items()}

    # dataset
    test_dataset_dict = {
        k: SpecDatasetForPytorch(
            v,
            # transform=DataTransform(**setting),
            transform=DataTransform2(**setting),
        ) for k, v in test_dict.items()
    }

    # dataloader
    test_dataloader_dict = {
        k: torch.utils.data.DataLoader(
            v, batch_size=1, shuffle=False,
        ) for k, v in test_dataset_dict.items()
    }

    return test_dataloader_dict


class SpecDatasetForPytorch(torch.utils.data.Dataset):
    def __init__(self, file_list, transform):
        self._file_list = file_list
        self._transform = transform

    def __len__(self):
        return len(self._file_list)

    def __getitem__(self, index):
        joblib_path = self._file_list[index]
        with open(joblib_path, mode='rb') as f:
            ds = joblib.load(f)

        spectrum_data = ds.T

        return self._transform(spectrum_data)
