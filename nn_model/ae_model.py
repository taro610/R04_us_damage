import pickle

import functools

import seaborn as sns

import time
import os
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import cv2


def get_activation_by_name(name: str) -> nn.Module:
    """Get activation function by name (string)."""

    activations = {
        "relu": nn.ReLU(),
        "prelu": nn.PReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }

    def error_func(key: str):
        raise ValueError(key, "is not a valid activation function")

    return activations.get(name, error_func(name))


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class AutoEncoderBN(nn.Module):
    def __init__(
            self,
            z_dim: int = 1024,  # number of hidden neurons
            model_activation: Optional[nn.Module] = nn.LeakyReLU(inplace=True),
            encoder_output_activation: Optional[nn.Module] = None,
            decoder_output_activation: Optional[nn.Module] = None,

    ):
        super().__init__()
        # 乱数列の次元
        self._z_dim = z_dim
        self._model_activation = model_activation
        self._encoder_output_activation = encoder_output_activation
        self._decoder_output_activation = decoder_output_activation

        '''
        転置畳込み
        https://dajiro.com/entry/2020/05/24/114113
        Hout = (Hin - 1)S - 2P +FH
        FH: カーネルサイズ
        P: パディング
        S: ストライド

        畳込み
        https://dajiro.com/entry/2020/05/24/114113
        Hout= (Hin+2P-FH)/S + 1
        FH: カーネルサイズ
        P: パディング
        S: ストライド
        '''

        # Build encoder
        # in 128x128
        modules = []
        # bc*128*128*1ch → bc*64*64*16h
        modules.append(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=4, stride=2, padding=1)
        )
        modules.append(nn.BatchNorm2d(16))
        modules.append(self._model_activation)
        # bc*64*64*16ch → bc*32*32*32ch
        modules.append(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=4, stride=2, padding=1)
        )
        modules.append(nn.BatchNorm2d(32))
        modules.append(self._model_activation)
        # bc*32*32*32ch → bc*16*16*64ch
        modules.append(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=4, stride=2, padding=1)
        )
        modules.append(nn.BatchNorm2d(64))
        modules.append(self._model_activation)
        # bc*16*16*64ch → bc*8*8*128ch
        modules.append(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=4, stride=2, padding=1)
        )
        modules.append(nn.BatchNorm2d(128))
        modules.append(self._model_activation)
        # フラットに伸ばして
        modules.append(nn.Flatten())
        modules.append(nn.Linear(8 * 8 * 128, self._z_dim))
        # encoderの最終出力のアクティベーション
        if self._encoder_output_activation is not None:
            modules.append(self._encoder_output_activation)

        self.__encoder = nn.Sequential(*modules)

        # Build decoder
        modules = []
        modules.append(nn.Linear(self._z_dim, 8 * 8 * 128))
        modules.append(nn.BatchNorm1d(8 * 8 * 128))
        modules.append(self._model_activation)
        modules.append(Reshape(-1, 128, 8, 8))
        modules.append(nn.Dropout(p=0.5, inplace=False))
        # bc*8*8*128ch →　bc*16*16*64ch
        modules.append(
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=4, stride=2, padding=1)
        )
        modules.append(nn.BatchNorm2d(64))
        modules.append(self._model_activation)
        # bc*16*16*64h → bc*32*32*32ch
        modules.append(
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=4, stride=2, padding=1)
        )
        modules.append(nn.BatchNorm2d(32))
        modules.append(self._model_activation)
        # bc*32*32*64ch → bc*64*64*32ch
        modules.append(
            nn.ConvTranspose2d(in_channels=32, out_channels=16,
                               kernel_size=4, stride=2, padding=1)
        )
        modules.append(nn.BatchNorm2d(16))
        modules.append(self._model_activation)
        # bc*64*64*16ch → bc*128*128*1ch
        modules.append(
            nn.ConvTranspose2d(in_channels=16, out_channels=1,
                               kernel_size=4, stride=2, padding=1)
        )
        # decoderの最終出力のアクティベーション
        if self._decoder_output_activation is not None:
            modules.append(self._decoder_output_activation)

        self.__decoder = nn.Sequential(*modules)

    @property
    def encoder(self):
        return self.__encoder

    def forward(self, x):
        x = self.__encoder(x)
        x = self.__decoder(x)
        return x


class AutoEncoderDrop(nn.Module):
    def __init__(
            self,
            z_dim: int = 1024,  # number of hidden neurons
            model_activation: Optional[nn.Module] = nn.ReLU(),
            encoder_output_activation: Optional[nn.Module] = None,
            decoder_output_activation: Optional[nn.Module] = nn.Sigmoid(),

    ):
        super().__init__()
        # 乱数列の次元
        self._z_dim = z_dim
        self._model_activation = model_activation
        self._encoder_output_activation = encoder_output_activation
        self._decoder_output_activation = decoder_output_activation

        '''
        転置畳込み
        https://dajiro.com/entry/2020/05/24/114113
        Hout = (Hin - 1)S - 2P +FH
        FH: カーネルサイズ
        P: パディング
        S: ストライド

        畳込み
        https://dajiro.com/entry/2020/05/24/114113
        Hout= (Hin+2P-FH)/S + 1
        FH: カーネルサイズ
        P: パディング
        S: ストライド
        '''

        # Build encoder
        # in 128x128
        modules = []
        # bc*128*128*1ch → bc*64*64*16h
        modules.append(nn.ZeroPad2d((1, 2, 1, 2)))  # l,r,t,b  bc*131*131*1ch
        modules.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2))
        modules.append(self._model_activation)
        # bc*64*64*32ch
        modules.append(nn.ZeroPad2d((1, 2, 1, 2)))  # l,r,t,b  bc*67*67*32ch
        modules.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2))  # bc*32*32*64ch
        modules.append(self._model_activation)
        modules.append(nn.Dropout(0.2))
        # bc*32*32*64ch
        modules.append(nn.ZeroPad2d((1, 2, 1, 2)))  # l,r,t,b  bc*35*35*64ch
        modules.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2))  # bc*16*16*128ch
        modules.append(self._model_activation)
        modules.append(nn.Dropout(0.2))
        # bc*16*16*128ch
        modules.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0))  # bc*7*7*256ch
        modules.append(self._model_activation)
        modules.append(nn.Dropout(0.3))
        # bc*7*7*256ch
        modules.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0))  # bc*3*3*512ch
        modules.append(self._model_activation)
        modules.append(nn.Dropout(0.3))
        # bc*3*3*512ch
        modules.append(nn.Conv2d(512, self._z_dim, kernel_size=3))  # bc*1*1*self._z_dim

        # encoderの最終出力のアクティベーション
        if self._encoder_output_activation is not None:
            modules.append(self._encoder_output_activation)

        self.__encoder = nn.Sequential(*modules)

        # Build decoder
        self._fc_d = nn.Sequential(
            nn.ConvTranspose2d(self._z_dim, 512, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self._conv5d = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self._conv4d = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self._conv3d = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self._conv2d = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
        )
        self._conv1d = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2)
        )

    @property
    def encoder(self):
        return self.__encoder

    def forward(self, x):
        x = self.__encoder(x)
        # decoder
        x = self._fc_d(x)
        x = self._conv5d(x)
        x = self._conv4d(x)
        x = self._conv3d(x)[:, :, 1:-1, 1:-1]
        x = self._conv2d(x)[:, :, 1:-1, 1:-1]
        x = self._conv1d(x)[:, :, 0:-1, 0:-1]
        if self._decoder_output_activation is not None:
            x = self._decoder_output_activation(x)
        return x


class AutoEncoderDrop64(nn.Module):
    def __init__(
            self,
            z_dim: int = 40,  # number of hidden neurons
            # model_activation: Optional[nn.Module] = nn.ReLU(),
            encoder_output_activation: Optional[nn.Module] = None,
            decoder_output_activation: Optional[nn.Module] = nn.Sigmoid(),

    ):
        super().__init__()
        # 乱数列の次元
        self._z_dim = z_dim
        # self._model_activation = model_activation
        self._encoder_output_activation = encoder_output_activation
        self._decoder_output_activation = decoder_output_activation

        '''
        転置畳込み
        https://dajiro.com/entry/2020/05/24/114113
        Hout = (Hin - 1)S - 2P +FH
        FH: カーネルサイズ
        P: パディング
        S: ストライド

        畳込み
        https://dajiro.com/entry/2020/05/24/114113
        Hout= (Hin+2P-FH)/S + 1
        FH: カーネルサイズ
        P: パディング
        S: ストライド
        '''

        # Build encoder
        # in 64x64
        modules = []
        # bc*64*64*1ch → bc*64*64*16h
        modules.append(nn.ZeroPad2d((1, 2, 1, 2)))  # l,r,t,b  bc*131*131*1ch
        modules.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2))
        modules.append(nn.ReLU())
        # bc*64*64*32ch
        modules.append(nn.ZeroPad2d((1, 2, 1, 2)))  # l,r,t,b  bc*67*67*32ch
        modules.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2))  # bc*32*32*64ch
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.2))
        # # bc*32*32*64ch
        # bc*16*16*128ch
        modules.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0))  # bc*7*7*256ch
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.3))
        # bc*7*7*256ch
        modules.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0))  # bc*3*3*512ch
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.3))
        # bc*3*3*512ch
        modules.append(nn.Conv2d(256, self._z_dim, kernel_size=3))  # bc*1*1*self._z_dim

        # encoderの最終出力のアクティベーション
        if self._encoder_output_activation is not None:
            modules.append(self._encoder_output_activation)

        self.__encoder = nn.Sequential(*modules)

        # Build decoder
        self._fc_d = nn.Sequential(
            nn.ConvTranspose2d(self._z_dim, 256, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self._conv4d = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self._conv3d = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self._conv2d = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
        )
        self._conv1d = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2)
        )

    @property
    def encoder(self):
        return self.__encoder

    def forward(self, x):
        x = self.__encoder(x)
        # decoder
        x = self._fc_d(x)
        x = self._conv4d(x)
        x = self._conv3d(x)
        x = self._conv2d(x)[:, :, 1:-1, 1:-1]
        x = self._conv1d(x)[:, :, 0:-1, 0:-1]
        if self._decoder_output_activation is not None:
            x = self._decoder_output_activation(x)
        return x
