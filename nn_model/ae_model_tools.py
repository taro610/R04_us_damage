import pickle

import functools

import seaborn as sns

import time
import os
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import cv2
from scipy import ndimage

learning_rate = 1e-4
learning_rate_fin = 1e-6
# AE_EPOCH_IMAGE_SAVE_N = 100
AE_EPOCH_IMAGE_SAVE_N = 10


def func_lr(epoch, total_epoch, start=learning_rate, stop=learning_rate_fin):
    half = total_epoch // 2
    rate = stop / start
    if epoch <= half:
        return 1
    else:
        return epoch * (rate - 1.0) / half + (2.0 - rate)


def ae_train_model(
        model,
        dataloader,
        num_epochs,
        output_img_dir='./',
        loss_list: List = [],
):
    """

    Parameters
    ----------
    model
    dataloader
    num_epochs
    output_img_dir
    loss_list : 損失の推移を確認（格納）するためのリストを渡しておくこと

    Returns
    -------

    """
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("使用デバイス：", device)

    # ネットワークをGPUへ
    model.to(device)

    # 最適化手法の設定
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 誤差関数を定義
    criterion = nn.MSELoss()  # 画像の場合大体MeanSquaredErrorを用いるらしい

    # scheduler
    lr_f = functools.partial(func_lr, total_epoch=num_epochs)
    # scheduler = LambdaLR(optimizer, lr_lambda=func_lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_f)

    # ネットワークがある程度固定であれば高速化される
    torch.backends.cudnn.benchmark = True

    batch_size = dataloader.batch_size

    # epochのループ
    for epoch in range(num_epochs):
        losses = []

        # 開始時間を保存
        t_epoch_start = time.time()
        epoch_loss = 0.0

        print('-------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-------------')
        print('（train）')

        # 追加しておく
        model.train()

        for i, imges in tqdm(enumerate(dataloader), total=len(dataloader)):
            #         for i, imges in tqdm(enumerate(dataloader)):
            #         for i, imges in enumerate(dataloader):
            # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
            if imges.size()[0] == 1:
                continue

            # GPUが使えるならGPUにデータを送る
            imges = imges.to(device, dtype=torch.float)

            ''' --- forward --- '''
            # 再構成画像を生成
            output = model(imges)

            ''' --- backward --- '''
            optimizer.zero_grad()
            loss = criterion(output, imges)
            #             ssim_value = ssim_out.item()
            loss.backward()
            optimizer.step()

            ''' ---記録 --- '''
            epoch_loss += loss.item()
            losses.append(loss.cpu().detach().numpy())

        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_Loss:{:.4f}'.format(epoch, epoch_loss / batch_size))
        print('epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, num_epochs, np.average(losses)))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        scheduler.step()
        loss_list.append(np.average(losses))
        t_epoch_start = time.time()

        ''' -------------------------
        動作確認
        -------------------------- '''
        if (epoch % AE_EPOCH_IMAGE_SAVE_N == 0) or (epoch == 10) or (epoch == 30):
            batch_s = 5

            with sns.axes_style("white"):  # imshowのときに白いグリッドが入るのを避ける
                fig = plt.figure(figsize=(15, 6))
                for i in range(0, batch_s):
                    # 上の段に訓練データ
                    plt.subplot(2, 5, i + 1)
                    plt.imshow(imges[i][0].cpu().detach().numpy(), 'gray')
                    # 下の段に生成データ
                    plt.subplot(2, 5, 5 + i + 1)
                    plt.imshow(output[i][0].cpu().detach().numpy(), 'gray')

                #             plt.savefig("output_img/epoch{}.png".format(epoch))
                plt.savefig(output_img_dir + f"epoch{epoch}.png")
                plt.close()


''' -----------------------------------------  '''


def ae_compute_recons_data(model, dataloader, mid_features: List = [], pkl_file_path: str = None):
    """
    学習済みのAEモデルを使って、再構成画像や中間層の出力などを算出する
    Parameters
    ----------
    model : AE model
    dataloader : pytorch dataloader
    mid_features : latant featers strage (潜在空間を格納するためのリストを渡しておくこと)
    pkl_file_path : recons data (np.ndarray) -> save pkl file

    Returns
    -------
    np.ndarray[
        org_img,
        rec_img,
        diff_img,
        mse,
        mid
    ]
    """
    # 再構成誤差、中間層などを格納する
    in_out_diff = []

    if (pkl_file_path is not None) and (os.path.exists(pkl_file_path)):
        print('load train set feature from: %s' % pkl_file_path)
        with open(pkl_file_path, 'rb') as f:
            in_out_diff = pickle.load(f)
    else:
        def forward_hook(module, inputs, outputs):
            # 順伝搬の出力を features というグローバル変数に記録する

            # 1. detach でグラフから切り離す。
            # 2. clone() でテンソルを複製する。モデルのレイヤーで ReLU(inplace=True) のように
            #    inplace で行う層があると、値がその後のレイヤーで書き換えられてまい、
            #    指定した層の出力が取得できない可能性があるため、clone() が必要。
            out = outputs.detach().clone()
            #             print(type(out))
            #             print(type(out.cpu().detach().numpy()))
            #             assert False, ''
            mid_features.append(out.cpu().detach().numpy())

        # コールバック関数を登録する。
        # handle_1 = model._AutoEncoder__encoder[-1].register_forward_hook(forward_hook)
        handle_1 = model.encoder[-1].register_forward_hook(forward_hook)  # model にencoderというgetterがある前提

        # GPUが使えるかを確認
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("使用デバイス：", device)

        # ネットワークをGPUへ
        model.to(device)

        model.eval()

        # 誤差関数を定義
        criterion = nn.MSELoss()

        for image in tqdm(dataloader, '| feature extraction | train |'):
            # GPUが使えるならGPUにデータを送る
            image = image.to(device, dtype=torch.float)

            ''' --- forward --- '''
            # 再構成画像を生成
            with torch.no_grad():  # 計算グラフを作らない
                output = model(image)
            loss_normal = criterion(output, image)

            org_img = image.cpu().detach().numpy()
            rec_img = output.cpu().detach().numpy()
            diff_img = org_img - rec_img
            mse = loss_normal.cpu().detach().numpy()
            mid = np.asarray(mid_features)
            in_out_diff.append([org_img, rec_img, diff_img, mse, mid])

            # 中間層のプレースホルダをクリア
            mid_features = []

        in_out_diff = np.asarray(in_out_diff)

        if pkl_file_path is not None:
            # # 保存する場合はコメントアウト(大容量になるので保存しない)
            # with open(pkl_file_path, 'wb') as f:
            #     pickle.dump(in_out_diff, f)
            pass

        # コールバック関数を解除する。
        handle_1.remove()

    return in_out_diff


