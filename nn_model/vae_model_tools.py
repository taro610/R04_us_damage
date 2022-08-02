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

import torch.nn.functional as F


# im_size = 128
# batch_size = 128
# z_size = 512
# vae = VAE(zsize=z_size, layer_count=5)


def loss_function(recon_x, x, mu, logvar):
    BCE = torch.mean((recon_x - x) ** 2)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD * 0.1


# learning_rate = 1e-4
# learning_rate_fin = 1e-6
# # AE_EPOCH_IMAGE_SAVE_N = 100
AE_EPOCH_IMAGE_SAVE_N = 10
#

# num_epoc = 10
# learning_rate = 0.0005
learning_rate = 0.0003


# def vae_loss_function(label, predict, mu, log_var):
#     reconstruction_loss = F.binary_cross_entropy(predict, label, reduction='sum')
#     kl_loss = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#     vae_loss = reconstruction_loss + kl_loss
#     return vae_loss, reconstruction_loss, kl_loss

def vae_train_model(
        model,
        dataloader,
        num_epochs,
        image_dim: Tuple[int, int] = (128, 128),
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
    image_size = image_dim[0] * image_dim[1]

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("使用デバイス：", device)

    # ネットワークをGPUへ
    model.to(device)

    # 追加しておく
    model.train()
    model.weight_init(mean=0, std=0.02)

    # 最適化手法の設定
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=1e-5)

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

        if (epoch + 1) % 8 == 0:
            optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        for i, imges in tqdm(enumerate(dataloader), total=len(dataloader)):
            #         for i, imges in tqdm(enumerate(dataloader)):
            #         for i, imges in enumerate(dataloader):

            model.train()

            # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
            if imges.size()[0] == 1:
                continue
            # print(f'image shape: {imges.shape}')

            # GPUが使えるならGPUにデータを送る
            x = imges.to(device, dtype=torch.float32)
            # x = imges.to(device, dtype=torch.float32)

            # 入力が0-1の範囲という想定で、-1～1に範囲になるよう修正
            # （vaeの最終出力層がtanhで-1~1の出力範囲なので入出力を合わせる）
            x = x * 0.5 - 1.0

            ''' --- forward --- '''
            x_recon, mu, log_var = model(x)
            # print(f'x_recon shape: {x_recon.shape}')

            # 損失関数の計算
            loss_re, loss_kl = loss_function(x_recon, x, mu, log_var)
            loss = loss_re + loss_kl
            # loss, recon_loss, kl_loss = vae_loss_function(x, x_recon, mu, log_var)

            '''recons img [ btch, ch, w, h ]'''
            recon_img = x_recon.view(-1, 1, image_dim[0], image_dim[1])  # tanhから出ているので－1～1だと思う
            # －1～1を0～1の範囲になるように修正
            recon_img = recon_img * 0.5 + 0.5
            # print(f'recon_img shape: {recon_img}')

            ''' --- backward --- '''
            optimizer.zero_grad()
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
                    plt.imshow(recon_img[i][0].cpu().detach().numpy(), 'gray')

                #             plt.savefig("output_img/epoch{}.png".format(epoch))
                plt.savefig(output_img_dir + f"epoch{epoch}.png")
                plt.close()


''' -----------------------------------------  '''


def vae_compute_recons_data(
        model,
        dataloader,
        image_dim: Tuple[int, int] = (128, 128),
        mid_features: List = [],
        pkl_file_path: str = None
):
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
    image_size = image_dim[0] * image_dim[1]

    # 再構成誤差、中間層などを格納する
    in_out_diff = []

    if (pkl_file_path is not None) and (os.path.exists(pkl_file_path)):
        print('load train set feature from: %s' % pkl_file_path)
        with open(pkl_file_path, 'rb') as f:
            in_out_diff = pickle.load(f)
    else:

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
            # x = image.to(device, dtype=torch.float32).view(-1, image_size)
            x = image.to(device, dtype=torch.float32)

            # 入力が0-1の範囲という想定で、-1～1に範囲になるよう修正
            # （vaeの最終出力層がtanhで-1~1の出力範囲なので入出力を合わせる）
            x = x * 0.5 - 1.0

            ''' --- forward --- '''
            # 再構成画像を生成
            with torch.no_grad():  # 計算グラフを作らない
                # x_recon, mu, log_var, z = model(x)
                x_recon, mu, log_var = model(x)
                # resultsample = torch.cat([x, x_recon]) * 0.5 + 0.5
                resultsample = x_recon
                # ch =1
                recon_img = resultsample.view(-1, 1, image_dim[0], image_dim[1])  # tanhから出ているので－1～1だと思う
                # －1～1を0～1の範囲になるように修正
                recon_img = recon_img * 0.5 + 0.5

            # print(x_recon.shape)
            # print(x.shape)
            # print(recon_img.shape)

            loss_img = criterion(recon_img, image)

            org_img = image.cpu().detach().numpy()
            rec_img = recon_img.cpu().detach().numpy()
            diff_img = org_img - rec_img
            loss = loss_img.cpu().detach().numpy()
            mid = mu.cpu().detach().numpy()
            in_out_diff.append([org_img, rec_img, diff_img, loss, mid])

        in_out_diff = np.asarray(in_out_diff)

        if pkl_file_path is not None:
            # # 保存する場合はコメントアウト(大容量になるので保存しない)
            # with open(pkl_file_path, 'wb') as f:
            #     pickle.dump(in_out_diff, f)
            pass

    return in_out_diff
