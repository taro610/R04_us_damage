from abc import ABCMeta, abstractmethod

import numpy
from scipy import ndimage
import numpy
from typing import Optional, Callable, Union, Sequence, Dict, Tuple, List

import torch


class ArrayTransform(metaclass=ABCMeta):
    """
    画像データ等の多次元numpy.ndarray加工用
    """

    @abstractmethod
    def __call__(self, arr: numpy.ndarray) -> numpy.ndarray:
        pass


# class Float2Uint(ArrayTransform):
#     def __call__(self, data: numpy.ndarray, **kwargs) -> numpy.ndarray:
#         return data.astype(numpy.uint8)
#
#
# class Uint2Float(ArrayTransform):
#     def __call__(self, data: numpy.ndarray, **kwargs) -> numpy.ndarray:
#         return data.astype(numpy.float32)


class ArrayCompose(ArrayTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        > transforms.Compose([
        >     transforms.CenterCrop(10),
        >     transforms.ToTensor(),
        > ])
    """

    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, data, **kwargs):
        for t in self._transforms:
            data = t(data, **kwargs)

        return data


class NdimageZoom(ArrayTransform):
    def __init__(self, zoom: Union[float, Sequence[float]]):
        """
        適用するズームの倍率を指定してscipy.ndimage()でサイズを変更する
        """
        self._zoom = zoom

    def __call__(self, arr: numpy.ndarray) -> numpy.ndarray:
        array = ndimage.zoom(arr, self._zoom)
        return array


class NdimageResize(ArrayTransform):
    def __init__(self, size: Union[int, Sequence[int]]):
        """
        変更後のサイズを指定することで、ズーム倍率を計算して処理する
        計算したズームの倍率を用いてscipy.ndimage()でサイズを変更する
        """
        self._size = size

    def __call__(self, array: numpy.ndarray) -> numpy.ndarray:
        zoom_seq = self.get_zoom_seq(array)
        array = ndimage.zoom(array, zoom_seq)
        return array

    def get_zoom_seq(self, arr: numpy.ndarray) -> Sequence:
        dims = arr.ndim
        shape = arr.shape

        size_seq = self._size

        if not isinstance(size_seq, Sequence):
            size_seq = [size_seq] * dims

        zoom_seq = [x / y for (x, y) in zip(size_seq, shape)]

        return zoom_seq


class Standardize(ArrayTransform):
    def __init__(self, mean, std):
        """
            Normalize to range from [min, max] to [0, 1] based on dataset quick stat check.
        """
        self._mean = mean
        self._std = std

    def __call__(self, arr: numpy.ndarray) -> numpy.ndarray:
        array = (arr - self._mean) / self._std
        return array


class Normalize_0to1(ArrayTransform):
    def __init__(self, min_clip, max_clip):
        """
            Normalize to range from [min, max] to [0, 1] based on dataset quick stat check.
        """
        self._min = min_clip
        self._max = max_clip

    def __call__(self, arr: numpy.ndarray) -> numpy.ndarray:
        # Normalize to range from [min, max] to [0, 1] based on dataset quick stat check.
        array = (arr - self._min) / (self._max - self._min)
        array = numpy.clip(array, 0., 1.)
        return array


class ToTorchTensor1ch(object):
    def __init__(self, device=None, is_image=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.non_batch_shape_len = 2 if is_image else 1

    def __call__(self, array: numpy.ndarray):
        # (dims)
        if len(array.shape) == self.non_batch_shape_len:
            return torch.Tensor(array).unsqueeze(0).to(self.device)
        # (batch, dims)
        return torch.Tensor(array).unsqueeze(1).to(self.device)


class GCN_simple(ArrayTransform):
    '''
    Contrast Normalization

    https://qiita.com/dsanno/items/ad84f078520f9c9c3ed1
    '''

    def __call__(self, arr: numpy.ndarray) -> numpy.ndarray:
        mean = numpy.mean(arr)
        std = numpy.std(arr)

        arr = (arr - mean) / std

        return arr


class GCN_simple_2(ArrayTransform):
    '''
    https://datascience.stackexchange.com/questions/15110/how-to-implement-global-contrast-normalization-in-python
    https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/preprocessing.py
    '''

    def __init__(self, s=1.0, sqrt_bias=0., epsilon=1e-8):
        self._s = s
        self._sqrt_bias = sqrt_bias
        self._epsilon = epsilon

    def __call__(self, X: numpy.ndarray) -> numpy.ndarray:
        # replacement for the loop
        X_average = numpy.mean(X)
        X = X - X_average

        # `su` is here the mean, instead of the sum
        contrast = numpy.sqrt(self._sqrt_bias + numpy.mean(X ** 2))

        X = self._s * X / max(contrast, self._epsilon)
        return X


class GlobalConstractNormalization(ArrayTransform):
    def __init__(self, scale='l2'):
        """
        これは各ファイル単位でNormalizationする処理だと思う

            Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
            which is either the standard deviation, L1- or L2-norm across features (pixels).
            Note this is a *per sample* normalization globally across features (and not across the dataset).

            Yoshua Bengio's Deep Learning book (section 12.2.1.1 pg. 442)
            https://www.deeplearningbook.org/

        """
        assert scale in ('l1', 'l2')
        self._scale = scale

    def __call__(self, arr: numpy.ndarray) -> numpy.ndarray:

        n_features = int(numpy.prod(arr.shape))  # shapeの配列要素積

        mean = numpy.mean(arr)  # mean over all features (pixels) per sample
        arr -= mean

        if self._scale == 'l1':
            x_scale = numpy.mean(numpy.abs(arr))

        if self._scale == 'l2':
            x_scale = numpy.sqrt(numpy.sum(arr ** 2)) / n_features

        arr /= x_scale

        return arr
