import re
from typing import List, Union, Dict, Tuple, Optional

import pathlib
from pathlib import Path

from functools import partial
from abc import ABCMeta, abstractmethod


# ********************


def include_list_picker(txt_list: List[str], include: str):
    """
    文字列を格納したリストを渡し、指定した文字が含まれる要素のみピックする
    """
    return [t for t in txt_list if include in t]


def ana_count_head_gen(start: int, stop: int):
    """
    こんなリストを作る
        ['ac1_', 'ac2_', 'ac3_']
    """
    return ["ac{}_".format(i) for i in range(start, stop + 1)]


def ana_count_select_picker(txt_list: List[str], start: int, stop: int):
    """
    指定した範囲の['ac{start}_'～'ac{stop}_']の文字を含む要素だけ取り出す
    """
    pick_list = []
    include_str = ana_count_head_gen(start, stop)

    for s in include_str:
        pick_list.extend(include_list_picker(txt_list, s))

    return pick_list


def include_list_picker_strs(txt_list: List[str], include_list: List[str]):
    pick_list = []
    for s in include_list:
        picked = [t for t in txt_list if s in t]
        pick_list.extend(picked)
    return pick_list


# ********************


class PathTools(object):
    '''
    指定したディレクトリもしくはファイルが存在するか確認して、そのPathオブジェクトを生成する機能を有するクラス
    '''

    @staticmethod
    def dir_path_obj(dir_path: str) -> Path:
        '''
        dir_path : 解析対象とするディレクトリの存在を確認し、存在する場合はPathオブジェクトを生成
        '''
        assert dir_path, "Need to specify the directory path. It is currently empty."

        # Pathオブジェクトを生成
        p_dir = Path(dir_path)

        assert p_dir.is_dir(), "The directory '{0}' does not exist!".format(str(p_dir))

        return p_dir

    @staticmethod
    def file_path_obj(file_path: str) -> Path:
        assert file_path, "Need to specify the file path. It is currently empty."

        # Pathオブジェクトを生成
        p_file = Path(file_path)

        assert p_file.is_file(), "The file '{0}' does not exist!".format(str(p_file))

        return p_file

    @staticmethod
    def paths_sort(paths: List[pathlib.Path]):
        #         return sorted(paths, key = lambda x: int(x.name))
        return sorted(paths, key=lambda x: x.name)


# ********************

class PathAnalyzer(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, **kwargs):
        pass


class PickupHoleProcessingDataList(PathAnalyzer):
    """
    解析対象となるディレクトリはcall()時にファイルパスを与えて指定する。
    その中から、file_suffixで指定した拡張子のモノだけが解析対象となる

    'I:\\experimental_data\\100_MCkakou\\20211208-163640-083421_ac2_st2.0_zp-1.980.hdf5'
     のようなファイルパスの文字列の中に指定した文字があるものだけが取り出される

    フィルタリングに用いる以下2つのリストに関しては数値のリストを与えてこのクラス内で生成する
    hole_num_list->['ac1_', 'ac2_', 'ac3_']
    step_num_list->['_st3.0', '_st4.0']

    その他にフィルタリング用の文字を与えたい場合はkey_wardsで指定する

    """

    def __init__(
            self,
            file_suffix: Union[str, List[str]],
            hole_num_list: Optional[List[int]] = None,
            step_num_list: Optional[List[int]] = None,
            key_wards: Optional[Union[str, List[str]]] = None,
            recursive=False,
    ):
        self._file_suffix = file_suffix
        self._file_picker = GetFileListBySuffix(self._file_suffix, recursive=recursive)
        self._key_wards_list = key_wards
        self._hole_wards_list = [] if hole_num_list is None else self.gen_hole_wards_list(hole_num_list)
        self._step_wards_list = [] if step_num_list is None else self.gen_step_wards_list(step_num_list)

    @property
    def key_wards_list(self) -> Optional[List[str]]:
        return self._key_wards_list

    @property
    def hole_wards_list(self) -> List[str]:
        return self._hole_wards_list

    @property
    def step_wards_list(self) -> List[str]:
        return self._step_wards_list

    @staticmethod
    def pick_by_include_str(txt_list: List[str], str_list: Union[str, List[str]]):
        if type(str_list) is str:
            str_list = [str_list]

        pick_list = []
        for s in str_list:
            picked = [t for t in txt_list if s in t]
            pick_list.extend(picked)
        return pick_list

    @staticmethod
    def gen_hole_wards_list(num_list: List[int]):
        """
        こんなリストを作る
        ['ac1_', 'ac2_', 'ac3_']
        """
        return ["ac{}_".format(i) for i in num_list]

    @staticmethod
    def gen_step_wards_list(num_list: List[int]):
        """
        こんなリストを作る
        ['_st3.0', '_st4.0']
        """
        return ["_st{}.0".format(i) for i in num_list]

    def __call__(self, src_dir: str = None) -> List[str]:
        file_list = self._file_picker(src_dir)
        if self._key_wards_list is not None:
            file_list = self.pick_by_include_str(file_list, self._key_wards_list)
        file_list = self.pick_by_include_str(file_list, self._hole_wards_list)
        file_list = self.pick_by_include_str(file_list, self._step_wards_list)

        return file_list


class GetFilesParentNameList(PathAnalyzer):
    """
    ファイルのパスを与え、そのパイルパスの存在を確認する。
    ファイルが存在する場合は、親のディレクトリのディレクトリ名をリストで返す。
    リストのインデックスの0番側が直上の親ディレクトリ名
    """

    def __init__(self):
        self._path_tools = PathTools()

    def __call__(self, input_file_path: str) -> List[str]:
        # 与えられたファイルパスのpathオブジェクト
        p_file = self._path_tools.file_path_obj(input_file_path)
        return [str(p.name) for p in p_file.parents]


class GetFileListBySuffix(PathAnalyzer):
    """
    __init__で指定した拡張子ファイルのファイル名をList[str]で返す
    解析対象のディレクトリは__call__時に文字列で指定

    file_suffix :  Union[str, List[str]]
        検索対象とするファイルの拡張子('.'付きの文字列)。複数の場合はリストに並べて渡す。
    recursive : bool
        再帰的に探索するかどうか
    """

    def __init__(self, file_suffix: Union[str, List[str]], recursive: bool = False):
        assert file_suffix, "Need to specify the file extension. It is currently empty."
        self._path_tools = PathTools()
        self._recursive = recursive

        self._file_suffix_list = []
        if type(file_suffix) is str:
            self._file_suffix_list.append(file_suffix)
        if type(file_suffix) is list:
            self._file_suffix_list = file_suffix

    def __call__(self, src_dir: str = None) -> List[str]:
        """
        :param src_dir : 解析対象のディレクトリを指定
        :return : List[str]
        """

        file_names = []

        # ディレクトリの有無のチェックとPathオブジェクトの生成
        # src_dir=Noneの場合はカレントディレクトリのpathオブジェクトを返す
        if not src_dir:
            p_src_dir = Path.cwd()
        else:
            p_src_dir = self._path_tools.dir_path_obj(src_dir)

        for suffix in self._file_suffix_list:
            wild_card = "*" + suffix
            if self._recursive:  # 再帰的に探索する場合
                wild_card = "**/" + wild_card

            # glob() で拡張子指定して抽出
            p_file = [p for p in p_src_dir.glob(wild_card)]

            # ファイル名をソートしておく（不要かも？）
            p_file = self._path_tools.paths_sort(p_file)

            # 絶対バスの文字列を取り出す
            p_file_str = [str(p.resolve()) for p in p_file]

            # データの追加
            file_names.extend(p_file_str)

        return file_names


class OutputFilePathGenerator(PathAnalyzer):
    """
    ファイルのパスを与え、そのパイルパスの存在を確認する。
    ファイルが存在する場合は、そのパイルパスに対応する出力予定のファイルパス
    （元ファイルの拡張子を変更したもの）を指定された拡張子で生成し、その絶対パスを文字列で返す。
    出力先のディレクトリが存在しない場合はディレクトリの生成も行う。
    """

    def __init__(self, out_suffix: str, output_dir: str = None, add_name: str = None):
        self._path_tools = PathTools()
        self._output_suffix = out_suffix  # 生成する（出力）予定のファイルの拡張子
        self._add_name = add_name  # 入力したファイル名に文字列を追加したい場合に使用する
        self._output_dir = output_dir  # 出力先のディレクトリ（指定しない場合は入力ファイルのディレクトリと同じ場所）

    def __call__(self, input_file_path: str, output_dir=None, add_name: str = None):

        assert self._output_suffix, 'Provide suffix of output file!'

        if output_dir:
            out_dir = output_dir
        else:
            out_dir = self._output_dir

        # 与えられたファイルパスのpathオブジェクト
        p_file = self._path_tools.file_path_obj(input_file_path)

        # 出力先のディレクトリのpathオブジェクト（デフォルトは入力ファイルの親ディレクトリ）
        p_out_dir = p_file.parent

        # 出力先が指定されている場合はディレクトリを生成し、そこを出力先に指定
        if out_dir:
            p_out_dir = self._path_tools.dir_path_obj(out_dir)
            if not p_out_dir.exists():
                p_out_dir.mkdir()

        # 出力ファイル名を指定
        # 元ファイル名の拡張子をjoblibに変えたもの
        p_out_file = p_file.with_suffix(self._output_suffix)

        # 名前の追加
        add_name = (self._add_name or "") + (add_name or "")  # None の場合も文字列として扱いたい
        if add_name:
            p_out_file = p_out_file.with_name(p_out_file.stem + add_name + p_out_file.suffix)

        if p_file == p_out_file:  # ファイルパスが一致してしまう場合
            # 元のファイル名の先頭に'_'を追加した名前にする
            p_out_file = p_out_file.with_name('_' + p_out_file.name)

        # 出力の絶対ファイルパスをstrで取得
        # (そのファイルが存在しない状態でpathオブジェクトをresolve()しても絶対pathは取得できないので注意)
        p_out_file = p_out_dir.joinpath(p_out_file.name)

        return str(p_out_file.resolve())


class ZeroFillFileRename(GetFileListBySuffix):
    """
    __init__で指定した拡張子ファイルのファイル名を抽出し、ファイル名の数字の部分をゼロ埋めしてリネームする
    解析対象のディレクトリは__call__時に文字列で指定

    file_suffix : 検索対象とするファイルの拡張子('.'付きの文字列)。複数の場合はリストに並べて渡す
    split : ファイル名を分割する際のsplit文字
    zfill_dim : ゼロ埋めの桁数

    ファイル名を指定した文字で分割し、数値に置き換えることが可能な部分についてゼロ埋めしたファイル名に置き換える
    """

    def __init__(self, file_suffix: Union[str, List[str]], split: str = '_', zfill_dim: int = 6):
        self._path_tools = PathTools()
        super().__init__(file_suffix)
        self._zfill_dim = zfill_dim
        self._split = split

    def __call__(self, src_dir: str = None):
        """
        src_dir : 解析対象のディレクトリを指定
        """

        file_names = super().__call__(src_dir)

        p_file_names = [Path(f) for f in file_names]

        for p in p_file_names:
            file_path = p.resolve()
            new_name = self.zfill_name(file_path)
            print(new_name)
            p.rename(new_name)

    def zfill_name(self, file_path: str) -> str:
        """
        ファイルパスを与えると、ゼロ埋めしたファイル名を返す
        """
        p_file = self._path_tools.file_path_obj(file_path)
        stem = p_file.stem
        suffix = p_file.suffix
        split = stem.split(self._split)

        # 整数値に置き換えられる文字列の場合は一旦数値にしてから文字に戻す（ゼロ埋め文字をキャンセル）
        replaced = list(map(lambda x: str(int(x)) if x.isdecimal() else x, split))

        # 整数値に置き換えられる文字列の場合は、zfillでゼロ埋めした文字に変換する
        replaced = list(map(lambda x: x.zfill(self._zfill_dim) if x.isdecimal() else x, replaced))

        # ファイルのstem（ファイル名の拡張子を除外したもの
        r_stem = self._split.join(replaced)

        p_file = p_file.with_name(r_stem + suffix)

        return str(p_file.resolve())


class AddNumFileRename(GetFileListBySuffix):
    """
    __init__で指定した拡張子ファイルのファイル名を抽出し、ファイル名のaheadとbehindして指定した文字の間に
    ある文字を切りだし、切り出した文字が数値に変換できる場合は、数値に変換して、指定した値を加算した文字列に書き換える
    解析対象のディレクトリは__call__時に文字列で指定

    file_suffix : 検索対象とするファイルの拡張子('.'付きの文字列)。複数の場合はリストに並べて渡す
    add_value : 文字に加算する数値（int）
    ahead : 加算する文字を切り出すときの直前の文字列
    behaind : 加算する文字を切り出すときの直後の文字列
    zfill_dim : 加算した後にゼロ埋めする場合の桁数

    例えば、
    [
        'I:\\experimental_data\\999_programing_testdata\\01\\20201218_ac0_st1.0_zp-0.980.hdf5',
        ...
    ]

    を
    anf = AddNumFileRename(
        file_suffix=['.hdf5', '.h5'],
        add_value=10,
        ahead='_ac',
        behind='_st',
        zfill_dim=1,
    )
    anf(dir_path)
    と処理すると

     [
        'I:\experimental_data\999_programing_testdata\01\20201218_ac10_st1.0_zp-0.980.hdf5',
        ...
    ]

    """

    def __init__(
            self,
            file_suffix: Union[str, List[str]],
            add_value: int = 0,
            ahead: Optional[str] = None,
            behind: Optional[str] = None,
            zfill_dim: Optional[int] = None,
    ):
        super().__init__(file_suffix)
        self._add_value = add_value
        self._ahead = ahead
        self._behind = behind
        self._zfill_dim = zfill_dim

    def __call__(self, src_dir: str = None):
        """
        src_dir : 解析対象のディレクトリを指定
        """

        file_names = super().__call__(src_dir)

        p_file_names = [Path(f) for f in file_names]

        for p in p_file_names:
            file_path = p.resolve()
            new_name = self.add_num_filename(file_path)
            print(new_name)
            p.rename(new_name)

    @staticmethod
    def between_str_picker(txt: str, ahead: str = None, behind: str = None):
        """
        txtから、aheadとbehindして指定した文字の間にある文字を切りだして返す
        """
        ahead_idx = 0 if ahead is None else txt.find(ahead) + len(ahead)
        behind_idx = len(txt) if behind is None else txt.find(behind)

        #         target = target if (target:=txt[ahead_idx:behind_idx]) else None
        #         front = front if (front:=txt[:ahead_idx]) else None
        #         back = back if (back:=txt[behind_idx:]) else None

        target = txt[ahead_idx:behind_idx]
        front = txt[:ahead_idx]
        back = txt[behind_idx:]

        return target, front, back

    @staticmethod
    def is_num(s: str) -> bool:
        """
        負の値あるいは小数の値の文字列にはピリオド.やマイナス-が入っているため、isascii()以外のメソッドではFalseとなる。
        isascii()ではTrueとなるが、その他の記号やアルファベットが含まれていてもTrueとなるため、数値に変換できる文字列か
        どうかの判定には向かない。
        """
        try:
            float(s)
        except ValueError:
            return False
        else:
            return True

    def add_num_filename(self, file_name: str):
        p_file = Path(file_name)
        stem = p_file.stem
        suffix = p_file.suffix

        target, front, back = self.between_str_picker(stem, self._ahead, self._behind)

        # 整数値に置き換えられる文字列の場合は一旦数値にしてから文字に戻す（ゼロ埋め文字をキャンセル）
        #         target = str(int(target)+self._add_value) if target.isdecimal() else target
        target = str(int(target) + self._add_value) if self.is_num(target) else target

        if (self._zfill_dim is not None) and (self._zfill_dim > 0):
            # 整数値に置き換えられる文字列の場合は、zfillでゼロ埋めした文字に変換する
            #             target = target.zfill(self._zfill_dim) if target.isdecimal() else target
            target = target.zfill(self._zfill_dim) if self.is_num(target) else target

        # ファイルのstem（ファイル名の拡張子を除外したもの
        r_stem = front + target + back

        p_file = p_file.with_name(r_stem + suffix)

        return str(p_file.resolve())
