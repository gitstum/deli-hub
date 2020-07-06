#--coding:utf-8--

import time
import random
import numpy as np
import pandas as pd

from tools import Tools
from features.indicators import *


class Terminal(Tools):

    # TODO: finalize this.

    name = 'Terminal'
    class_data = pd.Series()

    map_value_list = []
    map_value_type = None
    class_func = None
    class_args = []
    class_arg_mutable = []

    edge_start_num_range = {'start': 2, 'end': 5, 'sep': 1}
    feature_window_ratio_range = {'start': 0.01, 'end': 0.05, 'sep': 0.001}

    classifier_map = {

        # feature 单项切割
        'cut_number': dict(
            function=Tools.cut_number,
            indicator_list=[],
            edge_mut_range={'start': -100.0, 'end': 100.0, 'sep': True},
            zoom_distance={'too_short': None, 'too_long': None},
            map_value_type='vector',
            ),
        'cut_rank': dict(
            function=Tools.cut_rank,
            indicator_list=[],
            edge_mut_range={'start': 0.02, 'end': 0.98, 'sep': 0.02},
            zoom_distance={'too_short': None, 'too_long': None},
            map_value_type='vector',
            ),
        'cut_sigma': dict(
            function=Tools.cut_sigma,
            indicator_list=[MA, WEMA, MovingSTD],
            edge_mut_range={'start': -2.0, 'end': 2.0, 'sep': 0.05},
            zoom_distance={'too_short': 0.11, 'too_long': 1.6},
            map_value_type='vector',
            ),
        'cut_distance': dict(
            function=Tools.cut_distance,
            indicator_list=[],
            edge_mut_range={'start': 0.02, 'end': 0.98, 'sep': 0.02},
            zoom_distance={'too_short': None, 'too_long': None},
            map_value_type='vector',
            ),

        # feature 双项对比 (将同一个indicator计算出两个feature进行比较)
        'compare_distance': dict(
            function=Tools.compare_distance,
            indicator_list=[],
            edge_mut_range={'start': -100.0, 'end': 100.0, 'sep': True},
            zoom_distance={'too_short': None, 'too_long': None},
            map_value_type='vector',
            ),
        'compare_sigma': dict(
            function=Tools.compare_sigma,
            indicator_list=[MA, WEMA, MovingSTD],
            edge_mut_range={'start': -2.0, 'end': 2.0, 'sep': 0.05},
            zoom_distance={'too_short': 0.11, 'too_long': 1.6},
            map_value_type='vector',
            ),

        # feature 多项排列条件
        'perm_add': dict(
            function=Tools.perm_add,
            indicator_list=[MA, WEMA, MovingSTD],
            map_value_type='condition',
            ),
        'perm_sub': dict(
            function=Tools.perm_sub,
            indicator_list=[MA, WEMA, MovingSTD],
            map_value_type='condition',
            ),
        'perm_up': dict(
            function=Tools.perm_up,
            indicator_list=[MA, WEMA, MovingSTD],
            map_value_type='condition',
            ),
        'perm_down': dict(
            function=Tools.perm_down,
            indicator_list=[MA, WEMA, MovingSTD],
            map_value_type='condition',
            ),

        # feature 多项趋势
        'sig_trend_strict': dict(
            function=Tools.sig_trend_strict,
            indicator_list=[MA, WEMA, MovingSTD],
            map_value_type='vector',
            ),
        'sig_trend_loose': dict(
            function=Tools.sig_trend_loose,
            indicator_list=[MA, WEMA, MovingSTD],
            map_value_type='vector',
            ),
        'sig_trend_start_end': dict(
            function=Tools.sig_trend_start_end,
            indicator_list=[MA, WEMA, MovingSTD],
            map_value_type='vector',
            ),
    }

    def __init__(self, *, name=None, edge_num=None, pbs):

        self.name = self.get_id('terminal')

    def __generate_random_edges(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    def get_args(self):

        args_dict = dict(
            name=self.name,
            )
        return args_dict

    # ------------------------------------------------------------------------------------------------------------------
    def create_terminal(self):
        pass

    def add_score(self, score):
        pass

    def update_score(self, sortino_score):
        pass

    def mutate_args(self, pbs):
        pass

    def cal(self):
        pass

