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
    mapped_data = pd.Series()  
    # class_type_all = ['vector', 'condition', 'multiplier']

    terminal_pbs = dict(
        merge_pb=0.1, 
        smooth_pb=0.1,
        clear_pb=0.1,
        cut_pb=0.1,
        revalue_pb=0.1,
        jump_pb=0.1,

        pop_pb=0.1,
        insert_pb=0.1,
        move_pb=0.1,

        window_pb=0.1,
        refeature_pb=0.1
        )

    node_data = dict(
        edge_start_num_range = {'start': 3, 'end': 6, 'sep': 1},  # include both ends.
        feature_window_ratio_range = {'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
        )  # 初始的node_data 就这么多内容

    classifier_map = {

        # feature 单项切割
        'cut_number': dict(
            function=Tools.cut_number,
            indicator_list=[],
            edge_mut_range={'keep': False, 'start': -100.0, 'end': 100.0, 'sep': True},   # include both ends. 
            zoom_distance={'too_short': None, 'too_long': None},   # exclude both ends.
            map_type=['vector', 'multiplier'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_start_num_range={'keep': False, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            ),
        'cut_rank': dict(
            function=Tools.cut_rank,
            indicator_list=[],
            edge_mut_range={'keep': True, 'start': 0.02, 'end': 0.98, 'sep': 0.02},   # include both ends.
            zoom_distance={'too_short': None, 'too_long': None},   # exclude both ends.
            map_type=['vector', 'multiplier'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_start_num_range={'keep': False, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            ),
        'cut_sigma': dict(
            function=Tools.cut_sigma,
            indicator_list=[MA, WEMA, MovingSTD],
            edge_mut_range={'keep': False, 'start': -2.0, 'end': 2.0, 'sep': 0.05},   # include both ends.
            zoom_distance={'too_short': 0.11, 'too_long': 1.6},   # exclude both ends.
            map_type=['vector', 'multiplier'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_start_num_range={'keep': False, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            ),
        'cut_distance': dict(
            function=Tools.cut_distance,
            indicator_list=[],
            edge_mut_range={'keep': True, 'start': 0.02, 'end': 0.98, 'sep': 0.02},   # include both ends.
            zoom_distance={'too_short': None, 'too_long': None},   # exclude both ends.
            map_type=['vector', 'multiplier'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_start_num_range={'keep': False, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            ),

        # feature 双项对比 (将同一个indicator计算出两个feature进行比较)
        'compare_distance': dict(
            function=Tools.compare_distance,
            indicator_list=[MA, WEMA, MovingSTD],
            edge_mut_range={'keep': False, 'start': -100.0, 'end': 100.0, 'sep': True},   # include both ends.
            zoom_distance={'too_short': None, 'too_long': None},   # exclude both ends.
            map_type=['vector', 'multiplier'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_start_num_range={'keep': False, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            ),
        'compare_sigma': dict(
            function=Tools.compare_sigma,
            indicator_list=[MA, WEMA, MovingSTD],
            edge_mut_range={'keep': False, 'start': -2.0, 'end': 2.0, 'sep': 0.05},   # include both ends.
            zoom_distance={'too_short': 0.11, 'too_long': 1.6},   # exclude both ends.
            map_type=['vector', 'multiplier'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_start_num_range={'keep': False, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            ),

        # feature 多项排列条件
        'perm_add': dict(
            function=Tools.perm_add,
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_start_num_range={'keep': False, 'start': 2, 'end': 5, 'sep': 1}  # include both ends.
            ),
        'perm_sub': dict(
            function=Tools.perm_sub,
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_start_num_range={'keep': False, 'start': 2, 'end': 5, 'sep': 1}  # include both ends.
            ),
        'perm_up': dict(
            function=Tools.perm_up,
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_start_num_range={'keep': False, 'start': 2, 'end': 5, 'sep': 1}  # include both ends.
            ),
        'perm_down': dict(
            function=Tools.perm_down,
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_start_num_range={'keep': False, 'start': 2, 'end': 5, 'sep': 1}  # include both ends.
            ),

        # feature 多项趋势
        'sig_trend_strict': dict(
            function=Tools.sig_trend_strict,
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['vector'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_start_num_range={'keep': False, 'start': 2, 'end': 5, 'sep': 1}  # include both ends.
            ),
        'sig_trend_loose': dict(
            function=Tools.sig_trend_loose,
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['vector'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_start_num_range={'keep': False, 'start': 2, 'end': 5, 'sep': 1}  # include both ends.
            ),
        'sig_trend_start_end': dict(
            function=Tools.sig_trend_start_end,
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['vector'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_start_num_range={'keep': False, 'start': 2, 'end': 5, 'sep': 1}  # include both ends.
            ),

    }

    classifier_group = {
        'cut': [Tools.cut_number, Tools.cut_rank, Tools.cut_sigma, Tools.cut_distance],
        'compare': [Tools.compare_distance, Tools.compare_sigma],
        'permutation': [Tools.perm_add, Tools.perm_sub, Tools.perm_up, Tools.perm_down],
        'trend': [Tools.sig_trend_strict, Tools.sig_trend_loose, Tools.sig_trend_start_end]

    }

    def __init__(self, *, df_source, terminal_pbs=None, node_data=None, classifier_map=None):

        self.name = self.get_id('terminal')
        self.mapped_data = pd.Series()

        if not terminal_pbs:
            self.terminal_pbs = Terminal.terminal_pbs
        else:
            self.terminal_pbs = terminal_pbs

        if not node_data:
            self.node_data = Terminal.node_data
        else:
            self.node_data = node_data

        if not classifier_map:
            self.classifier_map = Terminal.classifier_map
        else:
            self.classifier_map = classifier_map

    def __get_classifier_detail(self, func):

        detail = {}
        for name, value in self.classifier_map.items():
            if value['function'] == func:
                detail = self.classifier_map[name]
                break

        return detail

    def __get_classifier_function(self, indicator):

        class_func_list = []
        for value in self.classifier_map.values():

            if indicator in value['indicator_list']:
                class_func_list.append(value['function'])

        class_func = random.choice(class_func_list)
        return class_func

    def __get_indicator(self):

        choice_box = []
        while len(choice_box) < 1:
            choice_box = self.__choose_indicator_by_weight()

        func = random.choice(choice_box)
        return func

    def __choose_indicator_by_weight(self):

        indicator_pb_dict = self.__get_indicator_pb_dict()  # 避免重复计算，放在这里

        func_list = []
        for func, weight in indicator_pb_dict:
            if random.random() < weight:
                func_list.append(func)

        return func_list

    def __get_indicator_pb_dict(self):
        """根据各indicator中可变异的参数数目，来决定初始被抽取的概率"""

        indicator_pb_dict = {}

        indicator_list = []
        for value in self.classifier_map.values():
            indicator_list += value['indicator_list']

        indicator_set = set(indicator_list)
        weight_all = 0
        for indicator in indicator_set:
            weight = indicator.get_indicator_mutable_dimension_num(indicator.default_range) + 1  # +1: 照顾0变异的情况
            indicator_pb_dict[indicator] = weight
            weight_all += weight

        for key, value in indicator_pb_dict.items():
            new_value = value / weight_all
            indicator_pb_dict[key] = new_value

        return indicator_pb_dict

    def __generate_random_map_value(self):

        map_type = self.node_data['map_type']
        value_num = len(self.node_data['class_args']) + 1
        map_value_list = []

        if map_type == 'vector':
            map_value_list = np.random.randint(-1, 2, size=value_num)

        elif map_type == ('condition' or 'cond'):
            map_value_list = np.random.randint(0, 2, size=value_num)

        elif map_type == ('multiplier' or 'mult'):
            map_value_list = np.geomspace(0.25, 2.5, num=value_num)

        map_value_list = list(map_value_list)

        return map_value_list


    def __generate_random_edges(self):

        start = self.node_data['edge_start_num_range']['start']
        sep = self.node_data['edge_start_num_range']['sep']
        end = self.node_data['edge_start_num_range']['end'] + sep
        edge_num = np.random.choice(list(range(start, end, sep)))

        start = self.node_data['edge_mut_range']['start']
        sep = self.node_data['edge_mut_range']['sep']
        end = self.node_data['edge_mut_range']['end']
        if sep == 0:
            return [start]  # sep==0: only value is start_value
        elif sep is True:
            sep = (end - start) / 20
            if isinstance(start, float):
                keep_float = len(str(start).split('.')[1])
                sep = self.shorten_float(sep, keep_float)
            else:
                sep = int(sep)
        else:
            end += sep
        edge_choice_box = np.arange(start, end, sep)

        class_args = list(np.random.choice(edge_choice_box, edge_num))
        return class_args

    # ------------------------------------------------------------------------------------------------------------------
    def get_args(self):
        return self.node_data

    # ------------------------------------------------------------------------------------------------------------------
    def create_terminal(self):
        
        indicator = self.__get_indicator()
        class_func = self.__get_classifier_function(indicator)
        classifier_detail = self.__get_classifier_detail(class_func)

        map_type_box = set(indicator.map_type) & set(classifier_detail['map_type'])  # 取交集
        self.node_data['map_type'] = random.choice(map_type_box)
        self.node_data['class_func'] = class_func

        func_group = None
        for name, group in self.classifier_group:
            if class_func in group:
                func_group = name

        # TODO: finish.

        if func_group == ('cut' or 'compare'):

            if 'edge_mut_range' in classifier_detail:
                self.node_data['edge_mut_range'] = classifier_detail['edge_mut_range']
                self.node_data['class_edge_sep'] = classifier_detail['edge_mut_range']['sep']
                self.node_data['class_args'] = self.__generate_random_edges()
                self.node_data['class_arg_mutable'] = [True] * len(self.node_data['class_args'])
            if 'zoom_distance' in classifier_detail:
                self.node_data['class_zoom_long'] = classifier_detail['zoom_distance']['too_long']  # 照顾已经开发好的代码。。
                self.node_data['class_zoom_short'] = classifier_detail['zoom_distance']['too_short']

            self.node_data['map_value_list'] = self.__generate_random_map_value()

            self.node_data['class_kw'] = "TODO"
            self.node_data['class_kw_sep'] = "TODO"
            self.node_data['class_kw_ins'] = "TODO"

        elif func_group == ('permutation' or 'trend'):
            pass
        else:

            pass

        self.node_data['class_data'] = "TODO"


    def add_score(self, score):
        pass

    def update_score(self, sortino_score):
        pass

    def mutate_args(self, pbs):
        pass

    def cal(self):
        pass



if __name__ == '__main__':


    test = Terminal(df_source=None)
    print(test)

    print(MA.get_indicator_mutable_dimension_num(MA.default_range))
    print(WEMA.get_indicator_mutable_dimension_num(WEMA.default_range))