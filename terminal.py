#--coding:utf-8--

import time
import random
import numpy as np
import pandas as pd
import inspect 

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

        trend_pb=0.1,

        window_pb=0.1,
        refeature_pb=0.1
        )

    classifier_map = {

        # feature 单项切割
        'cut_number': dict(
            function=Tools.cut_number,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[],
            edge_mut_range={'start': -100.0, 'end': 100.0, 'sep': True, 'too_short': None, 'too_long': None},   # include both ends. 
            edge_mut_range_keep={'start': None, 'end': None, 'sep': None, 'too_short':True, 'too_long': True},
            map_type=['vector', 'multiplier'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_start_num_range={'keep': False, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            ),
        'cut_rank': dict(
            function=Tools.cut_rank,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[],
            edge_mut_range={'start': 0.02, 'end': 0.98, 'sep': 0.02, 'too_short': None, 'too_long': None},   # include both ends.
            edge_mut_range_keep={'start': None, 'end': None, 'sep': None, 'too_short':True, 'too_long': True},
            map_type=['vector', 'multiplier'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_start_num_range={'keep': False, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            ),
        'cut_sigma': dict(
            function=Tools.cut_sigma,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA, MovingSTD],
            edge_mut_range={'start': -2.0, 'end': 2.0, 'sep': 0.05, 'too_short': 0.11, 'too_long': 1.6},   # include both ends.
            edge_mut_range_keep={'start': None, 'end': None, 'sep': None, 'too_short':True, 'too_long': True},
            map_type=['vector', 'multiplier'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_start_num_range={'keep': False, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            ),
        'cut_distance': dict(
            function=Tools.cut_distance,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[],
            edge_mut_range={'start': 0.02, 'end': 0.98, 'sep': 0.02, 'too_short': None, 'too_long': None},   # include both ends.
            edge_mut_range_keep={'start': None, 'end': None, 'sep': None, 'too_short':True, 'too_long': True},
            map_type=['vector', 'multiplier'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_start_num_range={'keep': False, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            ),

        # feature 双项对比 (将同一个indicator计算出两个feature进行比较)
        'compare_distance': dict(
            function=Tools.compare_distance,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA, MovingSTD],
            edge_mut_range={'start': -100.0, 'end': 100.0, 'sep': True, 'too_short': None, 'too_long': None},   # include both ends.
            edge_mut_range_keep={'start': None, 'end': None, 'sep': None, 'too_short':True, 'too_long': True},
            map_type=['vector', 'multiplier'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_start_num_range={'keep': False, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            ),
        'compare_sigma': dict(
            function=Tools.compare_sigma,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA, MovingSTD],
            edge_mut_range={'start': -2.0, 'end': 2.0, 'sep': 0.05, 'too_short': 0.11, 'too_long': 1.6},   # include both ends.
            edge_mut_range_keep={'start': None, 'end': None, 'sep': None, 'too_short':True, 'too_long': True},
            map_type=['vector', 'multiplier'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_start_num_range={'keep': False, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            ),

        # feature 多项排列条件
        'perm_add': dict(
            function=Tools.perm_add,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_num_range={'keep': True, 'start': 2, 'end': 5, 'sep': 1}  # include both ends.
            ),
        'perm_sub': dict(
            function=Tools.perm_sub,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_num_range={'keep': True, 'start': 2, 'end': 5, 'sep': 1}  # include both ends.
            ),
        'perm_up': dict(
            function=Tools.perm_up,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_num_range={'keep': True, 'start': 2, 'end': 5, 'sep': 1}  # include both ends.
            ),
        'perm_down': dict(
            function=Tools.perm_down,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_num_range={'keep': True, 'start': 2, 'end': 5, 'sep': 1}  # include both ends.
            ),

        # feature 多项趋势
        'sig_trend_strict': dict(
            function=Tools.sig_trend_strict,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA],
            map_type=['vector'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_num_range={'keep': True, 'start': 2, 'end': 5, 'sep': 1}  # include both ends.
            ),
        'sig_trend_loose': dict(
            function=Tools.sig_trend_loose,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA],
            map_type=['vector'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_num_range={'keep': True, 'start': 2, 'end': 5, 'sep': 1}  # include both ends.
            ),
        'sig_trend_start_end': dict(
            function=Tools.sig_trend_start_end,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA],
            map_type=['vector'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_num_range={'keep': True, 'start': 2, 'end': 5, 'sep': 1}  # include both ends.
            ),
    }

    classifier_group = {
        'cut': [Tools.cut_number, Tools.cut_rank, Tools.cut_sigma, Tools.cut_distance],
        'compare': [Tools.compare_distance, Tools.compare_sigma],
        'permutation': [Tools.perm_add, Tools.perm_sub, Tools.perm_up, Tools.perm_down],
        'trend': [Tools.sig_trend_strict, Tools.sig_trend_loose, Tools.sig_trend_start_end]

    }

    node_data = dict(terminal=True)  # 初始的node_data 就这么多内容

    def __init__(self, *, df_source, terminal_pbs=None, classifier_map=None, classifier_group=None, node_data=None):

        self.name = self.get_id('terminal')
        self.mapped_data = pd.Series()

        self.df_source = df_source

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
        weight_list = []
        for value in self.classifier_map.values():

            if indicator in value['indicator_list']:
                class_func_list.append(value['function'])
                weight_list.append(value['weight'])

        index = 0
        class_func_box = []
        while index < len(weight_list):
            mul = int(weight_list[index] * 10)  # weight适用范围：小数点后1位
            for i in range(mul):
                class_func_box.append(class_func_list[index])  # 权重越大，添加的次数越多
            index += 1

        class_func = random.choice(class_func_box)
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
        for func, weight in indicator_pb_dict.items():
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
        value_num = len(self.node_data['class_args_edges']) + 1
        map_value_list = []

        if map_type == 'vector':
            map_value_list = np.random.randint(-1, 2, size=value_num)

        elif map_type == ('condition' or 'cond'):
            map_value_list = np.random.randint(0, 2, size=value_num)

        elif map_type == ('multiplier' or 'mult'):
            map_value_list = np.geomspace(0.25, 4, num=value_num)

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
    def get_args(self, strip=True):

        node_data = self.node_data.copy()

        if strip:
            for key, value in node_data.items():
                if isinstance(value, pd.Series):
                    node_data[key] = self.node_data[key].copy()  # value包含默认浅拷贝的，要深拷贝，才不影响源数据。下同。
                    node_data[key] = 'pd.Series'  # value直接是sereis的，其实不用。保险起见。
                elif isinstance(value, list):
                    node_data[key] = self.node_data[key].copy()
                    n = 0
                    while n < len(value):
                        if isinstance(value[n], pd.Series):
                            node_data[key][n] = 'pd.Series'
                        n += 1
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        if isinstance(value2, pd.Series):
                            node_data[key][key2] = self.node_data[key][key2].copy()
                            node_data[key][key2] = 'pd.Series'

        return node_data

    def copy(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    def create_terminal(self):
        """
        general:
            mapped_data: through map_value_list / direct

            map_type
            class_func
            class_func_group
            class_data
            # class_args: cut_edges / features
            
        cut or compare:
            class_args_edges
            map_value_list
            edge_mut_range
            edge_mut_range_keep
            edge_start_num_range
            feature_window_ratio_range
            # class_args_mutable
            # class_edge_sep
            # class_zoom_long
            # class_zom_short
            class_kw
            class_kw_sep
            class_kw_ins

        perm or trend:
            feature_num_range
            feature_num
            class_args_features
            class_args_features_ins
        """

        self.node_data['class_data'] = pd.Series()
        
        indicator = self.__get_indicator()
        class_func = self.__get_classifier_function(indicator)
        classifier_detail = self.__get_classifier_detail(class_func)

        map_type_box = set(indicator.map_type) & set(classifier_detail['map_type'])  # 取交集
        if not map_type_box:
            print('IndexError: Cannot choose from an empty sequence!!!! %s, %s' % (indicator, classifier_detail))
            self.create_terminal()  # 如匹配错误，重新生成。
        self.node_data['map_type'] = random.choice(list(map_type_box))
        self.node_data['class_func'] = class_func

        func_group = None
        for name, group in self.classifier_group.items():
            if class_func in group:
                func_group = name
        self.node_data['class_func_group'] = func_group

        if func_group == 'cut' or func_group == 'compare':

            # require 'edge_mut_range' and 'zoom_distance' in classifier_detail
            self.node_data['edge_mut_range'] = classifier_detail['edge_mut_range']
            self.node_data['edge_mut_range_keep'] = classifier_detail['edge_mut_range_keep']
            self.node_data['edge_start_num_range'] = classifier_detail['edge_start_num_range']
            self.node_data['feature_window_ratio_range'] = classifier_detail['feature_window_ratio_range']

            self.node_data['class_args_edges'] = self.__generate_random_edges()
            self.node_data['map_value_list'] = self.__generate_random_map_value()

            self.node_data['class_kw'] = {}
            self.node_data['class_kw_ins'] = {}
            self.node_data['class_kw_sep'] = {}
            for kw in inspect.getfullargspec(class_func)[4]:

                if kw == 'window':
                    window_min = int(self.df_source.shape[0] * self.node_data['feature_window_ratio_range']['start'])
                    window_max = int(self.df_source.shape[0] * self.node_data['feature_window_ratio_range']['end'])
                    self.node_data['class_kw'][kw] = random.choice(list(range(window_min, window_max)))
                    self.node_data['class_kw_ins'][kw] = None
                    self.node_data['class_kw_sep'][kw] = self.node_data['feature_window_ratio_range']['sep']
                
                elif kw[:7] == 'feature':
                    instance = indicator(df_source=self.df_source)  # 创建实例
                    self.node_data['class_kw'][kw] = instance.cal()  # 计算，获得feature
                    self.node_data['class_kw_ins'][kw] = instance
                    self.node_data['class_kw_sep'][kw] = None
                
                else:
                    raise KeyError('Unknown keyword in class_function: %s. 9496' % class_func)

        elif func_group == 'permutation' or func_group == 'trend':

            self.node_data['feature_num_range'] = classifier_detail['feature_num_range']
            self.node_data['class_args_features'] = []
            self.node_data['class_args_features_ins'] = []

            start = classifier_detail['feature_num_range']['start']
            sep = classifier_detail['feature_num_range']['sep']
            end = classifier_detail['feature_num_range']['end'] + sep
            self.node_data['feature_num'] = np.random.choice(np.arange(start, end, sep))

            for num in range(self.node_data['feature_num']):
                instance = indicator(df_source=self.df_source)  # 创建实例
                self.node_data['class_args_features_ins'].append(instance)
                self.node_data['class_args_features'].append(instance.cal()) 

        else:
            raise ValueError('Uncategorized class_function: %s. 9484' % class_func)


    def add_score(self, score):
        pass

    def update_score(self, sortino_score):
        pass

    def mutate_args(self, pbs):
        pass

    def cal(self):
        pass



if __name__ == '__main__':
    pd.set_option('display.max_rows', 8)

    df = pd.read_csv('data/bitmex_price_1hour_2020q1.csv')

    test = Terminal(df_source=df)
    print(test)

    test.create_terminal()
    # print(test.get_args(strip=False))
    print(test.get_args())


    # print(MA.get_indicator_mutable_dimension_num(MA.default_range))
    # print(WEMA.get_indicator_mutable_dimension_num(WEMA.default_range))