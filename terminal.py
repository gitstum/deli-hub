#--coding:utf-8--

import time
import random
import numpy as np
import pandas as pd
import inspect 

from tools import Tools
from features.indicators import *
from indicator import Indicator


class Terminal(Tools):

    # TODO: finalize this.

    name = 'Terminal'
    mapped_data = pd.Series()  
    class_data = pd.Series()
    # class_type_all = ['vector', 'condition', 'multiplier']

    terminal_pbs = dict(
        merge_pb=0.15, 
        smooth_pb=0.15,
        clear_pb=0.15,
        cut_pb=0.15,
        revalue_pb=0.15,
        jump_pb=0.15,

        remul_pb=0.15,
        reverse_pb=0.15,

        pop_pb=0.15,
        insert_pb=0.15,
        move_pb=0.15,

        window_pb=0.15,
        refeature_pb=0.15,
        addfeature_pb=0.15,
        popfeature_pb=0.15,
        )

    classifier_map = {

        # feature 单项切割
        'cut_number': dict(
            function=Tools.cut_number,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[],
            edge_mut_range={'start': -100.0, 'end': 100.0, 'sep': True, 'too_short': None, 'too_long': None},   # include both ends. 
            edge_mut_range_keep={'start': None, 'end': None, 'sep': None, 'too_short':True, 'too_long': True},
            map_type=['vector', 'multiplier', 'condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_num_range={'keep': True, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            multiplier_range={'keep': True, 'start': 2, 'end': 5, 'sep':0.2},
            ),
        'cut_rank': dict(
            function=Tools.cut_rank,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[],
            edge_mut_range={'start': 0.02, 'end': 0.98, 'sep': 0.02, 'too_short': None, 'too_long': None},   # include both ends.
            edge_mut_range_keep={'start': None, 'end': None, 'sep': None, 'too_short':True, 'too_long': True},
            map_type=['vector', 'multiplier', 'condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_num_range={'keep': True, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            multiplier_range={'keep': True, 'start': 2, 'end': 5, 'sep':0.2},
            ),
        'cut_sigma': dict(
            function=Tools.cut_sigma,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA, MovingSTD],
            edge_mut_range={'start': -2.0, 'end': 2.0, 'sep': 0.05, 'too_short': 0.11, 'too_long': 1.6},   # include both ends.
            edge_mut_range_keep={'start': None, 'end': None, 'sep': None, 'too_short':True, 'too_long': True},
            map_type=['vector', 'multiplier', 'condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_num_range={'keep': True, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            multiplier_range={'keep': True, 'start': 2, 'end': 5, 'sep':0.2},
            ),
        'cut_distance': dict(
            function=Tools.cut_distance,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[],
            edge_mut_range={'start': 0.02, 'end': 0.98, 'sep': 0.02, 'too_short': None, 'too_long': None},   # include both ends.
            edge_mut_range_keep={'start': None, 'end': None, 'sep': None, 'too_short':True, 'too_long': True},
            map_type=['vector', 'multiplier', 'condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_num_range={'keep': True, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            multiplier_range={'keep': True, 'start': 2, 'end': 5, 'sep':0.2},
            ),

        # feature 双项对比 (将同一个indicator计算出两个feature进行比较)
        'compare_distance': dict(
            function=Tools.compare_distance,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA, MovingSTD],
            edge_mut_range={'start': -100.0, 'end': 100.0, 'sep': True, 'too_short': None, 'too_long': None},   # include both ends.
            edge_mut_range_keep={'start': None, 'end': None, 'sep': None, 'too_short':True, 'too_long': True},
            map_type=['vector', 'multiplier', 'condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_num_range={'keep': True, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            multiplier_range={'keep': True, 'start': 2, 'end': 5, 'sep':0.2},
            ),
        'compare_sigma': dict(
            function=Tools.compare_sigma,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA, MovingSTD],
            edge_mut_range={'start': -2.0, 'end': 2.0, 'sep': 0.05, 'too_short': 0.11, 'too_long': 1.6},   # include both ends.
            edge_mut_range_keep={'start': None, 'end': None, 'sep': None, 'too_short':True, 'too_long': True},
            map_type=['vector', 'multiplier', 'condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            edge_num_range={'keep': True, 'start': 3, 'end': 6, 'sep': 1},  # include both ends.
            feature_window_ratio_range={'keep': True, 'start': 0.01, 'end': 0.1, 'sep': 0.001},  # include_both_ends.
            multiplier_range={'keep': True, 'start': 2, 'end': 5, 'sep':0.2},
            ),

        # feature 多项排列条件
        'perm_add': dict(
            function=Tools.perm_add,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_num_range={'keep': True, 'start': 2, 'end': 5, 'sep': 1},  # include both ends.
            ),
        'perm_sub': dict(
            function=Tools.perm_sub,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_num_range={'keep': True, 'start': 2, 'end': 5, 'sep': 1},  # include both ends.
            ),
        'perm_up': dict(
            function=Tools.perm_up,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_num_range={'keep': True, 'start': 2, 'end': 5, 'sep': 1},  # include both ends.
            ),
        'perm_down': dict(
            function=Tools.perm_down,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA, MovingSTD],
            map_type=['condition'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_num_range={'keep': True, 'start': 2, 'end': 5, 'sep': 1},  # include both ends.
            ),

        # feature 多项趋势
        'sig_trend_strict': dict(
            function=Tools.sig_trend_strict,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA],
            map_type=['vector'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_num_range={'keep': True, 'start': 2, 'end': 5, 'sep': 1},  # include both ends.
            ),
        'sig_trend_loose': dict(
            function=Tools.sig_trend_loose,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA],
            map_type=['vector'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_num_range={'keep': True, 'start': 2, 'end': 5, 'sep': 1},  # include both ends.
            ),
        'sig_trend_start_end': dict(
            function=Tools.sig_trend_start_end,
            weight=1.0,  # weight of chance to be chose.
            indicator_list=[MA, WEMA],
            map_type=['vector'],  # 结合indicator中的设定，随机选择（选择后不变异）。
            feature_num_range={'keep': True, 'start': 2, 'end': 5, 'sep': 1},  # include both ends.
            ),
    }

    classifier_group = {
        'cut': [Tools.cut_number, Tools.cut_rank, Tools.cut_sigma, Tools.cut_distance],
        'compare': [Tools.compare_distance, Tools.compare_sigma],
        'permutation': [Tools.perm_add, Tools.perm_sub, Tools.perm_up, Tools.perm_down],
        'trend': [Tools.sig_trend_strict, Tools.sig_trend_loose, Tools.sig_trend_start_end]
    }

    node_data = dict(
        terminal=True,
        map_type=None,
        class_func=None,
        class_func_group=None,

        class_args_edges=None,
        map_value_list=None,
        edge_mut_range=None,
        edge_mut_range_keep=None,
        edge_num_range=None,
        feature_window_ratio_range=None,
        class_kw=None,
        class_kw_range=None,
        class_kw_ins=None,
        multiplier_range=None, 
        multiplier_ref=None,
    
        value_reverse=False,
        feature_num_range=None,
        feature_num=None,
        class_args_features=None,
        class_args_features_ins=None
        )  # 初始的node_data

    def __init__(self, *, df_source, terminal_pbs=None, classifier_map=None, classifier_group=None, node_data=None):

        self.name = self.get_id('terminal')
        self.mapped_data = pd.Series()

        self.df_source = df_source

        if not terminal_pbs:
            self.terminal_pbs = Terminal.terminal_pbs.copy()
        else:
            self.terminal_pbs = terminal_pbs

        if not node_data:
            self.node_data = Terminal.node_data.copy()
        else:
            self.node_data = node_data

        if not classifier_map:
            self.classifier_map = Terminal.classifier_map
        else:
            self.classifier_map = classifier_map

        if not classifier_group:
            self.classifier_group = Terminal.classifier_group
        else:
            self.classifier_group = classifier_group

    def __get_classifier_detail(self, func):

        detail = {}
        for name, value in self.classifier_map.items():
            if value['function'] == func:
                detail = self.classifier_map[name].copy()
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
            mul_ref = 2  # 开局主观设定
            mul_ref_start = 1 / mul_ref
            for value in np.geomspace(mul_ref_start, mul_ref, num=value_num):
                map_value_list.append(self.fit_to_minimal(value, min_range=0.001))

            self.node_data['multiplier_ref'] = mul_ref

        map_value_list = list(map_value_list)

        return map_value_list

    def __generate_random_edges(self):

        start = self.node_data['edge_num_range']['start']
        sep = self.node_data['edge_num_range']['sep']
        end = self.node_data['edge_num_range']['end'] + sep
        edge_num = np.random.choice(list(range(start, end, sep)))

        start = self.node_data['edge_mut_range']['start']
        sep = self.node_data['edge_mut_range']['sep']
        end = self.node_data['edge_mut_range']['end']
        if sep == 0:
            return [start]  # sep==0: only value is start_value
        elif sep is True:
            sep = (end - start) / 100
            if isinstance(start, float):
                keep_float = len(str(start).split('.')[1])
                sep = self.shorten_float(sep, keep_float)
            else:
                sep = int(sep)
        else:
            end += sep
        edge_choice_box = np.arange(start, end, sep)

        class_args_edges = []
        for edge in list(np.random.choice(edge_choice_box, edge_num)):
            if isinstance(sep, float):
                edge = self.fit_to_minimal(edge, min_range=sep, modify=True)
            class_args_edges.append(edge)
        class_args_edges.sort()

        return class_args_edges

    # ------------------------------------------------------------------------------------------------------------------
    
    def mutate_mapping_list(self):

        mutation_tag = False
        map_type = self.node_data['map_type']

        if map_type == 'vector' or map_type == 'condition':
            mutation_tag = Tools.mutate_mapping_list(self.node_data, **self.terminal_pbs)

        elif map_type == 'multiplier':
            if 'remul_pb' in self.terminal_pbs and random.random() < self.terminal_pbs['remul_pb']:

                start = self.node_data['multiplier_range']['start']
                sep = self.node_data['multiplier_range']['sep']
                end = self.node_data['multiplier_range']['end'] + sep
                mul_ref = self.mutate_value(self.node_data['multiplier_ref'], start_value=start, end_value=end, sep=sep)
                self.node_data['multiplier_ref'] = mul_ref

                value_num = len(self.node_data['class_args_edges']) + 1
                map_value_list = []
                for value in np.geomspace(1 / mul_ref, mul_ref, num=value_num):
                    map_value_list.append(self.fit_to_minimal(value, min_range=0.001))

                print('lv.4 mutation: multiplier_zoom reset.')
                mutation_tag = True
    
        return mutation_tag

    def mutate_edge(self):

        # TODO: test

        mutation_tag = False
        map_type = self.node_data['map_type']

        if map_type == 'vector' or map_type == 'condition':
            mutation_tag = Tools.mutate_edge(self.node_data, **self.terminal_pbs)

        elif map_type == 'multiplier':

            edge_num = len(self.node_data['class_args_edges'])
            if 'pop_pb' in self.terminal_pbs and random.random() < self.terminal_pbs['pop_pb']:
                edge_num -= 1

            if 'insert_pb' in self.terminal_pbs and random.random() < self.terminal_pbs['insert_pb']:
                edge_num += 1

            if 'move_pb' in self.terminal_pbs:
                moved = Tools.mutate_edge_move(self.node_data, move_pb=self.terminal_pbs['move_pb'])
                if moved:
                    print('lv.5 mutation: edge moved.')
                    mutation_tag = True

            change_edge_num = False  # 是否需要改变edge数量
            if edge_num != len(self.node_data['class_args_edges']):

                if self.node_data['edge_num_range']['keep']:  # 目前这个keep只对multiplier有效
                    min_edge_num = self.node_data['edge_num_range']['start']
                    max_edge_num = self.node_data['edge_num_range']['end']
                    if min_edge_num <= edge_num <= max_edge_num:
                        change_edge_num = True
                else:
                    change_edge_num = True

            # 处理pop/insert  --最简单的做法：edge中间随机加一刀/减一刀
            if change_edge_num:

                if edge_num < len(self.node_data['class_args_edges']):

                    remove_edge = random.choice(self.node_data['class_args_edges'])
                    self.node_data['class_args_edges'].remove(remove_edge)
                    print('lv.5 mutation: edge poped.')


                elif edge_num > len(self.node_data['class_args_edges']):

                    edge_sep = self.node_data['edge_mut_range']['sep']
                    edge_min = min(self.node_data['class_args_edges'])
                    edge_max = max(self.node_data['class_args_edges'])
                    if edge_sep is True:
                        edge_sep = (edge_max - edge_min) / 50
                        if isinstance(edge_min, float):
                            keep_float = len(str(edge_min).split('.')[1])
                            edge_sep = self.shorten_float(edge_sep, keep_float)
                        else:
                            edge_sep = int(edge_sep)

                    new_edge = np.random.choice(np.arange(edge_min + edge_sep, edge_max, edge_sep))
                    while new_edge in self.node_data['class_args_edges']:
                        new_edge += edge_sep
                    self.node_data['class_args_edges'].append(new_edge)
                    self.node_data['class_args_edges'].sort()
                    print('lv.5 mutation: edge inserted.')

                mutation_tag = True

        return mutation_tag

    def mutate_feature_window(self):
        """LV.6 特征参数进化"""

        mutation_tag = False

        if 'window' in self.node_data['class_kw']:
            if 'window_pb' in self.terminal_pbs and random.random() < self.terminal_pbs['window_pb']:
                
                window = self.node_data['class_kw']['window']
                sep = self.node_data['class_kw_range']['window']['sep']
                if sep is True:
                    sep = None
                if self.node_data['class_kw_range']['window']['keep']:
                    start = self.node_data['class_kw_range']['window']['start']
                    end = self.node_data['class_kw_range']['window']['end']
                else:
                    start, end = None, None

                new_window = self.mutate_value(window, start_value=start, end_value=end, sep=sep)
                self.node_data['class_kw']['window'] = new_window

                print('lv.6 mutation: feature window changed.')
                mutation_tag = True

        return mutation_tag

    # ------------------------------------------------------------------------------------------------------------------
    def get_args(self, strip=True):

        node_data = self.node_data.copy()

        if strip:
            for key, value in node_data.items():
                if isinstance(value, pd.Series):
                    node_data[key] = self.node_data[key].copy()  # value包含默认浅拷贝的，要深拷贝，才不影响源数据。下同。
                    node_data[key] = pd.Series()  # value直接是sereis的，其实不用。保险起见。
                elif isinstance(value, list):
                    node_data[key] = self.node_data[key].copy()
                    n = 0
                    while n < len(value):
                        if isinstance(value[n], pd.Series):
                            node_data[key][n] = pd.Series()  # 清空。同上下
                        n += 1
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        if isinstance(value2, pd.Series):
                            node_data[key][key2] = self.node_data[key][key2].copy()
                            node_data[key][key2] = pd.Series()

        return node_data

    def copy(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    def create_terminal(self):
        """
        general:

            map_type
            # class_data
            class_func
            class_func_group
            # class_args: cut_edges / features
            
        cut or compare:
            class_args_edges
            map_value_list
            edge_mut_range
            edge_mut_range_keep
            edge_num_range
            feature_window_ratio_range
            # class_args_mutable
            # class_edge_sep
            # class_zoom_long
            # class_zom_short
            class_kw
            # class_kw_sep
            class_kw_range
            class_kw_ins
            multiplier_range
            multiplier_ref

        perm or trend:
            value_reverse
            feature_num_range
            feature_num
            class_args_features
            class_args_features_ins
        """
        
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
            self.node_data['edge_num_range'] = classifier_detail['edge_num_range']
            self.node_data['feature_window_ratio_range'] = classifier_detail['feature_window_ratio_range']
            self.node_data['multiplier_range'] = classifier_detail['multiplier_range']

            self.node_data['class_args_edges'] = self.__generate_random_edges()
            self.node_data['map_value_list'] = self.__generate_random_map_value()

            self.node_data['class_kw'] = {}
            self.node_data['class_kw_ins'] = {}
            self.node_data['class_kw_range'] = {}

            kwarg_list = inspect.getfullargspec(class_func)[4]
            feature_num = 0
            for kw in kwarg_list:
                if kw[:7] == 'feature':
                    feature_num += 1
            refeature_pb_each = self.probability_each(object_num=feature_num, pb_for_all=self.terminal_pbs['refeature_pb'])

            for kw in kwarg_list:

                if kw == 'window':
                    window_min = int(self.df_source.shape[0] * self.node_data['feature_window_ratio_range']['start'])
                    window_sep = self.node_data['feature_window_ratio_range']['sep']  # for sep == True
                    if window_sep and window_sep is not True:
                        window_sep = int(self.df_source.shape[0] * window_sep)
                    window_max = int(self.df_source.shape[0] * self.node_data['feature_window_ratio_range']['end'])
                    self.node_data['class_kw'][kw] = random.choice(list(range(window_min, window_max)))
                    self.node_data['class_kw_ins'][kw] = None
                    self.node_data['class_kw_range'][kw] = self.node_data['feature_window_ratio_range'].copy()
                    self.node_data['class_kw_range'][kw]['start'] = window_min
                    self.node_data['class_kw_range'][kw]['sep'] = window_sep
                    self.node_data['class_kw_range'][kw]['end'] = window_max
                
                elif kw[:7] == 'feature':
                    instance = indicator(df_source=self.df_source, refeature_pb=refeature_pb_each)  # 创建实例
                    self.node_data['class_kw'][kw] = instance.cal()  # 计算，获得feature
                    self.node_data['class_kw_ins'][kw] = instance
                    self.node_data['class_kw_range'][kw] = None
                
                else:
                    raise KeyError('Unknown keyword in class_function: %s. 9496' % class_func)

        elif func_group == 'permutation' or func_group == 'trend':

            self.node_data['feature_num_range'] = classifier_detail['feature_num_range']
            self.node_data['class_args_features'] = []
            self.node_data['class_args_features_ins'] = []

            start = classifier_detail['feature_num_range']['start']
            sep = classifier_detail['feature_num_range']['sep']
            end = classifier_detail['feature_num_range']['end'] + sep
            feature_num = int(np.random.choice(np.arange(start, end, sep)))
            self.node_data['feature_num'] = feature_num
            
            # refeature_pb_each = self.probability_each(object_num=feature_num,
            #                                           pb_for_all=self.terminal_pbs['refeature_pb'])
            refeature_pb_each = self.terminal_pbs['refeature_pb']  # 特殊556

            for num in range(feature_num):
                instance = indicator(df_source=self.df_source, refeature_pb=refeature_pb_each)  # 创建实例
                self.node_data['class_args_features_ins'].append(instance)
                self.node_data['class_args_features'].append(instance.cal()) 
                del instance

        else:
            raise ValueError('Uncategorized class_function: %s. 9484' % class_func)

        self.mapped_data = self.cal()

        return self.mapped_data


    def add_score(self, score):
        pass

    def update_score(self, sortino_score):
        pass

    def mutate_args(self):

        # TODO: test.

        mutation_tag = False  # TODO: implement this

        map_type = self.node_data['map_type']
        func_group = self.node_data['class_func_group']

        if func_group == 'cut' or func_group == 'compare':

            lv4_mut_map_value = self.mutate_mapping_list()  # LV.4 特征分类赋值进化
            lv5_mut_edge = self.mutate_edge()  # LV.5 特征分类截取进化
            lv6_mut_window = self.mutate_feature_window()  # LV.6 特征进化-window

            lv6_mut_feature = False
            for name, instance in self.node_data['class_kw_ins'].items():
                if isinstance(instance, Indicator):
                    mut_flag = instance.mutate_args()  # LV.6 特征进化
                    if mut_flag:
                        self.node_data['class_kw'][name] = instance.cal()  # get new feature data
                        print('lv.7 mutation: class_kw instance changed.')
                        lv6_mut_feature = True

            if lv4_mut_map_value | lv5_mut_edge | lv6_mut_window | lv6_mut_feature:
                mutation_tag = True

        elif func_group == 'permutation' or func_group == 'trend':

            # lv.4
            if random.random() < self.terminal_pbs['reverse_pb']:
                if self.node_data['value_reverse'] is True:
                    self.node_data['value_reverse'] = False
                else:
                    self.node_data['value_reverse'] = True
                mutation_tag = True
                print('lv.4 mutation: map reversed')

            keep_range = self.node_data['feature_num_range']['keep']
            feature_num_max = self.node_data['feature_num_range']['end']
            feature_num_min = self.node_data['feature_num_range']['start']
            sep = self.node_data['feature_num_range']['sep']  # 就是1
            feature_num = self.node_data['feature_num']

            # 1. lv.6 减少feature数量
            if not keep_range or self.if_in_range(feature_num - sep, feature_num_min, feature_num_max):
                if random.random() < self.terminal_pbs['popfeature_pb']:

                    pop_num = random.choice(list(range(feature_num)))
                    self.node_data['class_args_features_ins'].pop(pop_num)
                    self.node_data['class_args_features'].pop(pop_num)
                    # instance = self.node_data['class_args_features_ins'][pop_num]  # 巨坑提示：要从内存清除列表容器中的实例，务必走这几步
                    # data = self.node_data['class_args_features'][pop_num]
                    # self.node_data['class_args_features_ins'].remove(instance)
                    # self.node_data['class_args_features'].remove(data)
                    # del instance
                    # del data
                    self.node_data['feature_num'] -= sep
                    print('lv.6 mutation: feature_num decreased.')

            # 2. lv.6 增加feature数量
            if not keep_range or self.if_in_range(feature_num + sep, feature_num_min, feature_num_max):
                if random.random() < self.terminal_pbs['addfeature_pb']:
                    
                    refeature_pb_each = self.terminal_pbs['refeature_pb']  # 特殊556
                    indicator = self.node_data['class_args_features_ins'][0].__class__
                    instance = indicator(df_source=self.df_source, refeature_pb=refeature_pb_each)
                    add_num = random.choice(list(range(feature_num)))

                    self.node_data['class_args_features_ins'].insert(add_num, instance)
                    self.node_data['class_args_features'].insert(add_num, instance.cal())
                    del instance

                    self.node_data['feature_num'] += sep
                    print('lv.6 mutation: feature_num increased.')

            # 3. lv.7 改变feature计算参数
            for num in range(len(self.node_data['class_args_features_ins'])):
                instance = self.node_data['class_args_features_ins'][num]
                mut_flag = instance.mutate_args()
                if mut_flag:
                    old_data = self.node_data['class_args_features'][num]
                    self.node_data['class_args_features'][num] = instance.cal()
                    del old_data
                    print('lv.7 mutation: class_args feature changed.')
                    mutation_tag = True

        return mutation_tag

    def cal(self):

        func_group = self.node_data['class_func_group']

        if func_group == 'cut' or func_group == 'compare':

            func = self.node_data['class_func']
            args = self.node_data['class_args_edges']
            kwargs = self.node_data['class_kw']

            self.class_data = func(*args, **kwargs)
            self.mapped_data = self.get_mapped_data(self.class_data,
                                                    map_value_list=self.node_data['map_value_list'])

        elif func_group == 'permutation' or func_group == 'trend':

            func = self.node_data['class_func']
            args = self.node_data['class_args_features']

            self.class_data = func(*args)
            self.mapped_data = self.get_mapped_data(self.class_data,
                                                    reverse=self.node_data['value_reverse'],
                                                    reverse_type=self.node_data['map_type'])
            self.mapped_data = self.class_data

        return self.mapped_data


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