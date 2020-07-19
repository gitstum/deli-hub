# --coding:utf-8--

import time
import random
import numpy as np
import pandas as pd
import inspect

from tools import Tools
from features.indicators import *
from indicator import Indicator
from private.parameters import Classifier


class Terminal(Tools):
    name = 'terminal'
    score_list = []
    score_num = 0
    avg_score = 0

    def __init__(self, *, df_source, terminal_pbs=None, classifier_map=None, classifier_group=None,
                 node_data=None, lv_mut_tag=None):

        # rank --------------------------------------------
        self.name = self.get_id('terminal')
        self.score_list = []
        self.score_num = 0
        self.avg_score = 0

        # get data ----------------------------------------
        self.node_result = pd.Series()  # weighted_data
        self.terminal_result = pd.Series()  # mapped_data
        self.class_data = pd.Series()

        self.depth = 0
        self.width = 0 
        self.population = 1

        # inputs ------------------------------------------
        self.df_source = df_source

        if not terminal_pbs:
            self.terminal_pbs = Classifier.terminal_pbs.copy()
        else:
            self.terminal_pbs = terminal_pbs.copy()

        if not classifier_map:
            self.classifier_map = Classifier.classifier_map
        else:
            self.classifier_map = classifier_map

        if not classifier_group:
            self.classifier_group = Classifier.classifier_group
        else:
            self.classifier_group = classifier_group

        # rebuild model -----------
        if not node_data:
            self.node_data = Classifier.node_data.copy()
        else:
            self.node_data = node_data.copy()

        if not lv_mut_tag:
            self.lv_mut_tag = Classifier.lv_mut_tag.copy()
        else:
            self.lv_mut_tag = lv_mut_tag.copy()

    def __get_classifier_detail(self, func):

        detail = {}
        for name, value in self.classifier_map.items():

            if value['function'] == func:
                detail = self.classifier_map[name].copy()

                for key, ref in self.classifier_map[name].items():
                    if isinstance(ref, dict) or isinstance(ref, list):
                        detail[key] = self.classifier_map[name][key].copy()  # 深拷贝所有内部可变容器，会重写

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
            weight = indicator.get_indicator_mutable_dimension_num() + 1  # +1: 照顾0变异的情况
            indicator_pb_dict[indicator] = weight
            weight_all += weight

        for key, value in indicator_pb_dict.items():
            new_value = value / weight_all
            indicator_pb_dict[key] = new_value

        return indicator_pb_dict

    def __get_edge_mut_range(self):
        """处理edge缺失的引用项"""

        edge_list = self.node_data['class_args_edges']
        edge_mut_range = self.node_data['edge_mut_range']

        start = edge_mut_range['start']
        sep = edge_mut_range['sep']
        end = edge_mut_range['end']
        too_short = edge_mut_range['too_short']
        too_long = edge_mut_range['too_long']

        # 1. sep
        if sep is True:  # 自动步长

            if not edge_list:  # 初始化过程中
                sep = (end - start) / 100

            else:
                sep = (max(edge_list) - min(edge_list)) / 20

            if isinstance(start, float):
                keep_float = len(str(start).split('.')[1]) + 1
                sep = self.shorten_float(sep, keep_float)
                if sep < 0.001:
                    sep = 0.001  # 防止 sep==0 的情况出现
            else:
                sep = int(sep)

        else:
            end += sep

        # 2. too_short
        if too_short is None:
            too_short = sep * 3

        # 3. too_long
        if too_long is None:
            too_long = (end - start) / 2

        edge_mut_range_new = {'start': start, 'end': end, 'sep': sep, 'too_short': too_short, 'too_long': too_long}
        return edge_mut_range_new

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

        edge_mut_range = self.__get_edge_mut_range()
        start = edge_mut_range['start']
        sep = edge_mut_range['sep']
        end = edge_mut_range['end']

        if sep == 0:
            return [start]  # sep==0: only value is start_value

        edge_choice_box = np.arange(start, end, sep)
        class_args_edges = []
        for edge in list(np.random.choice(edge_choice_box, edge_num)):

            if isinstance(sep, float):
                edge = self.fit_to_minimal(edge, min_range=sep, modify=True)
            class_args_edges.append(edge)

        class_args_edges.sort()

        return class_args_edges

    # LV.4-5 --------------------------------------------------------------------------
    def __update_node_terminal(self, **kwargs):
        """更新节点node数据"""

        updated = False

        for key, value in kwargs.items():

            if value != self.node_data[key]:
                if key == 'class_args_edges':
                    value.sort()  # 这个特殊，记得要排序一下！

                self.node_data[key] = value
                updated = True

        # 旧方法：
        # if map_value_list and map_value_list != self.node_data['map_value_list']:
        #     self.node_data['map_value_list'] = map_value_list.copy()
        #     updated = True

        # if class_args_edges and class_args_edges != self.node_data['class_args_edges']:
        #     class_args_edges.sort()  # 记得要排序一下！
        #     self.node_data['class_args_edges'] = class_args_edges.copy()
        #     updated = True

        return updated

    # LV.4 -----------------------------------------------------------------------------
    def __mutate_mapping_list(self):
        """LV.4 MUTATION: 特征分类赋值进化
        @return: True for any mutation happened for mapping_list(and classify_args)
        NOTE: inplace. all pb are independent. Error if node_data['class_args_edges'] contains other than edges.
        """

        mutation_tag = False
        edge_list = self.node_data['class_args_edges']
        map_type = self.node_data['map_type']
        func_group = self.node_data['class_func_group']
        edge_num_range = self.node_data['edge_num_range']

        if func_group == 'cut' or func_group == 'compare':

            keep = edge_num_range['keep']
            if keep:
                start = edge_num_range['start']
                end = edge_num_range['end']
            else:
                start = None
                end = None

            if map_type == 'multiplier':

                # 增强类型的数据，只进行整体赋值的更改
                if 'remul_pb' in self.terminal_pbs and self.terminal_pbs['remul_pb']:

                    remuled = self.__mutate_mapping_list_multiplier()
                    if remuled:
                        print('lv.4 mutation: multiplier_zoom reset.')
                        mutation_tag = True

            elif map_type == 'vector' or map_type == 'condition':

                # 注意，mutation的5项的顺序是有考虑的，且完成一个到下一个，不要随意改变先后次序或合并
                # 1. mapping_list 连续同值合并【优化】 --merge_pb
                if 'merge_pb' in self.terminal_pbs and self.terminal_pbs['merge_pb']:

                    target_edge_num = len(edge_list) - 1
                    if Tools.if_in_range(target_edge_num, start, end) and len(edge_list) > 1:

                        merged = self.__mutate_mapping_list_merge_same()
                        if merged:
                            mutation_tag = True
                            print('lv.4 mutation: mapping_list merged.')

                # 2. mapping_list 跳值平滑【优化】  --smooth_pb
                if 'smooth_pb' in self.terminal_pbs and self.terminal_pbs['smooth_pb']:

                    target_edge_num = len(edge_list) + 1
                    if Tools.if_in_range(target_edge_num, start, end) and map_type == 'vector':

                        smoothed = self.__mutate_mapping_list_jump_smooth()
                        if smoothed:
                            mutation_tag = True
                            print('lv.4 mutation: mapping_list smoothed.')

                # 3. mapping_list 同值间异类剔除【半优化】  --clear_pb
                if 'clear_pb' in self.terminal_pbs and self.terminal_pbs['clear_pb']:

                    target_edge_num = len(edge_list) - 1
                    if Tools.if_in_range(target_edge_num, start, end):

                        cleared = self.__mutate_mapping_list_clear_between()
                        if cleared:
                            mutation_tag = True
                            print('lv.4 mutation: mapping_list cleared.')

                # 4. mapping_list 数目增加 --cut_pb
                if 'cut_pb' in self.terminal_pbs and self.terminal_pbs['cut_pb']:

                    target_edge_num = target_edge_num = len(edge_list) + 1
                    if Tools.if_in_range(target_edge_num, start, end):

                        cut = self.__mutate_mapping_list_cut_within()
                        if cut:
                            mutation_tag = True
                            print('lv.4 mutation: mapping_list cut.')

                # 5. mapping_list 赋值变异 --revalue_pb  (这一步才是这个函数正儿八经最应该做的事情）
                if 'revalue_pb' in self.terminal_pbs and self.terminal_pbs['revalue_pb']:

                    changed = self.__mutate_mapping_list_change_value()
                    if changed:
                        mutation_tag = True
                        print('lv.4 mutation: mapping_list value_changed.')

        # elif func_group == 'permutation' or func_group == 'trend':  # 综合考虑，将reverse作为所有类别都可能发生的lv4.map_value变异
        if 'reverse_pb' in self.terminal_pbs and self.terminal_pbs['reverse_pb']:

            map_reversed = self.__mutate_mapping_list_reverse()
            if map_reversed:
                mutation_tag = True
                print('lv.4 mutation: map reversed.')

        return mutation_tag

    # LV.4 -----------------------------------------------------------------------------
    def __mutate_mapping_list_reverse(self):

        reverse_pb = self.terminal_pbs['reverse_pb']
        map_type = self.node_data['map_type']
        map_value_list = self.node_data['map_value_list'].copy()

        if random.random() < reverse_pb:

            arr = np.array(map_value_list)
            arr_ref = arr.copy()

            if map_type == 'vector':
                arr *= -1
                map_value_list = list(arr)

            elif map_type == 'condition':
                arr[arr_ref == 0] = 1
                arr[arr_ref != 0] = 0
                map_value_list = list(arr)

            elif map_type == 'multiplier':
                map_value_list.reverse()

        node_updated = self.__update_node_terminal(map_value_list=map_value_list)

        return node_updated

    # LV.4 -----------------------------------------------------------------------------
    def __get_mapped_data(self):
        """映射函数"""

        class_data = self.class_data.copy()
        mapped_data = class_data.copy()
        map_value_list = self.node_data['map_value_list']
        func_group = self.node_data['class_func_group']

        num = 0
        if func_group == 'trend':
            num = -1
        for value in map_value_list:
            mapped_data[class_data == num] = value
            num += 1

        return mapped_data

    # LV.4 -----------------------------------------------------------------------------
    def __mutate_one_value_in_map(self, value):

        map_type = self.node_data['map_type']
        jump_pb = self.terminal_pbs['jump_pb']
        new_value = value

        if map_type == 'vector':  # bracket = [-1, 0, 1]  # value的值域

            if value == 0:
                if random.random() < 0.5:
                    new_value = -1
                else:
                    new_value = 1

            else:
                if random.random() < jump_pb:  # jump_pb本身就是each的
                    if value < 0:
                        new_value = 1
                    else:
                        new_value = -1
                else:
                    new_value = 0

        elif map_type == ('cond' or 'condition'):  # bracket = [0, 1]

            if value != 0:
                new_value = 0
            else:
                new_value = 1

        elif map_type == ('multiplier' or 'mult'):
            new_value = value  # multiplier类别进不来这里。以防万一。

        return new_value

    # LV.4 -----------------------------------------------------------------------------
    def __mutate_mapping_list_change_value(self):
        """mapping_list 赋值变异 """

        revalue_pb = self.terminal_pbs['revalue_pb']
        mapping_list = self.node_data['map_value_list']
        mapping_list_new = self.node_data['map_value_list'].copy()

        revalue_pb_each = Tools.probability_each(object_num=len(mapping_list),
                                                 pb_for_all=revalue_pb)  # 各元素发生变异的独立概率
        num = 0
        for value in mapping_list:
            if random.random() < revalue_pb_each:
                new_value = self.__mutate_one_value_in_map(value)
                mapping_list_new[num] = new_value
            num += 1

        node_updated = self.__update_node_terminal(map_value_list=mapping_list_new)

        return node_updated

    # LV.4 -----------------------------------------------------------------------------
    def __mutate_mapping_list_cut_within(self):
        """mapping_list 数目增加：通过将某段赋值切开（但赋值仍相同）"""

        edge_list_new = self.node_data['class_args_edges'].copy()
        if not edge_list_new:
            return False  # 缺乏切割的参考edge值，会导致出错。

        cut_pb = self.terminal_pbs['cut_pb']
        mapping_list_new = self.node_data['map_value_list'].copy()
        edge_mut_range = self.__get_edge_mut_range()
        sep = edge_mut_range['sep']
        zoom_short_edge = edge_mut_range['too_short']
        zoom_at_border = zoom_short_edge * self.terminal_pbs['border_zoom_mul']  # 如新增在两端，用此确定切割的edge值

        if zoom_short_edge < sep * 3:
            zoom_short_edge = sep * 3  # 切割两端都需要至少（可等于）保留一个sep，故3

        cut_tag_list = list(range(len(mapping_list_new)))
        cut_pb_each = Tools.probability_each(object_num=len(cut_tag_list), pb_for_all=cut_pb)
        add_num = 0
        for i in cut_tag_list:
            if random.random() < cut_pb_each:

                if i == 0:
                    new_edge = Tools.fit_to_minimal(edge_list_new[0] - zoom_at_border, min_range=sep)

                elif i == (len(cut_tag_list) - 1):
                    new_edge = Tools.fit_to_minimal(edge_list_new[-1] + zoom_at_border, min_range=sep)

                else:
                    edge_before = edge_list_new[i + add_num - 1]
                    edge_after = edge_list_new[i + add_num]
                    cut_zoom = abs(edge_after - edge_before)

                    if cut_zoom < zoom_short_edge:
                        # print('zoom too short, no cut within. issue: 1965')
                        new_edge = None

                    else:
                        cut_zoom = cut_zoom - 2 * sep  # 保留sep后的真实可切割空间
                        # new_edge = (edge_before + edge_after) / 2  # 最简单的方式是取中间值，但不够随机
                        new_edge = Tools.fit_to_minimal(edge_before + sep + cut_zoom * random.random(),
                                                        min_range=sep)  # 升级版处理方式：保留sep的随机切割

                if new_edge is not None:
                    edge_list_new.insert(i, new_edge)
                    add_value = mapping_list_new[i + add_num]
                    mapping_list_new.insert(i + add_num, add_value)  # 仅仅切割开，并不改变赋值

                    add_num += 1

        node_updated = self.__update_node_terminal(map_value_list=mapping_list_new, class_args_edges=edge_list_new)

        return node_updated

    # LV.4 -----------------------------------------------------------------------------
    def __mutate_mapping_list_clear_between(self):
        """mapping_list 同值间异类剔除【半优化】
        注：这个功能在LV.5 中有重复，且更科学（所以这里的概率设置低一些）
        """

        clear_pb = self.terminal_pbs['clear_pb']
        mapping_list_new = self.node_data['map_value_list'].copy()
        edge_list_new = self.node_data['class_args_edges'].copy()
        sep = self.__get_edge_mut_range()['sep']

        clear_order_list = []
        last_value_1 = None  # last value in mapping_list
        last_value_2 = None  # last value of the last value

        num = -1  # NOTE, different here.
        for value in mapping_list_new:

            if last_value_2 == value:  # 前前一个等于当前 --既包括中间异类，也包括中间同类
                clear_order_list.append(num)

            last_value_2 = last_value_1
            last_value_1 = value
            num += 1

        if not clear_order_list:
            return False  # 不存在同值间异

        clear_pb_each = Tools.probability_each(object_num=len(clear_order_list), pb_for_all=clear_pb)
        pop_num = 0
        for i in clear_order_list:
            if random.random() < clear_pb_each:

                old_edge_before = edge_list_new[i - pop_num - 1]
                old_edge_after = edge_list_new[i - pop_num]
                cut_zoom = old_edge_after - old_edge_before
                new_edge_value = Tools.fit_to_minimal(old_edge_before + cut_zoom * random.random(),
                                                      min_range=sep)  # 处理方式：随机切割

                # 如在fit_to_minimal之后和两端相等，那就没法切割出有意义的赋值区间了
                if old_edge_before < new_edge_value < old_edge_after:
                    edge_list_new.pop(i - pop_num)
                    edge_list_new.pop(i - pop_num - 1)
                    edge_list_new.insert(i - pop_num - 1, new_edge_value)

                    mapping_list_new.pop(i - pop_num)  # 清除中间值，两头值并没有合并
                    pop_num += 1

        node_updated = self.__update_node_terminal(map_value_list=mapping_list_new, class_args_edges=edge_list_new)

        return node_updated

    # LV.4 -----------------------------------------------------------------------------
    def __mutate_mapping_list_jump_smooth(self):
        """mapping_list 跳值平滑（插入）【优化】"""

        smooth_pb = self.terminal_pbs['smooth_pb']
        smooth_zoom_mul = self.terminal_pbs['smooth_zoom_mul']  # 抹掉分类的临界值 的 倍数
        mapping_list_new = self.node_data['map_value_list'].copy()
        edge_list_new = self.node_data['class_args_edges'].copy()

        edge_mut_range = self.__get_edge_mut_range()
        add_zoom = edge_mut_range['too_short'] * smooth_zoom_mul  # 预先设定zoom值（固定），插入。后期可考虑动态调整
        sep = edge_mut_range['sep']

        add_tag_list = []
        last_value_1 = None  # last value in mapping_list
        num = 0
        for value in mapping_list_new:

            if last_value_1 is None:
                last_value_1 = value
                num += 1
                continue

            if value * last_value_1 < 0:
                add_tag_list.append(num)

            last_value_1 = value
            num += 1

        if not add_tag_list:
            return False  # 不存在跳值

        smooth_pb_each = Tools.probability_each(object_num=len(add_tag_list), pb_for_all=smooth_pb)
        add_num = 0
        for i in add_tag_list:

            if random.random() < smooth_pb_each:
                mapping_list_new.insert(i + add_num, 0)  # 插入0到符号变化之间，以平滑

                arg_tag = i + add_num - 1
                old_edge = edge_list_new[i - 1]
                new_edge_before = Tools.fit_to_minimal(old_edge - add_zoom / 2, min_range=sep)
                new_edge_after = Tools.fit_to_minimal(old_edge + add_zoom / 2, min_range=sep)

                edge_list_new.pop(arg_tag)
                edge_list_new.insert(arg_tag, new_edge_after)
                edge_list_new.insert(arg_tag, new_edge_before)

        node_updated = self.__update_node_terminal(map_value_list=mapping_list_new, class_args_edges=edge_list_new)

        return node_updated

    # LV.4 -----------------------------------------------------------------------------
    def __mutate_mapping_list_merge_same(self):
        """mapping_list 连续同值合并【优化】"""

        merge_pb = self.terminal_pbs['merge_pb']
        mapping_list_new = self.node_data['map_value_list'].copy()
        edge_list_new = self.node_data['class_args_edges'].copy()

        merge_tag_list = []
        last_value_1 = None  # last value in mapping_list
        num = 0
        for value in mapping_list_new:

            if value == last_value_1:
                merge_tag_list.append(num)  # 记录相同赋值的后者的下标

            last_value_1 = value
            num += 1

        if not merge_tag_list:
            return False  # 不存连续部分

        merge_pb_each = Tools.probability_each(object_num=len(merge_tag_list), pb_for_all=merge_pb)
        pop_num = 0
        for i in merge_tag_list:

            if len(edge_list_new) <= 1:
                break  # 如出现多个（如2个）切割edge，但对应的value（3个）相同时，有可能同时被pop，故限制

            if random.random() < merge_pb_each:
                mapping_list_new.pop(i - pop_num)
                edge_list_new.pop(i - pop_num - 1)  # 相同赋值的后者的下标，对应于前一个下标的切割器
                pop_num += 1

        node_updated = self.__update_node_terminal(map_value_list=mapping_list_new, class_args_edges=edge_list_new)

        return node_updated

    # LV.4 -----------------------------------------------------------------------------
    def __mutate_mapping_list_multiplier(self):

        if not random.random() < self.terminal_pbs['remul_pb']:
            return False

        keep = self.node_data['multiplier_ref_range']['keep']
        if keep:
            start = self.node_data['multiplier_ref_range']['start']
            end = self.node_data['multiplier_ref_range']['end']
        else:
            start = 1.1  # >1: 有效范围
            end = None
        sep = self.node_data['multiplier_ref_range']['sep']

        mul_ref = Tools.mutate_value(self.node_data['multiplier_ref'], start_value=start, end_value=end, sep=sep)

        value_num = len(self.node_data['class_args_edges']) + 1  # value 比 edge 多一个
        map_value_list = []
        for value in np.geomspace(1 / mul_ref, mul_ref, num=value_num):
            map_value_list.append(self.fit_to_minimal(value, min_range=0.001))

        node_updated = self.__update_node_terminal(map_value_list=map_value_list, multiplier_ref=mul_ref)

        return node_updated

    # LV.5 ----------------------------------------------------------------------------
    def __mutate_edge(self):
        """LV.5 MUTATION: 特征分类截取进化函数，对edge的值和数量进行优化"""

        mutation_tag = False
        map_type = self.node_data['map_type']
        now_num = len(self.node_data['map_value_list'])

        keep = self.node_data['edge_num_range']['keep']
        if keep:
            start = self.node_data['edge_num_range']['start']
            end = self.node_data['edge_num_range']['end']
        else:
            start = None
            end = None

        if map_type == 'vector' or map_type == 'condition':

            # 1. edge 太近删除
            if 'pop_pb' in self.terminal_pbs and self.terminal_pbs['pop_pb'] \
                    and Tools.if_in_range(now_num - 1, start, end):

                poped = self.__mutate_edge_pop()
                if poped:
                    mutation_tag = True
                    print('lv.5 mutation: class_edge poped.')

            # 2. edge 太远插入
            if 'insert_pb' in self.terminal_pbs and self.terminal_pbs['insert_pb'] \
                    and Tools.if_in_range(now_num + 2, start, end):

                inserted = self.__mutate_edge_insert()
                if inserted:
                    mutation_tag = True
                    print('lv.5 mutation: class_edge inserted.')

            # 3. edge 切割点移动
            if 'move_pb' in self.terminal_pbs and self.terminal_pbs['move_pb']:
                moved = self.__mutate_edge_move()
                if moved:
                    mutation_tag = True
                    print('lv.5 mutation: class_edge moved.')

        elif map_type == 'multiplier':

            edge_num = len(self.node_data['class_args_edges'])
            if 'pop_pb' in self.terminal_pbs and random.random() < self.terminal_pbs['pop_pb']:
                edge_num -= 1

            if 'insert_pb' in self.terminal_pbs and random.random() < self.terminal_pbs['insert_pb']:
                edge_num += 1

            if 'move_pb' in self.terminal_pbs:
                moved = self.__mutate_edge_move()
                if moved:
                    print('lv.5 mutation: edge moved.')
                    mutation_tag = True

            num_change = self.__mutate_edge_mult_change(edge_num)
            if num_change:
                mutation_tag = True

        return mutation_tag

    # LV.5 ----------------------------------------------------------------------------
    def __mutate_edge_move(self):
        """切割边界变异：移动边界"""

        move_pb = self.terminal_pbs['move_pb']
        edge_list = self.node_data['class_args_edges']
        edge_mut_range = self.__get_edge_mut_range()
        sep = edge_mut_range['sep']
        keep = self.node_data['edge_mut_range_keep']
        if keep['start']:
            start = edge_mut_range['start']
        else:
            start = None
        if keep['end']:
            end = edge_mut_range['end']
        else:
            end = None

        edge_list_new = []  # 这里的写法与 mutate_mapping_list_* 中的不同，因之前考虑到部分参数不是edge的情况（现已删除相关检验）
        move_pb_each = Tools.probability_each(object_num=len(edge_list), pb_for_all=move_pb)
        for edge in edge_list:

            if random.random() < move_pb_each:
                new_edge = Tools.mutate_value(edge, start_value=start, end_value=end, sep=sep)

                n = 0
                while Tools.check_in_list(new_edge, edge_list):
                    if random.random() < 0:
                        new_edge -= sep
                    else:
                        new_edge += sep

                edge_list_new.append(new_edge)

            else:
                edge_list_new.append(edge)

        node_updated = self.__update_node_terminal(class_args_edges=edge_list_new)

        return node_updated

    # LV.5 ----------------------------------------------------------------------------
    def __mutate_edge_insert(self):
        """切割边界变异：太远了，插入"""

        insert_pb = self.terminal_pbs['insert_pb']
        zoom_distance_mul = self.terminal_pbs['zoom_distance_mul']
        edge_list = self.node_data['class_args_edges']
        edge_list_new = self.node_data['class_args_edges'].copy()
        mapping_list = self.node_data['map_value_list']
        mapping_list_new = self.node_data['map_value_list'].copy()

        edge_mut_range = self.__get_edge_mut_range()
        long_edge = edge_mut_range['too_long']
        zoom_distance_edge = edge_mut_range['too_short'] * zoom_distance_mul
        sep = edge_mut_range['sep']

        long_tag_list = []
        old_edge = None
        num = 0
        for edge in edge_list:

            if old_edge is not None:
                if edge - old_edge > long_edge:
                    long_tag_list.append(num)

            old_edge = edge
            num += 1

        if not long_tag_list:
            return False  # 不存在太长的区间

        insert_pb_each = Tools.probability_each(object_num=len(long_tag_list), pb_for_all=insert_pb)
        add_num = 0
        for i in long_tag_list:

            if random.random() < insert_pb_each:

                edge_before = edge_list_new[i + add_num - 1]
                edge_after = edge_list_new[i + add_num]

                zoom_line_before = edge_before + zoom_distance_edge  # 真实取值起点，与前方edge保持一定距离
                zoom_line_after = edge_after - zoom_distance_edge
                zoom = zoom_line_after - zoom_line_before

                if zoom < zoom_distance_edge:
                    continue  # double check, to see if the zoom is big enough for a cut in.

                point1 = Tools.fit_to_minimal(zoom_line_before + zoom * random.random(), min_range=sep)
                point2 = Tools.fit_to_minimal(zoom_line_before + zoom * random.random(), min_range=sep)

                if point1 != point2:
                    edge_list_new.insert(i + add_num, max(point1, point2))
                    edge_list_new.insert(i + add_num, min(point1, point2))

                    cut_value = mapping_list[i]
                    mapping_list_new.insert(i + add_num, self.__mutate_one_value_in_map(cut_value))
                    mapping_list_new.insert(i + add_num, cut_value)  # cut_value中间夹着一个变异后的value

                    add_num += 2  # NOTE 2 here.

        node_updated = self.__update_node_terminal(map_value_list=mapping_list_new, class_args_edges=edge_list_new)

        return node_updated

    # LV.5 ----------------------------------------------------------------------------
    def __mutate_edge_pop(self):
        """切割边界变异：太近了，删除"""

        pop_pb = self.terminal_pbs['pop_pb']
        edge_list = self.node_data['class_args_edges']
        edge_list_new = self.node_data['class_args_edges'].copy()
        mapping_list_new = self.node_data['map_value_list'].copy()
        short_edge = self.__get_edge_mut_range()['too_short']

        short_tag_list = []
        old_edge = None
        num = 0
        for edge in edge_list:

            if old_edge is not None:
                if edge - old_edge < short_edge:
                    short_tag_list.append(num)

            old_edge = edge
            num += 1

        if not short_tag_list:
            return False  # no short zoom

        pop_pb_each = Tools.probability_each(object_num=len(short_tag_list), pb_for_all=pop_pb)
        pop_num = 0
        for i in short_tag_list:

            if random.random() < pop_pb_each:
                edge_list_new.pop(i - pop_num)
                mapping_list_new.pop(i - pop_num)
                pop_num += 1

        node_updated = self.__update_node_terminal(map_value_list=mapping_list_new, class_args_edges=edge_list_new)

        return node_updated

    # LV.5 ----------------------------------------------------------------------------
    def __mutate_edge_mult_change(self, target_edge_num):

        # 1. 数量检验
        edge_list = self.node_data['class_args_edges']
        if target_edge_num == len(edge_list):
            return False  # 所需edge数量已满足，无需改变什么

        # 2. 范围检验
        keep = self.node_data['edge_num_range']['keep']
        if keep:
            start = self.node_data['edge_num_range']['start']
            end = self.node_data['edge_num_range']['end']
        else:
            start = None
            end = None
        if not Tools.if_in_range(target_edge_num, start, end):
            return False  # 所需改变超出范围

        # 3. 开始处理edge --pop/insert  --最简单的做法：edge中间随机加一刀/减一刀
        edge_list_new = self.node_data['class_args_edges'].copy()
        mul_ref = self.node_data['multiplier_ref']
        edge_sep = self.__get_edge_mut_range()['sep']

        if target_edge_num < len(edge_list):

            remove_edge = random.choice(edge_list)
            edge_list_new.remove(remove_edge)
            print('lv.5 mutation: mult edge poped.')

        elif target_edge_num > len(edge_list):

            edge_min = min(edge_list)
            edge_max = max(edge_list)

            if edge_sep is True:
                edge_sep = (edge_max - edge_min) / 50
                if isinstance(edge_min, float):
                    keep_float = len(str(edge_min).split('.')[1])
                    edge_sep = self.shorten_float(edge_sep, keep_float)
                else:
                    edge_sep = int(edge_sep)

            choice_box = np.arange(edge_min + edge_sep, edge_max, edge_sep)
            if not list(choice_box):
                return False

            new_edge = np.random.choice(choice_box)
            while new_edge in edge_list:
                new_edge += edge_sep  # no same edge value

            edge_list_new.append(new_edge)
            print('lv.5 mutation: mult edge inserted.')

        # 4. 更新map_value
        value_num = len(edge_list_new) + 1  # value 比 edge 多一个
        mapping_list_new = []
        for value in np.geomspace(1 / mul_ref, mul_ref, num=value_num):
            mapping_list_new.append(self.fit_to_minimal(value, min_range=0.001))

        # 5. update (or not)
        node_updated = self.__update_node_terminal(map_value_list=mapping_list_new, class_args_edges=edge_list_new)

        return node_updated

    # LV.6 ----------------------------------------------------------------------------
    def __mutate_feature_window(self):
        """LV.6 特征参数进化 - window"""

        mutation_tag = False

        if 'window' in self.node_data['class_kw']:
            if 'window_pb' in self.terminal_pbs and random.random() < self.terminal_pbs['window_pb']:
                window = self.node_data['class_kw']['window']
                window_range = self.node_data['feature_window_range']  # or self.node_data['class_kw_range']['window']
                start = window_range['start']
                end = window_range['end']
                sep = window_range['sep']

                new_window = Tools.mutate_value(window, start_value=start, end_value=end, sep=sep)
                self.node_data['class_kw']['window'] = new_window

                print('lv.6 mutation: feature window changed.')
                mutation_tag = True

        return mutation_tag

    # LV.6 ----------------------------------------------------------------------------
    def __mutate_feature_decrease_num(self):

        decreased = False

        feature_num = self.node_data['feature_num']
        sep = self.node_data['feature_num_range']['sep']

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

            decreased = True
            print('lv.6 mutation: feature_num decreased.')

        return decreased

    # LV.6 ----------------------------------------------------------------------------
    def __mutate_feature_increase_num(self):

        increased = False

        feature_num = self.node_data['feature_num']
        sep = self.node_data['feature_num_range']['sep']

        if random.random() < self.terminal_pbs['addfeature_pb']:
            refeature_pb_each = self.terminal_pbs['refeature_pb']  # 特殊556
            indicator = self.node_data['class_args_features_ins'][0].__class__
            instance = indicator(df_source=self.df_source, refeature_pb=refeature_pb_each)
            add_num = random.choice(list(range(feature_num)))

            self.node_data['class_args_features_ins'].insert(add_num, instance)
            self.node_data['class_args_features'].insert(add_num, instance.cal())
            del instance

            self.node_data['feature_num'] += sep

            increased = True
            print('lv.6 mutation: feature_num increased.')

        return increased

    # LV.7 ----------------------------------------------------------------------------
    def __mutate_feature_param_from_dict(self):

        feature_changed = False

        for name, instance in self.node_data['class_kw_ins'].items():
            if isinstance(instance, Indicator):

                mut_flag = instance.mutate_feature()
                if mut_flag:
                    self.node_data['class_kw'][name] = instance.cal()  # get new feature data

                    feature_changed = True
                    print('lv.7 mutation: class_kw instance changed.')

        return feature_changed

    # LV.7 ----------------------------------------------------------------------------
    def __mutate_feature_param_from_list(self):

        feature_changed = False

        for num in range(len(self.node_data['class_args_features_ins'])):

            instance = self.node_data['class_args_features_ins'][num]
            mut_flag = instance.mutate_feature()

            if mut_flag:
                # old_data = self.node_data['class_args_features'][num]
                self.node_data['class_args_features'][num] = instance.cal()
                # del old_data  # not sure if the Series has been deleted from memory.

                feature_changed = True
                print('lv.7 mutation: class_args feature changed.')

        return feature_changed

    # -------------------------------------------------
    def __create_terminal_dict(self, indicator, classifier_detail):

        # require 'edge_mut_range' and 'zoom_distance' in classifier_detail
        self.node_data['edge_mut_range'] = classifier_detail['edge_mut_range'].copy()
        self.node_data['edge_mut_range_keep'] = classifier_detail['edge_mut_range_keep'].copy()
        self.node_data['feature_window_ratio_range'] = classifier_detail['feature_window_ratio_range'].copy()
        self.node_data['multiplier_ref_range'] = classifier_detail['multiplier_ref_range'].copy()
        self.node_data['edge_num_range'] = classifier_detail['edge_num_range'].copy()
        if self.node_data['map_type'] == 'multiplier':
            self.node_data['edge_num_range']['start'] += 1  # mult这种类别需要更多的edge
            self.node_data['edge_num_range']['end'] += 1  # 相反，cond、vector应对应更少的edge，但一开始可以多一点

        self.node_data['class_args_edges'] = self.__generate_random_edges()
        self.node_data['map_value_list'] = self.__generate_random_map_value()

        self.node_data['class_kw'] = {}
        self.node_data['class_kw_ins'] = {}
        self.node_data['class_kw_range'] = {}

        kwarg_list = inspect.getfullargspec(self.node_data['class_func'])[4]
        feature_num = 0
        for kw in kwarg_list:
            if kw[:7] == 'feature':
                feature_num += 1

        refeature_pb_each = Tools.probability_each(object_num=feature_num,
                                                   pb_for_all=self.terminal_pbs['refeature_pb'])
        for kw in kwarg_list:

            if kw == 'window':
                window_keep = self.node_data['feature_window_ratio_range']['keep']
                df_line_max = self.df_source.shape[0]  # window可=
                window_min = int(df_line_max * self.node_data['feature_window_ratio_range']['start'])
                window_sep = self.node_data['feature_window_ratio_range']['sep']  # for sep == True
                if window_sep and window_sep is not True:
                    window_sep = int(df_line_max * window_sep)
                    if window_sep < 1:
                        window_sep = 1
                window_max = int(df_line_max * self.node_data['feature_window_ratio_range']['end'])

                # window不能小于1（0：nan，-1：error），也不能大于df.shape[0] (nan)   -- keep is always True
                if window_keep:
                    self.node_data['feature_window_range'] = {'keep': True, 'start': window_min, 'end': window_max,
                                                              'sep': window_sep}
                else:
                    self.node_data['feature_window_range'] = {'keep': True, 'start': 1, 'end': df_line_max,
                                                              'sep': True}

                self.node_data['class_kw'][kw] = random.choice(list(range(window_min, window_max)))
                self.node_data['class_kw_ins'][kw] = None
                self.node_data['class_kw_range'][kw] = self.node_data[
                    'feature_window_range']  # 当前window特殊出来。但也在kw_range里面放一下吧。。

            elif kw[:7] == 'feature':
                instance = indicator(df_source=self.df_source, refeature_pb=refeature_pb_each)  # 创建实例
                self.node_data['class_kw'][kw] = instance.cal()  # 计算，获得feature
                self.node_data['class_kw_ins'][kw] = instance
                self.node_data['class_kw_range'][kw] = None

            else:
                raise KeyError('Unknown keyword in class_function: %s. 9496' % self.node_data['class_func'])

    # -------------------------------------------------
    def __create_terminal_list(self, indicator, classifier_detail):

        self.node_data['map_value_list'] = classifier_detail['map_value_list']
        self.node_data['feature_num_range'] = classifier_detail['feature_num_range'].copy()
        self.node_data['class_args_features'] = []
        self.node_data['class_args_features_ins'] = []

        start = classifier_detail['feature_num_range']['start']
        sep = classifier_detail['feature_num_range']['sep']
        end = classifier_detail['feature_num_range']['end'] + sep
        feature_num = int(np.random.choice(np.arange(start, end, sep)))
        self.node_data['feature_num'] = feature_num

        if not classifier_detail['feature_num_range']['keep']:
            self.node_data['feature_num_range']['start'] = 2  # >=2： 函数的有效范围
            self.node_data['feature_num_range']['end'] = None
            self.node_data['feature_num_range']['keep'] = True  # it's always True after initialization.

        # refeature_pb_each = Tools.probability_each(object_num=feature_num,
        #                                           pb_for_all=self.terminal_pbs['refeature_pb'])
        refeature_pb_each = self.terminal_pbs['refeature_pb']  # 特殊556

        for num in range(feature_num):
            instance = indicator(df_source=self.df_source, refeature_pb=refeature_pb_each)  # 创建实例
            self.node_data['class_args_features_ins'].append(instance)
            self.node_data['class_args_features'].append(instance.cal())

    # -------------------------------------------------
    def __should_be_copied(self, data):

        if isinstance(data, dict):
            return True
        if isinstance(data, list):
            return True
        if isinstance(data, Indicator):
            return True
        if isinstance(data, Terminal):
            return True
        if isinstance(data, pd.Series):
            return True
        if isinstance(data, pd.DataFrame):
            return False  # note
        return False

    # -------------------------------------------------
    def __copy_dict(self, dict_data):

        new_dict = dict_data.copy()

        for key, value in dict_data.items():
            if self.__should_be_copied(value):
                new_dict[key] = dict_data[key].copy()
            if isinstance(value, dict):
                new_dict[key] = self.__copy_dict(value)
                
        return new_dict

    # ------------------------------------------------------------------------------------------------------------------
    def create_terminal(self):
        """create a random terminal."""

        indicator = self.__get_indicator()
        class_func = self.__get_classifier_function(indicator)
        classifier_detail = self.__get_classifier_detail(class_func)

        # class function
        self.node_data['class_func'] = class_func

        # map_type
        map_type_box = set(indicator.map_type) & set(classifier_detail['map_type'])  # 取交集
        if not map_type_box:
            print('IndexError: Cannot choose from an empty sequence!!!! %s, %s' % (indicator, classifier_detail))
            self.create_terminal()  # 如匹配错误，重新生成。
        self.node_data['map_type'] = random.choice(list(map_type_box))

        # node_type (output value type)
        if self.node_data['map_type'] == 'condition':
            self.node_data['node_type'] = 'abs_value'
        else:
            self.node_data['node_type'] = 'pos_value'

        # function group
        func_group = None
        for name, group in self.classifier_group.items():
            if class_func in group:
                func_group = name
        self.node_data['class_func_group'] = func_group

        # others
        if func_group == 'cut' or func_group == 'compare':
            self.__create_terminal_dict(indicator, classifier_detail)

        elif func_group == 'permutation' or func_group == 'trend':
            self.__create_terminal_list(indicator, classifier_detail)

        else:
            raise ValueError('Uncategorized class_function: %s. 9484' % class_func)

        self.lv_mut_tag = Classifier.lv_mut_tag.copy()
        self.node_result = self.cal()

        return self.node_result

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
                    node_data[key] = self.node_data[key].copy()

                    for key2, value2 in value.items():
                        if isinstance(value2, pd.Series):
                            node_data[key][key2] = self.node_data[key][key2].copy()
                            node_data[key][key2] = pd.Series()

        return node_data

    def copy(self):

        new_instance = self.__class__(df_source=self.df_source,
                                      terminal_pbs=self.terminal_pbs,
                                      classifier_map=self.classifier_map,
                                      classifier_group=self.classifier_group,
                                      node_data=self.node_data,
                                      lv_mut_tag=self.lv_mut_tag)

        new_instance.__dict__ = self.__dict__.copy()
        new_instance.name = self.get_id('%s' % self.name.split('_')[0])

        # node_data深拷贝, 其他：score_list应该能自动返回veiw，node_result应能自动返回copy
        new_instance.__dict__['node_data'] = self.__copy_dict(self.__dict__['node_data'])

        # 旧方法备份：（比较精细，但太长了，还是全部都复制简单些）
        # new_node_data = new_instance.__dict__['node_data']  # view

        # if new_node_data['class_args_features_ins']:
        #     new_instance_list = []
        #     for instance in self.node_data['class_args_features_ins']:
        #         new_instance_list.append(instance.copy())

        #     new_node_data['class_args_features_ins'] = new_instance_list.copy()

        # if new_node_data['class_args_features']:
        #     new_node_data['class_args_features'] = self.node_data['class_args_features'].copy()

        # for name in ['class_kw', 'class_kw_ins', 'class_kw_range']:

        #     if new_node_data[name]:
        #         new_node_data[name] = self.node_data[name].copy()  # 字典

        #         for key, value in self.node_data[name].items():

        #             if isinstance(value, Indicator): 
        #                 new_node_data[name][key] = value.copy()  # class_kw_ins data
        #             elif isinstance(value, dict):
        #                 new_node_data[name][key] = value.copy()  # class_kw_range data
        #             elif isinstance(value, pd.Series):
        #                 new_node_data[name][key] = value.copy()  # class_kw data

        return new_instance

    def add_score(self, score):

        self.score_list.append(score)

        if self.node_data['class_args_features_ins']:
            indicator_list = []
            for instance in self.node_data['class_args_features_ins']:
                feature_class = instance.__class__  # 避免重复添加
                if not feature_class in indicator_list:
                    indicator_list.append(feature_class)
                    instance.add_score(score)

        if self.node_data['class_kw_ins']:
            indicator_list = []
            for instance in self.node_data['class_kw_ins'].values():
                if isinstance(instance, Indicator):
                    feature_class = instance.__class__
                    if not feature_class in indicator_list:
                        indicator_list.append(feature_class)
                        instance.add_score(score)

    def update_score(self, sortino_score):

        self.avg_score = (self.avg_score * self.score_num + sortino_score) / (self.score_num + 1)
        self.score_num += 1

        if self.node_data['class_args_features_ins']:
            indicator_list = []
            for instance in self.node_data['class_args_features_ins']:
                feature_class = instance.__class__
                if not feature_class in indicator_list:
                    indicator_list.append(feature_class)
                    instance.update_score(sortino_score)

        if self.node_data['class_kw_ins']:
            indicator_list = []
            for instance in self.node_data['class_kw_ins'].values():
                if isinstance(instance, Indicator):
                    feature_class = instance.__class__
                    if not feature_class in indicator_list:
                        indicator_list.append(feature_class)
                        instance.update_score(sortino_score)

    def mutate_terminal(self):

        # LV.2 权重变异
        self.lv_mut_tag[2] = Tools.mutate_weight(self.node_data, reweight_pb=self.terminal_pbs['reweight_pb'])

        # LV.4 特征分类赋值变异
        self.lv_mut_tag[4] = self.__mutate_mapping_list()

        func_group = self.node_data['class_func_group']

        if func_group == 'cut' or func_group == 'compare':

            self.lv_mut_tag[5] = self.__mutate_edge()  # LV.5 特征分类截取变异
            self.lv_mut_tag[6] = self.__mutate_feature_window()  # LV.6 特征变异-window
            self.lv_mut_tag[7] = self.__mutate_feature_param_from_dict()  # LV.7 feature计算参数变异

        elif func_group == 'permutation' or func_group == 'trend':

            feature_num_max = self.node_data['feature_num_range']['end']
            feature_num_min = self.node_data['feature_num_range']['start']
            sep = self.node_data['feature_num_range']['sep']  # 就是1
            feature_num = self.node_data['feature_num']

            if Tools.if_in_range(feature_num - sep, feature_num_min, feature_num_max):
                decreased = self.__mutate_feature_decrease_num()  # lv.6 减少feature数量
                if decreased:
                    self.lv_mut_tag[6] = True

            if Tools.if_in_range(feature_num + sep, feature_num_min, feature_num_max):
                increased = self.__mutate_feature_increase_num()  # lv.6 增加feature数量
                if increased:
                    self.lv_mut_tag[6] = True

            self.lv_mut_tag[7] = self.__mutate_feature_param_from_list()  # lv.7 改变feature计算参数

        mutation_tag = Tools.check_any_in_list(True, self.lv_mut_tag.values())

        return mutation_tag

    def recal(self):

        self.lv_mut_tag = Classifier.lv_mut_tag.copy()
        self.cal()

    def cal(self):

        # TODO: test分级计算

        if self.lv_mut_tag[7] or self.lv_mut_tag[6] or self.lv_mut_tag[5]:  # lv.7, lv.6, lv.5

            func_group = self.node_data['class_func_group']

            if func_group == 'cut' or func_group == 'compare':

                func = self.node_data['class_func']
                args = self.node_data['class_args_edges']
                kwargs = self.node_data['class_kw']
                self.class_data = func(*args, **kwargs)

            elif func_group == 'permutation' or func_group == 'trend':

                func = self.node_data['class_func']
                args = self.node_data['class_args_features']
                self.class_data = func(*args)

            self.terminal_result = self.__get_mapped_data()
            self.node_result = Tools.cal_weight(self.terminal_result, self.node_data['weight'])

        elif self.lv_mut_tag[4]:  # lv.4
            self.terminal_result = self.__get_mapped_data()
            self.node_result = Tools.cal_weight(self.terminal_result, self.node_data['weight'])

        elif self.lv_mut_tag[2]:  # lv.2
            self.node_result = Tools.cal_weight(self.terminal_result, self.node_data['weight'])

        for key in self.lv_mut_tag.keys():
            self.lv_mut_tag[key] = False  # reset mutation_tag

        return self.node_result


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
