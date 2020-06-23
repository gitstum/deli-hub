import os
import time
import datetime
import pandas as pd
import numpy as np
import random

import empyrical

import multiprocessing as mp

from backtest.constant import *


class Tools(object):

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def agg_cal(func, *args, process_num=None):
        """Multi-processing framework for functionX with agg_cal:

        To call:
            result1, result2 = functionX.agg_cal((param1,), (param2,))
            result1, result2 = Tools.agg_cal(functionX,
                                             (param1,), (param2,),
                                             process_num=4
                                            )
        """

        result_tags = []
        results = []

        if not process_num:
            pool = mp.Pool()
        else:
            pool = mp.Pool(processes=process_num)

        for i in args:
            tag = pool.apply_async(func, i)  # note i must be tuple
            result_tags.append(tag)

        pool.close()
        pool.join()

        for tag in result_tags:
            result = tag.get()
            results.append(result)

        return results

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def str_time(f=6):
        """获取str时间信息

        @param f: 秒后要保留的小数位
        @return: str时间
        """

        if f == 0: f = -1
        f -= 6
        if f > 0: f = 0

        if f == 0:
            now = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        else:
            now = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))[:f]

        return now

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def keep_float(number, float_num):
        """float简化：保留小数点后x位

        @param number: 要化简的数
        @param float_num: 要保留的小数位
        @return: 化简后的数
        """

        number = number * 10 ** float_num
        number = int(number)
        number = number / 10 ** float_num

        return number

    # 基因树 相关函数 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def mutate_value_map(node_data, *, mut_pb=0.25, jump_pb=0.1, cut_pb=0.05, clear_pb=0.05, merge_pb=0.15, smooth_pb=0.1):
        """
        @param node_data: dict of the node in node_map, include:
            - 'value_map': the LIST to get mutated.
            - 'map_type': to direct the mutation.
            - 'classify_args': to be changed if any mutation happened on value_map.
        @param mut_pb: probability of any  mutation happened.
        @param merge_pb: probability of merging 2(or more) same value into 1 value.
        @param cut_pb: probability of cutting 1 value into 2. ('classify_args' cut at random distance)
        @param clear_pb: probability of deleting the value between 2 same value.
        @param smooth_pb: probability of adding a 0 value betweent -1 & 1 (for 'vector' map_type only)
        @return: True for any mutation happened for value_map(and classify_args)

        NOTE: inplace. all pb are independent.
        """

        # TODO: test it.

        mutation_tag = False
        classify_addable = node_data['classify_args_addable']  # NOTE!!!!!!!!!!!!!!!!!!!!!!!! 
        map_type = node_data['map_type'] 
        classify_edge = node_data['classify_args']  # 分类的切割点边界值 

        # 注意，下面5项的顺序是有考虑的，且完成一个到下一个，不要随意改变先后次序或合并

        # 1. value_map 连续同值合并【优化】 --merge_pb
        if merge_pb and len(classify_edge) > 1:
            merged = Tools.mutate_value_map_merge_same(node_data, merge_pb=merge_pb)
            if merged:
                mutation_tag = True
                # print('merged')

        # 2. value_map 跳值平滑【优化】  --smooth_pb
        if map_type == 'vector' and classify_addable and smooth_pb:
            smoothed = Tools.mutate_value_map_jump_smooth(node_data, smooth_pb=smooth_pb)
            if smoothed:
                mutation_tag = True
                # print('smoothed')

        # 3. value_map 同值间异类剔除【半优化】  --clear_pb
        if clear_pb:
            cleared = Tools.mutate_value_map_clear_between(node_data, clear_pb=clear_pb)
            if cleared:
                mutation_tag = True
                # print('cleared')

        # 4. value_map 数目增加 --cut_pb
        if classify_addable and cut_pb:
            cut = Tools.mutate_value_map_cut_within(node_data, cut_pb=cut_pb)
            if cut:
                mutation_tag = True
                # print('cut')

        # 5. value_map 赋值变异 --mut_pb  (这一步才是这个函数正儿八经最应该做的事情）

        value_map = node_data['value_map']
        value_map_new = node_data['value_map'].copy()

        mut_pb_each = Tools.probability_each(object_num=len(value_map),
                                             pb_for_all=mut_pb)  # 各元素发生变异的独立概率

        n = 0
        for value in value_map:
            if random.random() < mut_pb_each:
                new_value = Tools.mutate_one_value_in_map(value, map_type=map_type, jump_pb=jump_pb)
                value_map_new[n] = new_value
            n += 1

        if value_map_new != value_map:
            node_data['value_map'] = value_map_new.copy()
            mutation_tag = True
            # print('value_changed.')

        return mutation_tag

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def mutate_value_map_cut_within(node_data, *, cut_pb, add_zoom_mul=1.5):
        """value_map 数目增加：通过将某段赋值切开（但赋值仍相同）

        注：这个功能在lv4 中有重复，且更科学（所以这里的概率设置低一些）
        """

        classify_edge_new = node_data['classify_args'].copy()
        if not classify_edge_new:
            return False  # 缺乏切割的参考edge值，会导致出错。

        value_map_new = node_data['value_map'].copy()
        zoom_of_sep = node_data['classify_args_sep']
        zoom_short_edge = node_data['classify_args_short_edge']
        zoom_at_border = zoom_short_edge * add_zoom_mul  # 如新增在两端，用此确定切割的edge值
        if zoom_short_edge < zoom_of_sep * 3:
            zoom_short_edge = zoom_of_sep * 3  # 切割两端都需要至少（可等于）保留一个sep，故3

        cut_tag_list = list(range(len(value_map_new)))

        cut_pb_each = Tools.probability_each(object_num=len(cut_tag_list),
                                             pb_for_all=cut_pb)

        add_num = 0
        for i in cut_tag_list:
            if random.random() < cut_pb_each:

                if i == 0:
                    new_edge = Tools.fit_to_minimal(classify_edge_new[0] - zoom_at_border,
                                                    min_range=zoom_of_sep)

                elif i == (len(cut_tag_list) - 1):
                    new_edge = Tools.fit_to_minimal(classify_edge_new[-1] + zoom_at_border,
                                                    min_range=zoom_of_sep)

                else:
                    edge_before = classify_edge_new[i + add_num - 1]
                    edge_after = classify_edge_new[i + add_num]
                    cut_zoom = abs(edge_after - edge_before)
                    if cut_zoom < zoom_short_edge:
                        print('zoom too short, no cut within. issue: 1965')
                        new_edge = None
                    else:
                        cut_zoom = cut_zoom - 2 * zoom_of_sep  # 保留sep后的真实可切割空间
                        # new_edge = (edge_before + edge_after) / 2  # 最简单的方式是取中间值，但不够随机
                        new_edge = Tools.fit_to_minimal(edge_before + zoom_of_sep + cut_zoom * random.random(),
                                                        min_range=zoom_of_sep)  # 升级版处理方式：保留sep的随机切割

                if new_edge is not None:
                    classify_edge_new.insert(i, new_edge)
                    add_value = value_map_new[i + add_num]
                    value_map_new.insert(i + add_num, add_value)  # 仅仅切割开，并不改变赋值

                    add_num += 1

        if value_map_new == node_data['value_map']:
            return False

        else:
            sorted(classify_edge_new)  # 记得要排序一下！
            node_data['value_map'] = value_map_new.copy()
            node_data['classify_args'] = classify_edge_new.copy()

            return True

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def mutate_value_map_clear_between(node_data, *, clear_pb):
        """value_map 同值间异类剔除【半优化】

        注：这个功能在lv4 中有重复，且更科学（所以这里的概率设置低一些）
        """

        value_map_new = node_data['value_map'].copy()
        classify_edge_new = node_data['classify_args'].copy()
        zoom_of_sep = node_data['classify_args_sep']

        clear_order_list = []
        last_value_1 = None  # last value in value_map
        last_value_2 = None  # last value of the last value

        num = -1  # NOTE, different here.
        for value in value_map_new:

            if last_value_2 == value:  # 前前一个等于当前 --既包括中间异类，也包括中间同类
                clear_order_list.append(num)

            last_value_2 = last_value_1
            last_value_1 = value
            num += 1

        if not clear_order_list:
            return False  # 不存在同值间异

        clear_pb_each = Tools.probability_each(object_num=len(clear_order_list),
                                               pb_for_all=clear_pb)
        pop_num = 0
        for i in clear_order_list:
            if random.random() < clear_pb_each:

                old_edge_before = classify_edge_new[i - pop_num - 1]
                old_edge_after = classify_edge_new[i - pop_num]
                cut_zoom = old_edge_after - old_edge_before
                new_edge_value = Tools.fit_to_minimal(old_edge_before + cut_zoom * random.random(),
                                                      min_range=zoom_of_sep)  # 处理方式：随机切割

                # 如在fit_to_minimal之后和两端相等，那就没法切割出有意义的赋值区间了
                if old_edge_before < new_edge_value < old_edge_after:
                    classify_edge_new.pop(i - pop_num)
                    classify_edge_new.pop(i - pop_num - 1)
                    classify_edge_new.insert(i - pop_num - 1, new_edge_value)

                    value_map_new.pop(i - pop_num)  # 清除中间值，两头值并没有合并
                    pop_num += 1

        if value_map_new == node_data['value_map']:
            return False

        else:
            sorted(classify_edge_new)  # 记得要排序一下！
            node_data['value_map'] = value_map_new.copy()
            node_data['classify_args'] = classify_edge_new.copy()

            return True

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def mutate_value_map_jump_smooth(node_data, *, smooth_pb, smooth_zoom_mul=1.5):
        """value_map 跳值平滑（插入）【优化】

        @param node_data:
        @param smooth_pb:
        @param smooth_zoom_mul: lv4 mutation 中抹掉分类的临界值的倍数
        """

        value_map_new = node_data['value_map'].copy()
        classify_edge_new = node_data['classify_args'].copy()

        # 这里的做法是，预先设定zoom值（固定），插入。后期可考虑动态调整zoom值(要考虑前后边界的情况：缺失、太窄等)
        add_zoom = node_data['classify_args_short_edge'] * smooth_zoom_mul
        zoom_of_sep = node_data['classify_args_sep']

        add_tag_list = []
        last_value_1 = None  # last value in value_map

        num = 0

        for value in value_map_new:

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

        smooth_pb_each = Tools.probability_each(object_num=len(add_tag_list),
                                                pb_for_all=smooth_pb)

        add_num = 0
        for i in add_tag_list:
            if random.random() < smooth_pb_each:
                value_map_new.insert(i + add_num, 0)  # 插入0到符号变化之间，以平滑

                arg_tag = i + add_num - 1
                old_edge = classify_edge_new[i - 1]
                new_edge_before = Tools.fit_to_minimal(old_edge - add_zoom / 2,
                                                       min_range=zoom_of_sep)
                new_edge_after = Tools.fit_to_minimal(old_edge + add_zoom / 2,
                                                      min_range=zoom_of_sep)

                classify_edge_new.pop(arg_tag)
                classify_edge_new.insert(arg_tag, new_edge_after)
                classify_edge_new.insert(arg_tag, new_edge_before)

        if value_map_new == node_data['value_map']:
            return False

        else:
            sorted(classify_edge_new)  # 记得要排序一下！
            node_data['value_map'] = value_map_new.copy()
            node_data['classify_args'] = classify_edge_new.copy()

            return True

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def mutate_value_map_merge_same(node_data, *, merge_pb):
        """value_map 连续同值合并【优化】"""

        value_map_new = node_data['value_map'].copy()
        classify_edge_new = node_data['classify_args'].copy()

        merge_tag_list = []
        last_value_1 = None  # last value in value_map

        num = 0
        for value in value_map_new:
            if value == last_value_1:
                merge_tag_list.append(num)  # 记录相同赋值的后者的下标
            last_value_1 = value
            num += 1

        if not merge_tag_list:
            return False  # 不存连续部分

        merge_pb_each = Tools.probability_each(object_num=len(merge_tag_list),
                                               pb_for_all=merge_pb)

        pop_num = 0
        for i in merge_tag_list:

            if len(classify_edge_new) <= 1:
                break  # 如出现多个（如2个）切割edge，但对应的value（3个）相同时，有可能同时被pop，故限制

            if random.random() < merge_pb_each:
                value_map_new.pop(i - pop_num)
                classify_edge_new.pop(i - pop_num - 1)  # 相同赋值的后者的下标，对应于前一个下标的切割器
                pop_num += 1

        if value_map_new == node_data['value_map']:
            return False

        else:
            sorted(classify_edge_new)  # 记得要排序一下！
            node_data['value_map'] = value_map_new.copy()
            node_data['classify_args'] = classify_edge_new.copy()

            return True

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def cal_event_pb(*args):
        """根据传入的独立概率事件，返回: 全部发生的概率，至少一个发生的概率，没有一个发生的概率"""

        all_happen_pb = 1
        no_happen_pb = 1

        for pb in args:
            all_happen_pb = all_happen_pb * pb
            no_happen_pb = no_happen_pb * (1 - pb)

        any_happen_pb = 1 - no_happen_pb

        result = dict(all_happen_pb=all_happen_pb,
                      any_happen_pb=any_happen_pb,
                      no_happen_pb=no_happen_pb)

        return result

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def probability_each(*, object_num, pb_for_all, pb_type='any'):
        """已知一系列事件的总概率（任一发生、全都发生），返回系列中每一个事件发生的概率。

        @param object_num: number of small event in the big event.
        @param pb_for_all: the probability of the big event
        @param pb_type:
            - any small event happen, the big event happen: "any"
            - all small event happen, the big event happen: "all"
        @return:
        """

        if pb_type == 'any':
            pb_for_each = 1 - (1 - pb_for_all) ** (1 / object_num)
        elif pb_type == 'all':
            pb_for_each = pb_for_all ** (1 / object_num)
        else:
            return

        return pb_for_each

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def mutate_one_value_in_map(value, *, map_type='vector', jump_pb=0.1):

        bracket = []

        if map_type == 'vector':
            bracket = [-1, 0, 1]  # value的值域
            if value == 0:
                if random.random() < 0.5:
                    new_value = -1
                else:
                    new_value = 1
            else:
                if random.random() < jump_pb:
                    if value < 0:
                        new_value = 1
                    else:
                        new_value = -1
                else:
                    new_value = 0

        elif map_type == 'cond' or map_type == 'condition':
            bracket = [0, 1]
            if value != 0:
                new_value = 0
            else:
                new_value = 1

        if not bracket:
            print('error 5687')
            return value

        return new_value

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_node_id():
        """获取节点的唯一ID"""

        return 'node_' + str(int(time.time() * 10 ** 6)) + '_' + str(int(random.randint(100000, 999999)))

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_node_name(node_map, node_mother_name):
        """新添加node时，获取node名称（如'ACB')

        @param node_map:
        @param node_mother_name:
        @return: None if the node can't be added.
        """

        # 检查错误
        if not node_mother_name in node_map.keys():
            print('NOTE: node_mother_name not in the node_map given. Node cannot be generated.')
            return  # 母节点名称不在所给字典内

        # 获取“兄弟”节点名称列表
        letter_list = []
        slice_end = len(node_mother_name)
        for k in node_map.keys():
            if k[:slice_end] == node_mother_name:
                letter = k[slice_end:]
                if letter:
                    letter_list.append(letter)

        # 获得命名
        if not letter_list:
            node_letter = NODE_LETTER_LIST[0]  # 母节点还未有任何子节点

        else:
            max_index = 0
            for i in letter_list:
                max_index = max(max_index, NODE_LETTER_LIST.index(i[:1]))

            new_index = max_index + 1
            if new_index >= len(NODE_LETTER_LIST):
                node_letter = None  # “兄弟”数量已太多，超出命名区间
            else:
                node_letter = NODE_LETTER_LIST[new_index]

        return node_mother_name + node_letter

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def exchange_node(d1, d2, d1_node_name, d2_node_name):
        """LV.1 结构交叉函数

        @param d1: node_map1，dict
        @param d2: node_map2, dict
        @param d1_node_name: d1字典中需交叉的节点key（如‘ABA’）
        @param d2_node_name: d2字典中需交叉的节点key
        @return: 交叉后的d1, d2

        NOTE: inplace
        """

        # 获取需剪切的keys

        d1_keys = []
        slice_end1 = len(d1_node_name)
        for k in d1.keys():
            if k[:slice_end1] == d1_node_name:
                d1_keys.append(k)

        d2_keys = []
        slice_end2 = len(d2_node_name)
        for k in d2.keys():
            if k[:slice_end2] == d2_node_name:
                d2_keys.append(k)

        # 移花：把数据单独复制出来，然后清理原dict数据

        tree1 = {}
        for k in d1_keys:
            tree1[k] = d1[k].copy()
            d1.pop(k)

        tree2 = {}
        for k in d2_keys:
            tree2[k] = d2[k].copy()
            d2.pop(k)

        # 定名：确定剪切出来的树叉的新名称

        d1_keys_new = []
        for k in d2_keys:
            new_key = d1_node_name + k[slice_end2:]
            d1_keys_new.append(new_key)

        d2_keys_new = []
        for k in d1_keys:
            new_key = d2_node_name + k[slice_end1:]
            d2_keys_new.append(new_key)

        # 接木：把拷贝出来的数据接回替换点

        for k_old, k_new in zip(d2_keys, d1_keys_new):
            d1[k_new] = tree2[k_old]

        for k_old, k_new in zip(d1_keys, d2_keys_new):
            d2[k_new] = tree1[k_old]

        return d1, d2

    # 分类序列生成函数 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def compare_distance(*sub_edge, feature1, feature2, window=0):
        """特征分类函数，差值对比，绝对距离（比大小：距离为0）  (缺点在于步长个性化太强，需要单独设置参数)

        @param sub_edge: the value which differs the subtract result. (NOTE: negative for feature1 < feature2!)
        @param window: unfunctional here.
        """

        pd.set_option('mode.chained_assignment', None)  # close SettingWithCopyWarning

        feature1.name = 'feature1'
        feature2.name = 'feature2'
        sub_edge = sorted(sub_edge)

        df_result = Tools.sig_merge(feature1, feature2)  # nan filled.
        df_result['difference'] = df_result['feature1'] - df_result['feature2']

        df_result['diff_tag'] = 0
        for num in list(range(len(sub_edge))):
            df_result['diff_tag'][df_result['difference'] >= sub_edge[num]] = num + 1

        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result['diff_tag']

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def compare_sigma(*sigma_edge, feature1, feature2, window=0):
        """特征分类函数，差值对比，平均标准差比例对比（比大小：比例为0）

        @param sigma_edge: the sigma value which differs the subtract result. (NOTE: negative for feature1 < feature2!)
        @param window: unfunctional here.
        """

        pd.set_option('mode.chained_assignment', None)  # close SettingWithCopyWarning

        feature1.name = 'feature1'
        feature2.name = 'feature2'
        sigma_edge = sorted(sigma_edge)

        df_result = Tools.sig_merge(feature1, feature2)  # nan filled.
        df_result['difference'] = df_result['feature1'] - df_result['feature2']

        if window == 0:
            sigma = (feature1.std() + feature2.std()) / 2
            df_result['sigma'] = sigma
        else:
            sigma1 = feature1.rolling(window).std().fillna(method='bfill')
            sigma2 = feature2.rolling(window).std().fillna(method='bfill')
            sigma1.name = 'sigma1'
            sigma2.name = 'sigma2'
            df_sigma = Tools.sig_merge(sigma1, sigma2)  # nan filled.
            df_result['sigma'] = (df_sigma['sigma1'] + df_sigma['sigma2']) / 2

        df_result['sigma_ratio'] = df_result['difference'] / df_result['sigma']

        df_result['diff_tag'] = 0
        for num in list(range(len(sigma_edge))):
            df_result['diff_tag'][df_result['sigma_ratio'] >= sigma_edge[num]] = num + 1

        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def cut_number(*cut_points, feature, window=0):
        """特征分类函数，自切割，常量切割  (缺点在于步长个性化太强，需要单独设置参数)

        @param feature: feature Series (timestamp index) to be cut
        @param cut_points: constant values to cut feature
        @param window: unfunctional.
        @return: cut result: 0, 1, 2 ...
        """

        feature.name = 'data'

        df_result = pd.DataFrame(feature)
        df_result['cut'] = 0

        cut_points = sorted(cut_points)  # NOTE it's sorted!
        for num in list(range(len(cut_points))):
            df_result['cut'][df_result['data'] >= cut_points[num]] = num + 1

        return df_result['cut']

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def cut_distance(*cut_points, feature, window=0):
        """特征分类函数，自切割，数值线性比例切割  (缺点在于不适应长尾，步长设置困难)

        @param feature: feature Series (timestamp index) to be cut
        @param window: rolling window for range reference
        @param cut_points: where of the whole distance(min-max) to cut feature , 0-1
        @return: cut result: 0, 1, 2 ...
        """

        pd.set_option('mode.chained_assignment', None)  # close SettingWithCopyWarning

        feature.name = 'data'
        df_result = pd.DataFrame(feature)
        df_result['cut'] = 0

        num = 0
        cut_points = sorted(cut_points)  # NOTE
        for point in cut_points:

            if window == 0:
                df_result['cut_ref'] = df_result['data'].min() + (
                        df_result['data'].max() - df_result['data'].min()) * point
            else:
                df_result['max_ref'] = df_result['data'].rolling(window).max().fillna(method='bfill')
                df_result['min_ref'] = df_result['data'].rolling(window).min().fillna(method='bfill')
                df_result['cut_ref'] = df_result['min_ref'] + (df_result['max_ref'] - df_result['min_ref']) * point

            df_result['cut'][df_result['data'] > df_result['cut_ref']] = num + 1
            num += 1

        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result['cut']

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def cut_rank(*cut_percents, feature, window=0):
        """特征分类函数，自切割，分布数量比例切割  (缺点在于window长度，太短没有意义，太长初始化太慢，直接使用全部数据会有未来函数问题)

        @param feature: feature Series (timestamp index) to be cut
        @param window: rolling window for range reference
        @param cut_percents: percent in float(0-1) to cut features apart
        @return: cut result: 0, 1, 2 ...
        """

        pd.set_option('mode.chained_assignment', None)  # close SettingWithCopyWarning

        feature.name = 'data'
        df_result = pd.DataFrame(feature)
        df_result['cut'] = 0

        num = 0
        cut_percents = sorted(cut_percents)  # NOTE
        for percent in cut_percents:

            if window == 0:
                df_result['cut_ref'] = df_result['data'].quantile(percent)  # 有利于提升速度（每列都分别rolling一下太慢）
            else:
                df_result['cut_ref'] = df_result['data'].rolling(window).quantile(percent).fillna(method='bfill')

            df_result['cut'][df_result['data'] > df_result['cut_ref']] = num + 1
            num += 1

        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result['cut']

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def cut_sigma(*cut_sigmas, feature, window=0):
        """特征分类函数，自切割，分布标准倍数切割  (缺点在于window长度，太短没有意义，太长初始化太慢，直接使用全部数据会有未来函数问题)

        @param feature: feature Series (timestamp index) to be cut
        @param window: rolling window for range reference
        @param cut_sigmas: float(0-1) of sigma to cut features apart
        @return: cut result: -2, -1, 0, 1, 2, 3...  (NOTE negative results here)

        NOTE: support negative-positive, positive-only and negative-only feature (但是外部的映射字典不要弄错)
        """

        pd.set_option('mode.chained_assignment', None)  # close SettingWithCopyWarning

        feature.name = 'data'
        df_result = pd.DataFrame(feature)
        df_result['cut'] = 0

        num_up = 0
        num_down = 0
        cut_sigmas = sorted(cut_sigmas)  # NOTE
        for sigma in cut_sigmas:

            if window == 0:
                feature_mean = feature.mean()
                feature_edge = feature.std() * sigma
                df_result['cut_ref_up'] = feature_mean + feature_edge
                df_result['cut_ref_down'] = feature_mean - feature_edge  # 可为负

            else:
                df_result['data_mean'] = df_result['data'].rolling(window).mean().fillna(method='bfill')
                df_result['sigma_value'] = df_result['data'].rolling(window).std().fillna(method='bfill') * sigma
                df_result['cut_ref_up'] = df_result['data_mean'] + df_result['sigma_value']
                df_result['cut_ref_down'] = df_result['data_mean'] - df_result['sigma_value']

            df_result['cut'][df_result['data'] > df_result['cut_ref_up']] = num_up + 1
            df_result['cut'][df_result['data'] < df_result['cut_ref_down']] = num_down - 1

            num_up += 1
            num_down -= 1

        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result['cut']

    # 条件函数 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def cond_1(*, cond, pos_should):
        """CONDITION method: 0/1 condition
        ---在满足a(0/1)条件(1)的条件下，使用b的pos_should，其余0 【限2列】

        @param cond: condition to accept pos_should
        @param pos_should: pos_should signals Series (weighted)
        @return: pos_should signal
        """

        pd.set_option('mode.chained_assignment', None)  # close SettingWithCopyWarning

        cond.name = 'condition'
        pos_should.name = 'signal'

        df_sig = pd.concat([cond, pos_should], axis=1, sort=False).fillna(method='ffill')
        df_sig['result_sig'] = df_sig['signal']
        df_sig['result_sig'][df_sig['condition'] <= 0] = 0

        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_sig['result_sig']

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def cond_2(*, cond, pos_should):
        """CONDITION method: -1/0/1 condition
        ---在满足cond方向（正负，对比0）的条件下，pos_should如同方向（正负），使用其值，其余为0 【限2列】

        @param cond: condition to accept pos_should
        @param pos_should: pos_should signals Series (weighted)
        @return: pos_should signal
        """

        df_sig = pd.concat([cond, pos_should], axis=1, sort=False).fillna(method='ffill')

        df_sig['result_ref'] = df_sig.iloc[:, 0] * df_sig.iloc[:, 1]
        df_sig['result_sig'] = df_sig.iloc[:, 1][df_sig['result_ref'] > 0]

        result_sig = df_sig['result_sig'].fillna(0)

        return result_sig

    # 合并 pos_should 信号的相关函数 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def sig_merge(*signals):
        """Merge signals(pos_should) Series into a DataFrame

        @param signals: signals Series
            index: timestamp
            value: pos_should signal
        @return:
        """

        # print(signals)

        list_series = []
        for s in signals:
            list_series.append(s)

        df_sig = pd.concat(list_series, axis=1, sort=False).fillna(method='ffill')

        # dev
        # print('sig_merge()   -----------------------')
        # print(type(df_sig))
        # print(df_sig)

        return df_sig

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def sig_weight(*, signal, weight):
        """Return a weighted pos_should signal Series.

        @param signal: a signal Series
            index: timestamp
            value: pos_should signal
        @param weight: weight of the signal
        @return: weighted pos_should signal Series
        """

        return signal * weight

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def sig_to_one(method, *signals):
        """分流函数

        @param method: method for merging signals into one
        @param signals: pos_should signals Series(weighted)f
        @return: pos_should signal
        """

        if not method in Method.ALL.value:
            print("Can't find method '%s' in constant.Method." % method)
            return pd.DataFrame()

        # debug
        # for i in signals:
        #     print(type(i))
        #     print(i)
        #     print('-' * 10)
        # print('-' * 20)

        if method == 'comb_sum':
            return Tools.comb_sum(*signals)
        elif method == 'comb_vote1':
            return Tools.comb_vote1(*signals)
        elif method == 'comb_vote2':
            return Tools.comb_vote2(*signals)
        elif method == 'comb_vote3':
            return Tools.comb_vote3(*signals)
        elif method == 'comb_min':
            return Tools.comb_min(*signals)

        # elif method == 'cond_2':
        #     return Tools.cond_2(*signals)
        elif method == 'perm_add':
            return Tools.perm_add(*signals)
        elif method == 'perm_sub':
            return Tools.perm_sub(*signals)
        elif method == 'perm_up':
            return Tools.perm_up(*signals)
        elif method == 'perm_down':
            return Tools.perm_down(*signals)

        else:
            print('No method assigned in staticmethod sig_to_one()')
            return pd.DataFrame()

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def comb_sum(*signals):
        """Pos_should signal merge method: comb_sum1  
        ---对各列signal进行加和

        @param signals: pos_should signals Series (weighted)
        @return: pos_should signal
        """

        df_sig = Tools.sig_merge(*signals)

        result_sig = df_sig.sum(axis=1)

        return result_sig

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def comb_vote1(*signals):
        """Pos_should signal merge method: comb_vote1  
        ---使用各列signal投票，加和，输出为：-1/0/1

        @param signals: pos_should signals Series (weighted)
        @return: pos_should signal  -- -1/0/1
        """

        result_ref = Tools.comb_sum(*signals)  # NOTE this depends on comb_sum1()

        df_result = pd.DataFrame(result_ref, columns=['result_ref'])
        df_result['result_sig'] = 0
        pd.set_option('mode.chained_assignment', None)  # close SettingWithCopyWarning 
        df_result['result_sig'][df_result['result_ref'] > 0] = 1
        df_result['result_sig'][df_result['result_ref'] < 0] = -1
        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result['result_sig']

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def comb_vote2(*signals):
        """Pos_should signal merge method: comb_vote2  
        ---使用各列signal投票，须无反对票，输出为：-1/0/1

        @param signals: pos_should signals Series (weighted)
        @return: pos_should signal  -- -1/0/1
        """

        df_sig = Tools.sig_merge(*signals)

        df_result = pd.DataFrame(df_sig.max(axis=1), columns=['sig_max'])
        df_result['sig_min'] = df_sig.min(axis=1)

        df_result['result_sig'] = 0
        pd.set_option('mode.chained_assignment', None)  # close SettingWithCopyWarning
        df_result['result_sig'][df_result['sig_max'] <= 0] = -1
        df_result['result_sig'][df_result['sig_min'] >= 0] = 1
        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result['result_sig']

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def comb_vote3(*signals):
        """Pos_should signal merge method: comb_vote3  
        ---使用各列signal投票，须全票通过，输出为：-1/0/1

        @param signals: pos_should signals Series (weighted)
        @return: pos_should signal  -- -1/0/1
        """

        df_sig = Tools.sig_merge(*signals)

        df_result = pd.DataFrame(df_sig.max(axis=1), columns=['sig_max'])
        df_result['sig_min'] = df_sig.min(axis=1)

        df_result['result_sig'] = 0
        pd.set_option('mode.chained_assignment', None)  # close SettingWithCopyWarning
        df_result['result_sig'][df_result['sig_max'] < 0] = -1
        df_result['result_sig'][df_result['sig_min'] > 0] = 1
        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result['result_sig']

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def comb_min(*signals):
        """Pos_should signal merge method: comb_min
        ---多/空方向：取各列signal中最小/最大的，以做多/空。如sig含有相反符号，则返回0（可用于判断）

        @param signals: pos_should signals Series (weighted)
        @return: pos_should signal
        """

        df_sig = Tools.sig_merge(*signals)

        df_result = pd.DataFrame(df_sig.max(axis=1), columns=['sig_max'])
        df_result['sig_min'] = df_sig.min(axis=1)

        df_result['result_sig'] = 0
        pd.set_option('mode.chained_assignment', None)  # close SettingWithCopyWarning
        df_result['result_sig'][df_result['sig_max'] < 0] = df_result['sig_max'][df_result['sig_max'] < 0]
        df_result['result_sig'][df_result['sig_min'] > 0] = df_result['sig_min'][df_result['sig_min'] > 0]
        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result['result_sig']

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def sig_trend(*signals):
        """Compare each Series line by line

        @param signals: Series
        @return: DataFrame
        """
        df_sig = Tools.sig_merge(*signals)

        column_num = len(df_sig.columns)
        num = 0
        df_trend = pd.DataFrame()  # if len(signals) < 2: return an empty df

        while num < column_num - 1:
            df_trend[num] = df_sig.iloc[:, num + 1] - df_sig.iloc[:, num]
            num += 1

        return df_trend

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def df_to_series(df):
        """Turn a pd.DataFrame into a tuple containing all the Series."""

        list_series = []
        column_num = len(df.columns)
        num = 0

        while num < column_num:
            list_series.append(df.iloc[:, num])
            num += 1

        return tuple(list_series)

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def sig_one_direction(*signals):
        """Check each df line to see if values (in each signal) goes straight or not

        @param signals: Series
        @return: one Series
            -1: signals go straight down
            1: signals go straight up
            0: signals don't go straight down (even/rebound within)
        """

        df_trend = Tools.sig_trend(*signals)
        trend_signals = Tools.df_to_series(df_trend)
        result_ref = Tools.comb_min(*trend_signals)  # 注意这里用拆包语法

        result_ref.name = 'result_ref'
        df_sig = pd.DataFrame(result_ref)
        df_sig['result_sig'] = 0
        df_sig['result_sig'][df_sig['result_ref'] > 0] = 1  # NOTE ">=" can't be set here
        df_sig['result_sig'][df_sig['result_ref'] < 0] = -1

        return df_sig['result_sig']

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def sig_direction(*signals):
        """Check each df line to see if values (in each signal) eventually goes up/down

        @param signals: Series
        @return:
            -1: signals go down eventually but not straight
            1: signals go up eventually but not straight
            0: other situations: even, straight
        """

        # if len(signals) < 3:
        #     return np.nan

        df_sig = Tools.sig_merge(*signals)

        df_sig['sig_start'] = df_sig.iloc[:, 0]
        df_sig['sig_end'] = df_sig.iloc[:, -2]  # -2: cause 'sig_start' became -1 
        df_sig['direction'] = df_sig['sig_end'] - df_sig['sig_start']

        result_ref = Tools.sig_one_direction(*signals)
        df_sig['one_direction'] = result_ref

        df_sig['result_ref'] = df_sig['direction'][df_sig['one_direction'] == 0]
        df_sig['result_sig'] = 0

        result_sig = df_sig['result_sig'].copy()
        result_sig[df_sig['result_ref'] > 0] = 1
        result_sig[df_sig['result_ref'] < 0] = -1

        return result_sig

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def perm_add(*signals):
        """Pos_should signal merge method: permutation addition
        --- 一直涨，sig值越来越大:1，否则0

        @param signals: pos_should signals Series (weighted)
        @return: signal: 0/1
        """

        result_ref = Tools.sig_one_direction(*signals)

        result_ref.name = 'result_ref'
        df_sig = pd.DataFrame(result_ref)
        df_sig['result_sig'] = 0
        df_sig['result_sig'][df_sig['result_ref'] > 0] = 1

        return df_sig['result_sig']

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def perm_sub(*signals):
        """Pos_should signal merge method: permutation subtract
        --- 一直跌，sig值越来越小:1， 否则0

        @param signals: pos_should signals Series (weighted)
        @return: signal: 0/1
        """

        result_ref = Tools.sig_one_direction(*signals)

        result_ref.name = 'result_ref'
        df_sig = pd.DataFrame(result_ref)
        df_sig['result_sig'] = 0
        df_sig['result_sig'][df_sig['result_ref'] < 0] = 1

        return df_sig['result_sig']

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def perm_up(*signals):
        """Pos_should signal merge method: permutation go up
        --- sig值震荡（含持平）上涨：1，否则0

        @param signals: pos_should signals Series (weighted)
        @return: signal: 0/1
        """

        result_ref = Tools.sig_direction(*signals)

        result_ref.name = 'result_ref'
        df_sig = pd.DataFrame(result_ref)
        df_sig['result_sig'] = 0
        df_sig['result_sig'][df_sig['result_ref'] > 0] = 1

        return df_sig['result_sig']

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def perm_down(*signals):
        """Pos_should signal merge method: permutation go up
        --- sig值震荡（含持平）下跌：1，否则0

        @param signals: pos_should signals Series (weighted)
        @return: signal: 0/1
        """

        result_ref = Tools.sig_direction(*signals)

        result_ref.name = 'result_ref'
        df_sig = pd.DataFrame(result_ref)
        df_sig['result_sig'] = 0
        df_sig['result_sig'][df_sig['result_ref'] < 0] = 1

        return df_sig['result_sig']

    # 数据准备相关函数 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_csv_names(file_path, start_date=None, end_date=None, cat='BITMEX.trade'):
        """To get all target file names. Date range support only for bitmex TRADE csv, for now

        @param file_path: path that contains all the csv files(each day, no break)
        @param start_date: str, format: '20150925'
        @param end_date: str, format: '20200101'. Note: end_day will NOT be included.
        @param cat: exchange.csv_type
        @return: file names list
        """

        file_list = sorted(os.listdir(file_path))
        for x in file_list.copy():
            if x[-4:] != '.csv':
                file_list.remove(x)

        if not cat[:6] == 'BITMEX':
            return file_list

        if start_date:
            if pd.to_datetime(file_list[0][6:-4]) < pd.to_datetime(start_date):
                while len(file_list) > 0:
                    if file_list[0][6:-4] == start_date:
                        break
                    file_list.pop(0)
        if end_date:
            if pd.to_datetime(file_list[-1][6:-4]) > pd.to_datetime(end_date):
                while len(file_list) > 0:
                    if file_list[-1][6:-4] == end_date:
                        file_list.pop(-1)  # end date is not included
                        break
                    file_list.pop(-1)

        return file_list

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def bitmex_time_format_change(bitmex_time):
        """Change the unfriendly bitmex time format!

        @param bitmex_time: original timestamp Series from bitmex trade/quote data
        @return: datetime64 timestamp Series.
        """

        bitmex_time = bitmex_time.str[:10] + ' ' + bitmex_time.str[11:]  # change the unfriendly time format!
        timestamp_series = pd.to_datetime(bitmex_time)

        return timestamp_series

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_kbar(df_kbar=pd.DataFrame(), data_path=None, unit='5t'):
        """To get the K price DataFrame from a smaller unit K price csv.

        @param df_kbar: a smaller unit K price DataFrame
        @param data_path: a smaller unit K price csv
        @param unit: pd.groupby() unit
        @return: kbar DataFrame
        """

        if not df_kbar.empty:
            if df_kbar.index.name == 'timestamp':
                df_kbar.reset_index(inplace=True)
        else:
            if data_path:
                df_kbar = pd.read_csv(data_path)
            else:
                print('Either df_kbar or data_path is required.')
                return

        df_kbar['timestamp'] = pd.to_datetime(df_kbar.timestamp)
        df = df_kbar.groupby(pd.Grouper(key='timestamp', freq=unit)).agg({'price_start': 'first',
                                                                          'price_end': 'last',
                                                                          'price_max': 'max',
                                                                          'price_min': 'min',
                                                                          'volume': 'sum',
                                                                          'ticks': 'sum'
                                                                          })

        return df

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_kbar_from_tick(file_path_trade, file_path_quote, unit, to_path=None, to_name=None, start_date=None,
                           end_date=None, exchange='BITMEX', symbol='XBTUSD'):
        """To get the K price DataFrame from tick price. Bitmex only. NO more than 1 day

        @param file_path_trade: path that contains the trade csv files(fromt bitmex)
        @param file_path_quote: path that contains the quote csv files(fromt bitmex)
        @param unit: compress unit: example: '30s', 't', '15t', 'h', 'd'. No more than 1 Day.
        @param to_path: path to save kbar csv file
        @param to_name: csv file name
        @param start_date: str, format: '20150925'
        @param end_date: str, format: '20200101'. Note: end_day will NOT be included.
        @param exchange: exchange
        @param symbol: symbol
        @return: compressed tick price DataFrame with columns:
            timestamp  --note that it's the period start timestamp
            price_start
            price_end
            price_max
            price_min
        """

        if not exchange == 'BITMEX':
            return

        t0 = time.time()

        # Two process for two data source
        df_kbar, df_ticks = Tools.agg_cal(Tools.compress_distribute,
                                          ('trade', file_path_trade, unit, start_date, end_date, symbol),
                                          ('quote', file_path_quote, unit, start_date, end_date, symbol),
                                          process_num=2)  # 我承认这种传参方式很奇葩。。。

        df_kbar = pd.concat([df_kbar, df_ticks], axis=1, sort=False)
        df_kbar.rename({'symbol': 'ticks'}, axis=1, inplace=True)

        print('%.3f | kbar generated successfully.' % (time.time() - t0))

        if to_path and to_name:
            df_kbar.to_csv('%s/%s.csv' % (to_path, to_name))
            print('%.3f | "%s" saved successfully to "%s".' % ((time.time() - t0), to_name, to_path))

        return df_kbar

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def compress_trade(file_path_trade, unit, start_date, end_date, symbol):
        """trade csv dealing: get basic price info"""

        list_kbar = []
        t0 = time.time()

        file_list_trade = Tools.get_csv_names(file_path_trade, start_date=start_date, end_date=end_date,
                                              cat='BITMEX.trade')
        for file in file_list_trade:

            df_new = pd.read_csv('%s/%s' % (file_path_trade, file),
                                 usecols=['timestamp', 'symbol', 'side', 'size', 'price'])
            if symbol not in list(df_new['symbol']):
                continue
            df_new = df_new[df_new.symbol == symbol]
            df_new.reset_index(drop=True, inplace=True)

            df_new['timestamp'] = Tools.bitmex_time_format_change(df_new['timestamp'])
            group_price = df_new.groupby(pd.Grouper(key='timestamp', freq=('%s' % unit))).price
            price_start = group_price.first()
            price_end = group_price.last()
            price_max = group_price.max()
            price_min = group_price.min()
            volume = df_new.groupby(pd.Grouper(key='timestamp', freq=('%s' % unit)))['size'].sum()

            df_g = pd.DataFrame(price_start)
            df_g.rename({'price': 'price_start'}, axis=1, inplace=True)
            df_g['price_end'] = price_end
            df_g['price_max'] = price_max
            df_g['price_min'] = price_min
            df_g['volume'] = volume

            list_kbar.append(df_g)

            print('%.3f | "%s" included.' % ((time.time() - t0), file))

        df_kbar = pd.concat(list_kbar, sort=False, ignore_index=False)

        return df_kbar

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def compress_quote(file_path_quote, unit, start_date, end_date, symbol):
        """quote csv dealing: get ticks count for each line(for flipage estimation)"""

        list_ticks = []
        t0 = time.time()

        file_list_quote = Tools.get_csv_names(file_path_quote, start_date=start_date, end_date=end_date,
                                              cat='BITMEX.quote')
        for file in file_list_quote:

            df_new = pd.read_csv('%s/%s' % (file_path_quote, file),
                                 usecols=['timestamp', 'symbol'])
            if symbol not in list(df_new['symbol']):
                continue
            df_new = df_new[df_new.symbol == symbol]
            df_new.reset_index(drop=True, inplace=True)

            df_new['timestamp'] = Tools.bitmex_time_format_change(df_new['timestamp'])
            ticks = df_new.groupby(pd.Grouper(key='timestamp', freq=('%s' % unit))).symbol.count()
            df_ticks = pd.DataFrame(ticks)
            list_ticks.append(df_ticks)

            print('%.3f | "%s" ticks counted.' % ((time.time() - t0), file))

        df_ticks = pd.concat(list_ticks, sort=False, ignore_index=False)

        return df_ticks

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def compress_distribute(direct_to, file_path, unit, start_date, end_date, symbol):
        result = None
        if direct_to == 'trade':
            result = Tools.compress_trade(file_path, unit, start_date, end_date, symbol)
        elif direct_to == 'quote':
            result = Tools.compress_quote(file_path, unit, start_date, end_date, symbol)

        return result

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def tick_comp_price(file_path, start_date=None, end_date=None):
        """Get the compressed traded tick price from file_path.

        @param file_path: path that contains all the csv files(each day, no break) with columns:
            timestamp
            price
        @param start_date: str, format: '20150925'
        @param end_date: str, format: '20200101'. Note: end_day will NOT be included.
        @param exchange: support only bitmex, for now
        @return: compressed tick price DataFrame with columns:
            timestamp
            price
        """

        # get all target file names
        file_list = Tools.get_csv_names(file_path, start_date=start_date, end_date=end_date)

        list_df = []
        for file in file_list:
            df_add = pd.read_csv('%s/%s' % (file_path, file))
            list_df.append(df_add)

        df_tick_comp_price = pd.concat(list_df, ignore_index=True, sort=False)

        return df_tick_comp_price

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_tick_comp_price(file_path, to_path=None, start_date=None, end_date=None, exchange='BITMEX',
                            symbol='XBTUSD'):
        """To get the compressed traded tick price from raw trade data in file_path.

        @param file_path: path that contains all the raw trade csv files(support bitmex only, for now).
        @param to_path: path to save csv file. None for not saving.
        @param start_date: str, format: '20150925'
        @param end_date: str, format: '20200101'. Note: end_day will NOT be included.
        @param exchange: support bitmex only, for now.
        @param symbol: trading target
        @return: compressed tick price DataFrame with columns:
            timestamp
            price
        """

        if not exchange == 'BITMEX':
            return

        # close SettingWithCopyWarning
        pd.set_option('mode.chained_assignment', None)

        t0 = time.time()

        list_tick_comp_price = []

        # get all target file names
        file_list = Tools.get_csv_names(file_path, start_date=start_date, end_date=end_date)

        # read and compress all the files in file_path
        for file in file_list:

            df = pd.read_csv('%s/%s' % (file_path, file), usecols=['timestamp', 'symbol', 'price', 'tickDirection'])
            if symbol not in list(df['symbol']):
                continue

            df = df[df.symbol == symbol]
            df.reset_index(inplace=True, drop=True)
            df['timestamp'] = Tools.bitmex_time_format_change(df['timestamp'])
            df = df[
                (df.tickDirection == 'PlusTick') | (
                        df.tickDirection == 'MinusTick')]  # keep only lines that price changed

            # 仅保留同方向连续吃两笔以上的tick --因为后续 limit order 成交的判断依据是：越过limit价格
            df['tickDirection_old'] = np.append(np.nan, df['tickDirection'][:-1])
            df['keep'] = 1
            df['keep'][df.tickDirection != df.tickDirection_old] = 0
            df.iloc[0, -1] = 1  # keep first line
            df = df[df['keep'] == 1]

            df.drop(['symbol', 'tickDirection', 'tickDirection_old', 'keep'], axis=1, inplace=True)

            list_tick_comp_price.append(df)
            print('%.3f | file "%s" included.' % ((time.time() - t0), file))

            # save csv files to to_path(folder)
            if to_path:
                df.to_csv(('%s/%s' % (to_path, file)), index=False)  # the same file name.
                print('%.3f | file "%s" saved to "%s".' % ((time.time() - t0), file, to_path))

        df_tick_comp_price = pd.concat(list_tick_comp_price)
        df_tick_comp_price.reset_index(drop=True, inplace=True)

        print('%.3f | trade tick price compressed successfully.' % (time.time() - t0))

        # reopen SettingWithCopyWarning
        pd.set_option('mode.chained_assignment', 'warn')

        return df_tick_comp_price

    # 回测过程相关函数 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def arr_orders_prepare(df_book, adjust=None):
        """
        adjust order_price to: 'auto', 'market', None for not adjusting.
        """

        # close SettingWithCopyWarning
        pd.set_option('mode.chained_assignment', None)

        df_book['timestamp'] = pd.to_datetime(df_book.timestamp)

        pos_should_old = np.append(np.array([POS_START]), df_book['pos_should'])
        df_book['pos_should_old'] = pos_should_old[:-1]
        df_book['trade_sig'] = df_book.pos_should - df_book.pos_should_old

        df_book['order_side'] = Direction.NONE
        df_book['order_side'][df_book.trade_sig > 0] = Direction.LONG  # or 1
        df_book['order_side'][df_book.trade_sig < 0] = Direction.SHORT  # or -1

        df_book['order_value'] = df_book.trade_sig.abs()

        df_book['pos_regress'] = 0  # False
        df_book['pos_regress'][df_book.pos_should.abs() < df_book.pos_should_old.abs()] = 1  # True

        df_book = df_book[df_book.order_side != Direction.NONE]
        df_book.reset_index(drop=True, inplace=True)

        # for LIMIT orders
        if not 'order_price' in df_book.columns:
            df_book['order_price'] = MARKET_PRICE  # choice: MARKET / end of period price. here the former.
        else:
            if adjust is None:
                pass
            elif adjust == 'market':
                df_book['order_price'] = MARKET_PRICE  # force to MARKET order
            elif adjust == 'auto':  # 'auto' --用pos_regress区分轻重缓急 
                df_book['order_price'][(df_book.pos_regress == 0) & (df_book.order_side == 1)] = \
                    df_book['price'][(df_book.pos_regress == 0) & (df_book.order_side == 1)] - LIMIT_DISTANCE
                df_book['order_price'][(df_book.pos_regress == 0) & (df_book.order_side == 0)] = \
                    df_book['price'][(df_book.pos_regress == 0) & (df_book.order_side == 0)] - LIMIT_DISTANCE

        arr_orders = np.array([
            df_book.timestamp.values,  # NOTE: auto change to int timestamp!
            df_book.order_side.values,
            df_book.order_price.values,
            df_book.order_value.values
        ], )

        # reopen SettingWithCopyWarning
        pd.set_option('mode.chained_assignment', 'warn')

        return arr_orders

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def arr_price_prepare(df_price):
        """Convert kbar pd.DataFrame to np.Array.

        @param df_price: kbar DataFrame, includes:
            timestamp  -- start time of the period. NOTE this is different than timestamp in df_book!
            price_start
            price_end
            price_max
            price_end
            ticks  --how many L1 changes the period has.
        @return: np.Array: [line_data, line_index], NOTE that timestamp is converted to int.
        """

        df_price['timestamp'] = pd.to_datetime(df_price.timestamp)

        arr_price = np.array([
            df_price.timestamp.values,  # NOTE: auto change to int timestamp!
            df_price.price_start.values,
            df_price.price_end.values,
            df_price.price_max.values,
            df_price.price_min.values,
            df_price.ticks.values
        ])

        return arr_price

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_bar_data(arr_price, num):
        """To get No.num kbar data in arr_price

        @param arr_price: fromt arr_price_prepare()
        @param num: index
        @return: each kbar data
        """
        line = arr_price[..., num]

        timestamp = line[0]
        price_start = line[1]
        price_end = line[2]
        price_max = line[3]
        price_min = line[4]
        ticks = line[5]

        return timestamp, price_start, price_end, price_max, price_min, ticks

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def trade_kbar(price_start, price_end, price_max, price_min,
                   order_side=None, order_price=None, ticks=None, first_time=True):
        """To judge if/how an order were traded, kbar method.

        @param price_*: price, number.
        @param order_side:
        @param order_price: None for MARKET order.
        @param ticks: None for no data(but better to have, to estimate slippage)
        @param first_time: for LIMIT order slippage and fee_rate estimation
            -- True if the order is made recently and has not yet been judged before
            -- False if the order has been judged before.
        @return: turple
            -- not traded: (np.nan, np.nan)
            -- traded: (trade_price, fee_rate)

        NOTE
            - 在调用本函数之前，另行判断是否overload --无法进入成交判断，因为没有能成功提交订单到交易所
        """

        if order_side == 0 or order_side == Direction.SHORT:
            trade_price, fee_rate = Tools.short_judge_kbar(price_start, price_end, price_max, price_min,
                                                           order_price=order_price, ticks=ticks, first_time=first_time)
        elif order_side == 1 or order_side == Direction.LONG:
            trade_price, fee_rate = Tools.long_judge_kbar(price_start, price_end, price_max, price_min,
                                                          order_price=order_price, ticks=ticks, first_time=first_time)
        else:
            trade_price, fee_rate = np.nan, np.nan

        return trade_price, fee_rate

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def short_judge_kbar(price_start, price_end, price_max, price_min,
                         order_price=None, ticks=None, first_time=True):
        """Short order judgement, kbar method."""

        bar_direction = price_end - price_start
        bar_range = price_max - price_min

        if np.isnan(bar_direction):
            return np.nan, np.nan  # no trades happened.

        # 1. MARKET / STOP / LIQUIDATION order  --first_time=True

        if not order_price:

            slippage = Tools.get_slippage(price_start, price_min, ticks, how='bad')
            if slippage < MIN_DISTANCE and np.random.random() < 0.5:
                slippage += MIN_DISTANCE  # for price_start can be bid_price_1 or ask_price_1, not sure.

            trade_price = price_start - slippage

            return trade_price, OrderFee.TAKER.value

        # 2. LIMIT order  --there are totally 10+1 possible path out of 5 parameter(32 %path) for new order

        # %path 16, 32
        if order_price >= price_max:
            return np.nan, np.nan  # order_price set too high, can't be traded at this bar

        if not first_time:
            return order_price, OrderFee.MAKER.value  # old LIMIT order (already there)

        # params
        maker_distance = order_price - price_start  # order can be a MAKER if distance is large enough.
        lowest_possible_trade_price = max(order_price, price_min)  # 可达(price_max+MIN_DISTANCE)，但会干扰比较计算的正负

        if order_price < price_end:
            # %path 1, 2, 18, 19, 22  --must get traded zone
            fee_rate = OrderFee.TAKER.value

            slippage = Tools.get_slippage(price_end, lowest_possible_trade_price, ticks, how='good')
            if slippage > 0:
                # slippage > 0 indicates an intense moment. as now the bar ends high, there is a chance to trade higher
                # than lowest_possible_trade_price.
                if bar_direction > 0:
                    p = 0.4  # short order against the trend
                else:
                    p = 0.2
                if np.random.random() < p:
                    lowest_possible_trade_price += slippage

            # %path 22  --higher than price_start, a chance to be MAKER.
            if maker_distance > 0:
                slippage_opp = Tools.get_slippage(price_end, price_start, ticks, how='bad')
                # not a standard usage for slippage  --assume a conservative opponent order slippage, then compare.
                if slippage_opp < maker_distance:
                    fee_rate = OrderFee.MAKER.value

            return lowest_possible_trade_price, fee_rate

        else:
            # %path 0, 4
            if maker_distance == 0:
                # order_price == price_start  -- %path 0 (this is not shown in the 32 possibility table)
                fee_rate = Tools.random_fee_rate(0.5)
                return lowest_possible_trade_price, fee_rate

            elif maker_distance < 0:
                # %path 4
                fee_rate = OrderFee.TAKER.value
                slippage_market = Tools.get_slippage(price_start, price_min, ticks, how='bad')
                market_price = price_start - slippage_market  # assume a market short order

                if lowest_possible_trade_price <= market_price:
                    return market_price, fee_rate
                else:
                    up_pin_percentage = (price_max - price_start) / bar_range
                    # in this situation, real up part is (price_max-lowest_possible_trade_price), but a bit too radical
                    if np.random.random() < up_pin_percentage:
                        return price_start, fee_rate  # if many trades happens upper, then still a big chance get traded
                    else:
                        return np.nan, np.nan

            # %path 8, 24
            slippage_edge = Tools.get_slippage(price_max, price_start, ticks, how='good')
            trade_edge = price_max - slippage_edge
            if lowest_possible_trade_price < trade_edge:
                return lowest_possible_trade_price, OrderFee.MAKER.value
            else:
                return np.nan, np.nan

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def long_judge_kbar(price_start, price_end, price_max, price_min,
                        order_price=None, ticks=None, first_time=True):
        """Long order judgement, kbar method. Check short_judge_kbar() for more notes."""

        bar_direction = price_end - price_start
        bar_range = price_max - price_min

        if np.isnan(bar_direction):
            return np.nan, np.nan

        # 1. MARKET / STOP / LIQUIDATION order

        if not order_price:

            slippage = Tools.get_slippage(price_max, price_start, ticks, how='bad')
            if slippage < MIN_DISTANCE and np.random.random() < 0.5:
                slippage += MIN_DISTANCE

            trade_price = price_start + slippage

            return trade_price, OrderFee.TAKER.value

        # 2. LIMIT order  --there are totally 10+1 possible path out of 5 parameter(32 %path) for new order

        # %path 1, 17
        if order_price <= price_min:
            return np.nan, np.nan

        if not first_time:
            return order_price, OrderFee.MAKER.value

        # params
        maker_distance = price_start - order_price
        highest_possible_trade_price = min(order_price, price_max)

        if order_price > price_end:
            # %path 4, 8, 16, 24, 32  --must get traded zone
            fee_rate = OrderFee.TAKER.value

            slippage = Tools.get_slippage(highest_possible_trade_price, price_end, ticks, how='good')
            if slippage > 0:
                if bar_direction < 0:
                    p = 0.4
                else:
                    p = 0.2
                if np.random.random() < p:
                    highest_possible_trade_price -= slippage

            # %path 4  --lower than price_start, a chance to be MAKER
            if maker_distance > 0:
                slippage_opp = Tools.get_slippage(price_start, price_end, ticks, how='bad')
                if slippage_opp < maker_distance:
                    fee_rate = OrderFee.MAKER.value

            return highest_possible_trade_price, fee_rate

        else:
            # %path 0, 22
            if maker_distance == 0:
                fee_rate = Tools.random_fee_rate(0.5)
                return highest_possible_trade_price, fee_rate

            elif maker_distance < 0:
                fee_rate = OrderFee.TAKER.value
                slippage_market = Tools.get_slippage(price_max, price_start, ticks, how='bad')
                market_price = price_start + slippage_market

                if highest_possible_trade_price >= market_price:
                    return market_price, fee_rate
                else:
                    down_pin_percentage = (price_start - price_min) / bar_range
                    if np.random.random() < down_pin_percentage:
                        return price_start, fee_rate
                    else:
                        return np.nan, np.nan

            # %path 2, 18
            slippage_edge = Tools.get_slippage(price_start, price_min, ticks, how='bad')
            trade_edge = price_min + slippage_edge
            if highest_possible_trade_price > trade_edge:
                return highest_possible_trade_price, OrderFee.MAKER.value
            else:
                return np.nan, np.nan

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_slippage(price_high, price_low, ticks=None, how='bad',
                     min_range=MIN_DISTANCE,
                     slippage_tick_min=SLIPPAGE_TICK_MIN,
                     slippage_tick_max=SLIPPAGE_TICK_MAX
                     ):
        """Astimate slippage for kbar judgement method, by price difference and ticks amount at the moment.

        @param price_high: price number
        @param price_low: price number
        @param ticks: None for no data
        @param how: 'good' or 'bad
        """

        slip_range = abs(price_high - price_low)
        slippage = 0

        if ticks is not None:

            if how == 'bad':
                if ticks < slippage_tick_min:
                    slippage = 0
                elif ticks < slippage_tick_max:
                    # Linear estimation, to enlarge the loss:
                    slippage = slip_range * (ticks - slippage_tick_min) / (slippage_tick_max - slippage_tick_min)
                    slippage -= min_range  # compensate for ticks == slippage_tick_min
                else:
                    # assume that it can still be traded, but at a worst price.
                    slippage = slip_range

            elif how == 'good':
                if ticks < slippage_tick_max:
                    slippage = 0
                else:
                    slippage = Tools.fit_to_minimal(np.random.random() * np.random.random() * slip_range)

        else:

            if how == 'bad':
                slippage = slip_range / 2
            elif how == 'good':
                slippage = 0

        slippage = abs(Tools.fit_to_minimal(slippage, min_range=min_range))

        return slippage

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def fit_to_minimal(float_number, min_range=MIN_DISTANCE, simplify=True):
        """To fit a price / price_delta / anydata to (exchange's) minimal distance."""

        result = round(float_number * (1 / min_range)) / (1 / min_range)

        if simplify:
            float_num = len(str(min_range).split('.')[1])
            result = Tools.keep_float(result, float_num)

        return result

        # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def random_fee_rate(p=0.5):
        """To return a random fee rate

        @param p: probability of being a market MAKER
        @return: fee rate value
        """
        if np.random.random() < p:
            fee_rate = OrderFee.MAKER.value
        else:
            fee_rate = OrderFee.TAKER.value

        return fee_rate

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_score(trading_record, df_kbar, capital=CAPITAL, annual_period=ANNUAL_PERIOD, save_file=None):
        """Get performance from trading records and prices.

        @param trading_record: Dict of traded result(and no other like cancelled), must include these keys:
            timestamp  --of the event
            side  --order direction
            price  --traded price in fiat
            order_value  --fiat value, volume in fiat, to detect if traded in this line.
            fee_rate  --in float

            new:
            order_price

        @param df_kbar: 1 minute / 1 hour close price df, includes:
            timestamp  --period time start
            price  --period end price

        @param annual_period: How many periods you want to cut one year into.
            Max: 365*24*60
            hour: 365*24
            day: 365
            week:  52
            month: 12
        @param save_file: to store results on disk.  --NOTE: file's max time period is equal to df_kbar!
        @return: annual score, python dict

        """

        pd.set_option('mode.chained_assignment', None)  # 关闭 SettingWithCopyWarning 警告

        # 1. 生成交易记录df_traded

        df_traded = pd.DataFrame(trading_record)
        if df_traded.shape[0] == 0:
            print('No trading orders recorded.')
            return Tools.bad_score()
        df_traded['timestamp'] = pd.to_datetime(df_traded.timestamp)

        # 2.合成运算所需df

        if not 'price' in df_kbar.columns:
            df_kbar['price'] = df_kbar['price_end']
        for i in df_kbar.columns:
            if i == 'price' or i == 'timestamp':
                continue
            df_kbar.drop(i, axis=1, inplace=True)

        df_kbar['timestamp'] = pd.to_datetime(df_kbar.timestamp)
        timedelta = df_kbar.loc[1, 'timestamp'] - df_kbar.loc[0, 'timestamp']
        # 让groupby的last()方法找到对应到price_end：
        df_kbar['timestamp'] += timedelta - pd.to_timedelta(1, unit='ns')

        start_time = df_traded.timestamp.min() - timedelta
        end_time = df_traded.timestamp.max() + timedelta
        df_kbar_pare = df_kbar[(df_kbar.timestamp > start_time) & (df_kbar.timestamp < end_time)]

        df = pd.concat([df_traded, df_kbar_pare], sort=False, ignore_index=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        # df.pos.fillna(method='ffill', inplace=True)
        # df.avg_price.fillna(method='ffill', inplace=True)
        # df.traded_vol_all.fillna(method='ffill', inplace=True)

        # 3. 更换计算单位

        df['re_direction'] = np.nan
        df['re_direction'][(df.side == Direction.SHORT) | (df.side == -1)] = 1
        df['re_direction'][(df.side == Direction.LONG) | (df.side == 1)] = -1
        df['re_price'] = 1 / df.price
        df['re_size'] = df.order_value * df.re_price

        # 4. 遍历

        re_pos = 0
        re_pos_old = 0
        re_avg_price = 0
        re_avg_price_old = 0
        list_re_pos = []  # 时刻持仓
        list_re_avg_price = []  # 时刻均价
        close_profit_list = []  # 时刻平仓获利

        arr_temp = np.array([
            df['re_direction'],
            df['re_price'],
            df['order_value']  # 注意这里是用order_value
        ])

        num = 0
        while num < arr_temp.shape[1]:

            data = arr_temp[..., num]
            re_direction = data[0]
            price = data[1]
            volume = data[2]

            re_avg_price_old = re_avg_price  # 注意要写在循环开头（右侧更新前）
            re_pos_old = re_pos

            # 计算均价、持仓 (成熟的算法)
            if re_direction == 1:
                re_pos += abs(volume)
                if re_pos > 0:
                    if re_pos > volume:
                        # 当前订单同向影响了均价
                        re_avg_price = ((re_pos - abs(volume)) * re_avg_price
                                        + volume * price) / re_pos
                    else:
                        # 当前订单导致仓位发生正负变化(或变为非0)
                        re_avg_price = price
                elif re_pos == 0:
                    # 当前订单导致仓位归零
                    re_avg_price = 0
                else:
                    # 当前订单给空单减仓，均价未变化
                    pass
            elif re_direction == -1:
                re_pos -= abs(volume)
                if re_pos < 0:
                    if abs(re_pos) > abs(volume):
                        re_avg_price = ((abs(re_pos) - abs(volume)) * re_avg_price
                                        + abs(volume) * price) / abs(re_pos)
                    else:
                        re_avg_price = price
                elif re_pos == 0:
                    re_avg_price = 0
                else:
                    pass

            # 平仓利润计算
            if re_pos * re_pos_old <= 0:
                if re_pos_old == 0:
                    # 此时是全新开仓
                    close_profit = 0
                else:
                    # 此时是平掉了一个方向的全部仓位
                    close_profit = re_pos_old * (price - re_avg_price_old)  # 这里有修改
            else:
                if abs(re_pos) < abs(re_pos_old):
                    # 此时是部分平仓
                    if re_pos > 0:
                        close_profit = - (re_pos - re_pos_old) * (price - re_avg_price_old)  # 这里的符号曾经错了
                    else:
                        close_profit = - abs(re_pos - re_pos_old) * (price - re_avg_price_old)
                else:
                    # 此时是单向加仓
                    close_profit = 0

            list_re_pos.append(re_pos)
            list_re_avg_price.append(re_avg_price)
            close_profit_list.append(close_profit)

            num += 1

        # 5. 向量化运算赋值

        df['re_pos'] = list_re_pos
        df['re_avg_price'] = list_re_avg_price
        df['re_avg_price'][df['re_avg_price'] == 0] = np.nan

        df['re_hold_return'] = df['re_pos'] * (df['re_price'] - df['re_avg_price'])
        df['re_hold_return'].fillna(0, inplace=True)
        df['holdings_value'] = df['re_hold_return'] * df['price']

        df['re_close'] = close_profit_list
        df['close_accumulate'] = df['re_close'].cumsum()

        df['re_fee'] = df['re_size'] * df['fee_rate']
        df['re_fee'].fillna(0, inplace=True)
        df['re_profit'] = df['re_close'] + df['re_fee']

        df['fee'] = df['re_fee'] * df['price']  # 这里对上面fee的单位做了转换。假设套保。
        df['fee_acc'] = df['fee'].cumsum()
        df['close_profit'] = df['re_close'] * df['price']  # 这里假设对每一笔交易的平仓后利润进行套保
        df['profit_acc'] = df['close_profit'].cumsum()

        df['re_returns'] = df['re_close'] + df['re_fee'] + df['re_hold_return']
        df['returns'] = df['close_profit'] + df['fee'] + df['holdings_value']

        df['re_returns_acc'] = df['re_close'].cumsum() + df['re_fee'].cumsum() + df['re_hold_return']
        df['returns_acc'] = df['profit_acc'] + df['fee_acc'] + df['holdings_value']

        # 6. 结果评估

        if df['returns_acc'].min() < - capital:
            return Tools.bad_score()  # lost all.

        # 合成基于固定时间单位的表格
        if annual_period == 365 * 24 * 60:
            freq = 't'
        elif annual_period == 365 * 24:
            freq = 'h'
        elif annual_period == 365:
            freq = 'd'
        else:
            minutes = 365 * 24 * 60 / annual_period  # valid for time period within one day
            freq = '%dt' % int(minutes)

        period_balance = df.groupby(pd.Grouper(key='timestamp', freq=freq)).returns_acc.last() + capital
        period_returns_ratio = period_balance.pct_change()  # key input.
        holds_yes = df.groupby(pd.Grouper(key='timestamp', freq=freq)).re_avg_price.sum()
        period_num = period_returns_ratio.shape[0] - 1
        annual_orders = df_traded.shape[0] * annual_period / period_num
        annual_volumes = df_traded.order_value.sum() * annual_period / period_num
        hold_time_ratio = holds_yes[holds_yes > 0].count() / period_num
        period_profit = period_returns_ratio[period_returns_ratio > 0].count()
        period_loss = period_returns_ratio[period_returns_ratio < 0].count()
        profit_time_ratio = period_profit / (period_profit + period_loss)

        dict_score = {'Sortino_ratio': empyrical.sortino_ratio(period_returns_ratio, annualization=annual_period),
                      'Calmar_ratio': empyrical.calmar_ratio(period_returns_ratio, annualization=annual_period),
                      'Annual_return': empyrical.annual_return(period_returns_ratio, annualization=annual_period),
                      'Max_drawdown': empyrical.max_drawdown(period_returns_ratio),
                      'Sharpe_ratio': empyrical.sharpe_ratio(period_returns_ratio, annualization=annual_period),
                      '年化交易次数': annual_orders,
                      '年化成交量': annual_volumes,
                      '持仓时间比': hold_time_ratio,
                      '浮盈时间比(如持仓)': profit_time_ratio
                      }

        # 7. save
        if save_file:
            df.to_csv('%s' % save_file, index=False)

        pd.set_option('mode.chained_assignment', 'warn')  # 重新打开警告

        return dict_score

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def bad_score():
        """Returns a very bad score, for:

        1. No trading record
        2. Results be so bad as to cause calculation issue.
        """

        dict_score = {'Sortino_ratio': -888,
                      'Calmar_ratio': -888,
                      'Annual_return': -888,
                      'Max_drawdown': -888,
                      'Sharpe_ratio': -888,
                      '年化交易次数': 0,
                      '年化成交量': 0,
                      '持仓时间比': 0,
                      '浮盈时间比(如持仓)': 0
                      }

        return dict_score


if __name__ == '__main__':
    # 准备虚拟的交易记录字典
    df1 = pd.read_csv('data/trading_record_example.csv')
    trading_record = {
        'timestamp': list(df1.timestamp.values),
        'side': list(df1.side.values),
        'price': list(df1.price.values),
        'order_value': list(df1.order_value.values),
        'fee_rate': list(df1.fee_rate.values),
        'pos': list(df1.pos.values),
        'avg_price': list(df1.avg_price.values),
        'traded_vol_all': list(df1.traded_vol_all)
    }

    # 准备历史记录df
    df_price = pd.read_csv('data/bitmex_price_1hour_2020q1.csv', usecols=['timestamp', 'price_end'])
    df_price.rename({'price_end': 'price'}, axis=1, inplace=True)

    # TEST IT

    t0 = time.time()

    score = Tools.get_score(trading_record,
                            df_price,
                            capital=1000,
                            annual_period=(365 * 24),
                            # save_file='trading_record_test(after).csv'
                            )
    print(score)

    print('time: %s' % (time.time() - t0))
