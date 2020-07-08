import os
import time
import datetime
import pandas as pd
import numpy as np
import random

import empyrical

import multiprocessing as mp

from constant import *


class Tools(object):

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def test(x='example_arg'):
        print('This is a test static method in Tools with argument: %s' % x)

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
    def str_time(keep_float=6):
        """获取str时间信息

        @param keep_float: 秒后要保留的小数位
        @return: str时间
        """

        if keep_float == 0: keep_float = -1
        keep_float -= 6
        if keep_float > 0: keep_float = 0

        if keep_float == 0:
            now = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        else:
            now = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))[:keep_float]

        return now

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def shorten_float(number, keep_float):
        """float简化：保留小数点后x位
        @param number: 要化简的数
        @param keep_float: 要保留的小数位
        @return: 化简后的数
        """

        number = int(number * 10 ** keep_float)
        number = number / 10 ** keep_float

        return number

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def fit_to_minimal(value, min_range=MIN_DISTANCE, modify=True):
        """To fit a price / price_delta / anydata to the (exchange's) minimal distance."""

        result = round(value * (1 / min_range)) / (1 / min_range)

        if modify:

            if isinstance(min_range, float):
                # 避免小数点后太长
                float_num = len(str(min_range).split('.')[1])
                result = Tools.shorten_float(result, float_num)

            else:
                # 整数进，整数出
                result = int(round(result))

        return result

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def cal_event_pb(*args, deep=False):
        """根据传入的独立概率事件计算发生概率

        @param args: probability(ies) of happening(s)
        @param deep: if deep=True, returns the probability of each 0~len(args) event happens
        @return: probability dict
        """

        if len(args) == 0:
            result = dict(all_happen_pb=1,  # 为了下面计算[]返回正确，这里将概率都设定为1
                          any_happen_pb=1,
                          no_happen_pb=1)
            return result

        all_happen_pb = 1
        no_happen_pb = 1

        for pb in args:
            all_happen_pb = all_happen_pb * pb
            no_happen_pb = no_happen_pb * (1 - pb)

        any_happen_pb = 1 - no_happen_pb

        result = dict(all_happen_pb=all_happen_pb,  # 全部发生的概率
                      any_happen_pb=any_happen_pb,  # 至少一个发生的概率
                      no_happen_pb=no_happen_pb  # 没有一个发生的概率
                      )

        if deep:
            # 计算0～n个事件发生的概率
            count_dict = {}
            num = 0
            while num <= len(args):  # NOTE '=' here.

                happen_list = []
                chose_list, left_list = Tools.cut_list_into_combination(num, args)

                for chose, left in zip(chose_list, left_list):
                    chose_happen = Tools.cal_event_pb(*chose)['all_happen_pb']
                    left_not_happen = Tools.cal_event_pb(*left)['no_happen_pb']
                    happen_list.append(chose_happen * left_not_happen)

                count_dict['%s event happen' % num] = sum(happen_list)
                num += 1

            result.update(count_dict)

        return result

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def cut_list_into_combination(combine_num, target_list):
        """将列表的n个元素组合，返回组合结果、剩余结果

        @param combine_num: how many value in target_list to get out
        @return: [[out_list1, out_list2, ...], [left_list1, left_list2, ...]]

        NOTE: test it to see: Tools.cut_list_into_combination(2, [0, 1, 2, 3, 4])
        """

        if combine_num <= 0:
            return [[]], [target_list]

        if combine_num > len(target_list):
            combine_num = len(target_list)

        # these lists consists of lists
        left_list = []
        chose_list = []
        right_list = []

        for i in list(range(len(target_list))):

            if i + combine_num > len(target_list):
                break  # 要为后续对right的取出，留下足够的元素

            left = target_list[:i]
            chose_value = target_list[i]
            right = target_list[i + 1:]

            left_list.append(left)
            chose_list.append([chose_value])
            right_list.append(right)

        if combine_num == 1:

            element_chose_list = chose_list
            element_left_list = []

            for left, right in zip(left_list, right_list):
                element_left = left + right  # list
                element_left_list.append(element_left)

        else:

            combine_num_next = combine_num - 1
            element_chose_list = []
            element_left_list = []

            for left, chose, right in zip(left_list, chose_list, right_list):

                element_chose_next, element_left_next = Tools.cut_list_into_combination(combine_num_next, right)

                for chose_next, left_next in zip(element_chose_next, element_left_next):
                    chose_combine = chose + chose_next
                    left_combine = left + left_next

                    element_chose_list.append(chose_combine)
                    element_left_list.append(left_combine)

        return element_chose_list, element_left_list

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
    def check_full_in_list(target, check_list, check_type='id'):
        """check if all item in the list are 'True'."""

        if not check_list:
            return False  # in case of []

        all_true = True

        for i in check_list:

            if check_type == 'id':
                if i is not target:  # 'is' to exclude 1 ( 1 == True is True here)
                    all_true = False
                    break

            if check_type == 'equal':
                if i != target:
                    all_true = False
                    break

        return all_true

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def check_any_in_list(target, check_list, check_type='id'):

        if not check_list:
            return False  # in case of []

        any_true = False

        for i in check_list:

            if check_type == 'id':
                if i is target:
                    any_true = True
                    break

            if check_type == 'equal':
                if i == target:
                    any_true = True
                    break

        return any_true

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def check_in_list(value, check_list):

        if value in check_list:
            return True
        else:
            return False

    # 模型 相关函数 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def save_model(save_path, df_path, node_map, root_function, scores):

        model = {}

        model['scores'] = scores
        model['data_path'] = df_path
        model['root_function'] = root_function

        for node in node_map.values():
            Tools.strip_node(node)
        model['node_map'] = node_map

        np.save('%s' % save_path, model)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def load_model(model_path, recal=False):

        model = np.load('%s' % model_path, allow_pickle=True)
        model = model.item()

        if recal:
            df = pd.read_csv(model['data_path'])
            df['timestamp'] = pd.to_datetime(df.timestamp)
            df.set_index('timestamp', drop=True, inplace=True)

            Tools.recal_model(df, model['node_map', model['root_function']])

        return model

    # 基因树 相关函数 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # LV.0 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_id(id_start='node'):
        """获取唯一ID：节点，分类器"""

        ID = id_start + '_' + str(int(time.time() * 10 ** 6)) + '_' + str(int(random.randint(100000, 999999)))
        return ID

    # LV.0 ------------------------------------------------------------------------------------------------------------
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

    # LV.0 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_value(value, *, start_value=None, end_value=None, sep=None, mul=3.0, scale=None, distribute='normal'):
        """ 对指定数值，根据相关参数进行变异。

        @param value: 待变异的数值
        @param sep: 最小步长
        @param mul:　整体变异幅度
        @param scale:　极端区域的变异幅度(>=0)
        @param distribute:
        @return: mutated value
        """

        if not distribute == 'normal':
            print('error 4765')
            return value  # 暂不支持其他类别变异

        if sep is None:
            # 20：主观设定，按 5% 自动计算步长。
            # 整数value的自动步长，要求value不能低于11，否则将sep将为0，无法逆转 [已解决]
            # 最大的缺点在于，对于数值大的value，sep也大，因此很难长期保持 --整体使得value的变异有变小的倾向
            sep = value / 20
            if isinstance(value, float):
                float_num = len(str(value).split('.')[1]) + 1
                sep = Tools.shorten_float(sep, float_num)
            else:
                if value >= 20:
                    sep = int(round(sep))
                else:
                    sep = 1            

        if sep == 0:
            return value

        if scale is None:
            scale = 1 / mul * 1.5  # 主观看着合适。。

        end_flut_mul = np.random.normal(loc=1.0, scale=scale)  # 末端变异增强参数
        delta = np.random.randn() * sep * mul * end_flut_mul
        new_value = Tools.fit_to_minimal(value + delta, min_range=sep)

        # 如value未变化，循环操作促使改变
        num = 0
        while new_value == value:

            if num > 30:
                if random.random() < 0.5:
                    new_value = value - sep
                else:
                    new_value = value + sep
                break  # 避免过度，浪费计算资源

            delta = np.random.randn() * sep * mul * end_flut_mul
            new_value = Tools.fit_to_minimal(value + delta, min_range=sep)
            num += 1


        # 范围检验
        if start_value and new_value < start_value:
            new_value = start_value
        if end_value and new_value > end_value:
            new_value = end_value

        # # 类型检验  --fit_to_minimal里面已经处理过了
        # if isinstance(value, int) or isinstance(sep, int):
        #     new_value = round(new_value)

        return new_value

    # LV.0 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def strip_node(node):
        """将某一节点的大数据剔除，节约存储。inplace"""

        # node结果数据
        node['node_result'] = 'stripped'

        if not node['terminal']:
            # 加权前数据 --无
            node['node_stripped'] = True
            return

        # terminal分类赋值数据 --无

        # terminal分类数据
        node['class_data'] = 'stripped'

        # 指标数据
        for kw, value in node['class_kw'].items():
            if isinstance(value, pd.Series):
                node['class_kw'][kw] = 'stripped'

        # TODO: update
        # 源数据
        # for func, kwargs in node['feature_args'].items():
        #     for param_name, value in kwargs.items():
        #         if isinstance(value, pd.DataFrame):
        #             node['feature_args'][func][param_name] = 'stripped'  # 源数据只有一个：df

        node['node_stripped'] = True

    # LV.0 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rebuild_leaf(node, df):
        """计算某个终端节点, inplace"""

        # LV.6 计算feature

        for func, kwargs in node['feature_args'].items():
            for name, param in kwargs.items():
                if name == 'df_source':
                    node['feature_args'][func][name] = df  # 添加源数据


        # TODO: update
        # for key, value in node['class_kw']:
        #     if value == 'stripped':
        #         func = node['class_kw_func'][key]
        #         feature = func(**node['feature_args'][func])
        #         node['class_kw'][key] = feature  # 还原feature数据

        # LV.5 计算分类
        class_function = node['class_func']
        node['class_data'] = class_function(*node['class_args_edges'], **node['class_kw'])  # 还原分类数据

        # LV.4 计算赋值
        mapped_data = Tools.get_mapped_data(node['class_data'], node['map_value_list'])

        # LV.3 --无

        # LV.2 赋予权重，获得节点数据
        node['node_result'] = mapped_data * node['weight']  # 还原结果数据

    # LV.0 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def build_branch(name, node_map):
        """计算某个中间节点。在确保下一级别 node_data 已有的情况下使用"""

        node = node_map[name]

        # 获取下一个级别的数据
        child_names = []
        slice_end = len(name)
        for k in node_map.keys():  # 这里用到了外围的node_map
            if k[:slice_end] == name and len(k) == slice_end + 1:
                child_names.append(k)

        args = []
        for child in child_names:
            args.append(node_map[child]['node_result'])

        # 计算节点数据
        node_function = node['node_function']
        node['node_result'] = node_function(*args) * node['weight']

    # LV.0 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def recal_model(df, node_map, root_function):
        """从 node_map 空壳获取模型的 pos_should"""

        # step1: 获取树叶 --解决terminal
        # node_map_copy = node_map.copy()  # avoid RuntimeError
        max_branch_len = 0
        for name, node in node_map.items():

            if node['terminal']:
                Tools.rebuild_leaf(node)
            else:
                max_branch_len = max(max_branch_len, len(name))

                # step2: 补充树枝
        waiting_room = {}
        for room_num in list(range(len(max_branch_len))):
            waiting_room[room_num + 1] = []  # +1 --对应name长度

        for name in node_map.keys():
            if not node['terminal']:
                room = len(name)
                waiting_room[room].append(name)

        room_num = max_branch_len
        while room_num >= 1:
            name_list = waiting_room[room_num]
            name_list.sort()

            for name in name_list:
                Tools.build_branch(name, node_map)

            room_num -= 1  # 要从最远的末稍开始计算，直到deputy node

        # step3: 获取模型的 pos_should
        deputy_list = []
        for name in node_map.keys():
            if len(name) == 1:
                deputy_list.append(name)
        deputy_list.sort()  # 就是为了这个sort，分开两块写

        root_args = []
        for name in deputy_list:
            root_args.append(node_map[name]['node_result'])

        pos_should_alpha = root_function(*root_args)

        return pos_should_alpha

    # LV.1 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def exchange_node(d1, d2, d1_node_name, d2_node_name):
        """LV.1 结构交叉进化函数

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

    # LV.2 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def cal_weight(signal, weight):
        """Return a weighted pos_should signal Series.

        @param signal: a signal Series
            index: timestamp
            value: pos_should signal
        @param weight: weight of the signal
        @return: weighted pos_should signal Series
        """

        return signal * weight

    # LV.2 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_weight(node_data, *, reweight_pb, sep=0.02, **kwargs):
        """LV.2 mutation: 权重进化函数

        NOTE: inplace.
        """

        node_type = node_data['node_type']
        if node_type == ('cond' or 'condition'):
            return False  # condition类不计算权重

        weight = node_data['weight']

        if random.random() < reweight_pb:
            node_data['weight'] = Tools.mutate_value(weight, sep=sep, mul=2)
            print('weight: changed.')
            return True

        else:
            return False

    # LV.3 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_lv3_methods():
        """node method / merge method"""

        lv3_method = dict(cond_1=Tools.cond_1,
                          cond_2=Tools.cond_2,

                          mult_simple=Tools.mult_simple,
                          mult_same=Tools.mult_same,
                          mult_abs=Tools.mult_abs,
                          divi_simple=Tools.divi_simple,
                          divi_same=Tools.divi_same,
                          divi_abs=Tools.divi_abs,

                          comb_sum=Tools.comb_sum,
                          comb_vote0=Tools.comb_vote0,
                          comb_vote1=Tools.comb_vote1,
                          comb_vote2=Tools.comb_vote2,
                          comb_vote3=Tools.comb_vote3,
                          comb_min=Tools.comb_min,

                          perm_add=Tools.perm_add,
                          perm_sub=Tools.perm_sub,
                          perm_up=Tools.perm_up,
                          perm_down=Tools.perm_down,
                          sig_trend_strict=Tools.sig_trend_strict,
                          sig_trend_loose=Tools.sig_trend_loose,
                          sig_trend_start_end=Tools.sig_trend_start_end
                          )

        return lv3_method

    # LV.3 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_node_function(node_data, *, refunc_pb, cross_refunc_pb):
        """LV.3 MUTATION: 信号合并函数进化"""

        func = node_data['node_function']
        new_func = None

        # 1. 类型改变。  --在改变类型后，传参约束条件会“变宽”，所以下列映射有方向性，是不可逆的。

        cross_exchange_dict = {Tools.mult_simple: Tools.mult_abs,  # 在simple有意义的情况下，计算结果不会有变化
                               Tools.divi_simple: Tools.divi_abs,  # 同上
                               Tools.mult_same: Tools.cond_2,  # 计算有意义区间相同，结果应该是相近的
                               Tools.divi_same: Tools.cond_3,  # 同上
                               }  # 这里只是随意打通一点，让“意外”发生，考虑不是太周全
        
        if func in cross_exchange_dict.keys():
            if random.random < cross_refunc_pb:
                new_func = cross_exchange_dict[func]

        # 2. 同类互换。

        # 可互换的函数分类（没有重叠，函数不会同时处在不同分类。下方pb_each计算的基础）
        exchange_dict = {'comb': [Tools.comb_sum, Tools.comb_min,
                                  Tools.comb_vote0, Tools.comb_vote1, Tools.comb_vote2, Tools.comb_vote3],
                         'perm': [Tools.perm_add, Tools.perm_up, Tools.perm_sub, Tools.perm_down],
                         'trend': [Tools.sig_trend_strict, Tools.sig_trend_loose, Tools.sig_trend_start_end],
                         '0/1': [Tools.cond_1, Tools.mult_simple, Tools.divi_simple],  # same rules for args
                         'mult': [Tools.cond_2, Tools.mult_same, Tools.mult_abs, Tools.divi_same, Tools.divi_abs]
                         }

        for method_list in exchange_dict.values():
            if func in method_list:

                method_list.remove(func)  # 每次都会重写exchange_dict，所以这里不用copy()
                if random.random() < refunc_pb:
                    target_num = random.randint(0, len(method_list) - 1)  # 不能同时有多个变异项（不独立），所以用这种方法
                    new_func = method_list[target_num]   # 如发生同类互换，会覆盖上方类型改变的结果。

        if new_func:
            node_data['node_function'] = new_func
            print('LV.3 function mutated.')
            return True     
        else:
            return False

    # LV.4-5 ----------------------------------------------------------------------------------------------------------
    @staticmethod
    def update_node_terminal(node_data, *, new_mapping_list=None, new_edge_list=None):
        """更新节点字典数据"""

        updated = False

        if new_mapping_list and new_mapping_list != node_data['map_value_list']:
            node_data['map_value_list'] = new_mapping_list.copy()
            updated = True

        if new_edge_list and new_edge_list != node_data['class_args_edges']:
            new_edge_list.sort()  # 记得要排序一下！
            node_data['class_args_edges'] = new_edge_list.copy()
            updated = True

        return updated

    # LV.4 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_mapped_data(class_data, map_value_list):
        """condition、multiplier 类别不需要经过这个函数"""

        mapped_data = class_data.copy()

        num = 0
        for value in map_value_list:
            mapped_data[mapped_data == num] = value
            num += 1

        return mapped_data  # not inplace. node_map 中并不保持分类映射后的数据，降低内存压力

    # LV.4 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_one_value_in_map(value, *, map_type='vector', jump_pb=0.1):

        new_value = value

        if map_type == 'vector':
            # bracket = [-1, 0, 1]  # value的值域
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

        elif map_type == ('cond' or 'condition'):
            # bracket = [0, 1]
            if value != 0:
                new_value = 0
            else:
                new_value = 1

        elif map_type == ('multiplier' or 'mult'):
            pass

        return new_value

    # LV.4 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_mapping_list(node_data, *, revalue_pb, jump_pb=0.2, cut_pb, clear_pb, merge_pb, smooth_pb, **kwargs):
        """LV.4 MUTATION: 特征分类赋值进化

        @param node_data: dict of the node in node_map, include:
            - 'map_value_list': the LIST to get mutated.
            - 'map_type': to direct the mutation.
            - 'class_args_edges': to be changed if any mutation happened on mapping_list.
        @param revalue_pb: probability of any  mutation happened.
        @param jump_pb:
        @param merge_pb: probability of merging 2(or more) same value into 1 value.
        @param cut_pb: probability of cutting 1 value into 2. ('class_args_edges' cut at random distance)
        @param clear_pb: probability of deleting the value between 2 same value.
        @param smooth_pb: probability of adding a 0 value betweent -1 & 1 (for 'vector' map_type only)
        @return: True for any mutation happened for mapping_list(and classify_args)

        NOTE: inplace. all pb are independent. Error if node_data['class_args_edges'] contains other than edges.
        """

        mutation_tag = False
        edge_list = node_data['class_args_edges']  # 分类的切割点边界值 
        map_type = node_data['map_type']

        if map_type == ('multiplier' or 'mult'):
            return mutation_tag  # 增强类型的数据，不进行分类赋值

        # 注意，mutation的5项的顺序是有考虑的，且完成一个到下一个，不要随意改变先后次序或合并
        # 1. mapping_list 连续同值合并【优化】 --merge_pb
        if merge_pb and len(edge_list) > 1:
            merged = Tools.mutate_mapping_list_merge_same(node_data, merge_pb=merge_pb)
            if merged:
                mutation_tag = True
                print('mapping_list: merged')

        # 2. mapping_list 跳值平滑【优化】  --smooth_pb
        if map_type == 'vector' and smooth_pb:
            smoothed = Tools.mutate_mapping_list_jump_smooth(node_data, smooth_pb=smooth_pb)
            if smoothed:
                mutation_tag = True
                print('mapping_list: smoothed')

        # 3. mapping_list 同值间异类剔除【半优化】  --clear_pb
        if clear_pb:
            cleared = Tools.mutate_mapping_list_clear_between(node_data, clear_pb=clear_pb)
            if cleared:
                mutation_tag = True
                print('mapping_list: cleared')

        # 4. mapping_list 数目增加 --cut_pb
        if cut_pb:
            cut = Tools.mutate_mapping_list_cut_within(node_data, cut_pb=cut_pb)
            if cut:
                mutation_tag = True
                print('mapping_list: cut')

        # 5. mapping_list 赋值变异 --revalue_pb  (这一步才是这个函数正儿八经最应该做的事情）
        if revalue_pb:
            changed = Tools.mutate_mapping_list_change_value(node_data, revalue_pb=revalue_pb, jump_pb=jump_pb)
            if changed:
                mutation_tag = True
                print('mapping_list: value_changed')

        return mutation_tag

    # LV.4 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_mapping_list_change_value(node_data, *, revalue_pb, jump_pb=0.2):
        """mapping_list 赋值变异 """

        mapping_list = node_data['map_value_list']
        mapping_list_new = node_data['map_value_list'].copy()
        map_type = node_data['map_type']

        revalue_pb_each = Tools.probability_each(object_num=len(mapping_list),
                                                 pb_for_all=revalue_pb)  # 各元素发生变异的独立概率

        n = 0
        for value in mapping_list:
            if random.random() < revalue_pb_each:
                new_value = Tools.mutate_one_value_in_map(value, map_type=map_type, jump_pb=jump_pb)
                mapping_list_new[n] = new_value
            n += 1

        # 更新node
        node_updated = Tools.update_node_terminal(node_data, new_mapping_list=mapping_list_new)

        return node_updated

    # LV.4 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_mapping_list_cut_within(node_data, *, cut_pb, add_zoom_mul=1.5):
        """mapping_list 数目增加：通过将某段赋值切开（但赋值仍相同）

        注：这个功能在LV.5 中有重复，且更科学（所以这里的概率设置低一些）
        """

        edge_list_new = node_data['class_args_edges'].copy()
        if not edge_list_new:
            return False  # 缺乏切割的参考edge值，会导致出错。

        mapping_list_new = node_data['map_value_list'].copy()
        zoom_of_sep = node_data['edge_mut_range']['sep']
        zoom_short_edge = node_data['edge_mut_range']['too_short']
        zoom_at_border = zoom_short_edge * add_zoom_mul  # 如新增在两端，用此确定切割的edge值
        if zoom_short_edge < zoom_of_sep * 3:
            zoom_short_edge = zoom_of_sep * 3  # 切割两端都需要至少（可等于）保留一个sep，故3

        cut_tag_list = list(range(len(mapping_list_new)))

        cut_pb_each = Tools.probability_each(object_num=len(cut_tag_list),
                                             pb_for_all=cut_pb)

        add_num = 0
        for i in cut_tag_list:
            if random.random() < cut_pb_each:

                if i == 0:
                    new_edge = Tools.fit_to_minimal(edge_list_new[0] - zoom_at_border,
                                                    min_range=zoom_of_sep)

                elif i == (len(cut_tag_list) - 1):
                    new_edge = Tools.fit_to_minimal(edge_list_new[-1] + zoom_at_border,
                                                    min_range=zoom_of_sep)

                else:
                    edge_before = edge_list_new[i + add_num - 1]
                    edge_after = edge_list_new[i + add_num]
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
                    edge_list_new.insert(i, new_edge)
                    add_value = mapping_list_new[i + add_num]
                    mapping_list_new.insert(i + add_num, add_value)  # 仅仅切割开，并不改变赋值

                    add_num += 1

        # 更新node
        node_updated = Tools.update_node_terminal(node_data, new_mapping_list=mapping_list_new,
                                                  new_edge_list=edge_list_new)

        return node_updated

    # LV.4 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_mapping_list_clear_between(node_data, *, clear_pb):
        """mapping_list 同值间异类剔除【半优化】

        注：这个功能在LV.5 中有重复，且更科学（所以这里的概率设置低一些）
        """

        mapping_list_new = node_data['map_value_list'].copy()
        edge_list_new = node_data['class_args_edges'].copy()
        zoom_of_sep = node_data['edge_mut_range']['sep']

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

        clear_pb_each = Tools.probability_each(object_num=len(clear_order_list),
                                               pb_for_all=clear_pb)
        pop_num = 0
        for i in clear_order_list:
            if random.random() < clear_pb_each:

                old_edge_before = edge_list_new[i - pop_num - 1]
                old_edge_after = edge_list_new[i - pop_num]
                cut_zoom = old_edge_after - old_edge_before
                new_edge_value = Tools.fit_to_minimal(old_edge_before + cut_zoom * random.random(),
                                                      min_range=zoom_of_sep)  # 处理方式：随机切割

                # 如在fit_to_minimal之后和两端相等，那就没法切割出有意义的赋值区间了
                if old_edge_before < new_edge_value < old_edge_after:
                    edge_list_new.pop(i - pop_num)
                    edge_list_new.pop(i - pop_num - 1)
                    edge_list_new.insert(i - pop_num - 1, new_edge_value)

                    mapping_list_new.pop(i - pop_num)  # 清除中间值，两头值并没有合并
                    pop_num += 1

        # 更新node
        node_updated = Tools.update_node_terminal(node_data, new_mapping_list=mapping_list_new,
                                                  new_edge_list=edge_list_new)

        return node_updated

    # LV.4 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_mapping_list_jump_smooth(node_data, *, smooth_pb, smooth_zoom_mul=1.5):
        """mapping_list 跳值平滑（插入）【优化】

        @param node_data:
        @param smooth_pb:
        @param smooth_zoom_mul: LV.5 mutation 中抹掉分类的临界值的倍数
        """

        mapping_list_new = node_data['map_value_list'].copy()
        edge_list_new = node_data['class_args_edges'].copy()

        # 这里的做法是，预先设定zoom值（固定），插入。后期可考虑动态调整zoom值(要考虑前后边界的情况：缺失、太窄等)
        add_zoom = node_data['edge_mut_range']['too_short'] * smooth_zoom_mul
        zoom_of_sep = node_data['edge_mut_range']['sep']

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

        smooth_pb_each = Tools.probability_each(object_num=len(add_tag_list),
                                                pb_for_all=smooth_pb)

        add_num = 0
        for i in add_tag_list:
            if random.random() < smooth_pb_each:
                mapping_list_new.insert(i + add_num, 0)  # 插入0到符号变化之间，以平滑

                arg_tag = i + add_num - 1
                old_edge = edge_list_new[i - 1]
                new_edge_before = Tools.fit_to_minimal(old_edge - add_zoom / 2,
                                                       min_range=zoom_of_sep)
                new_edge_after = Tools.fit_to_minimal(old_edge + add_zoom / 2,
                                                      min_range=zoom_of_sep)

                edge_list_new.pop(arg_tag)
                edge_list_new.insert(arg_tag, new_edge_after)
                edge_list_new.insert(arg_tag, new_edge_before)

        # 更新node
        node_updated = Tools.update_node_terminal(node_data, new_mapping_list=mapping_list_new,
                                                  new_edge_list=edge_list_new)

        return node_updated

    # LV.4 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_mapping_list_merge_same(node_data, *, merge_pb):
        """mapping_list 连续同值合并【优化】"""

        mapping_list_new = node_data['map_value_list'].copy()
        edge_list_new = node_data['class_args_edges'].copy()

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

        merge_pb_each = Tools.probability_each(object_num=len(merge_tag_list),
                                               pb_for_all=merge_pb)

        pop_num = 0
        for i in merge_tag_list:

            if len(edge_list_new) <= 1:
                break  # 如出现多个（如2个）切割edge，但对应的value（3个）相同时，有可能同时被pop，故限制

            if random.random() < merge_pb_each:
                mapping_list_new.pop(i - pop_num)
                edge_list_new.pop(i - pop_num - 1)  # 相同赋值的后者的下标，对应于前一个下标的切割器
                pop_num += 1

        # 更新node
        node_updated = Tools.update_node_terminal(node_data, new_mapping_list=mapping_list_new,
                                                  new_edge_list=edge_list_new)

        return node_updated

    # LV.5 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_lv5_method():
        """terminal method / classifier"""

        lv5_method = dict(cut_number=Tools.cut_number,
                          cut_rank=Tools.cut_rank,
                          cut_sigma=Tools.cut_sigma,
                          cut_distance=Tools.cut_distance,

                          compare_distance=Tools.compare_distance,
                          compare_sigma=Tools.compare_sigma,

                          perm_add=Tools.perm_add,
                          perm_sub=Tools.perm_sub,
                          perm_up=Tools.perm_up,
                          perm_down=Tools.perm_down,
                          sig_trend_strict=Tools.sig_trend_strict,
                          sig_trend_loose=Tools.sig_trend_loose,
                          sig_trend_start_end=Tools.sig_trend_start_end
                          )

        return lv5_method

    # LV.5 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_edge(node_data, *, move_pb, pop_pb, insert_pb, add_zoom_mul=1.5, **kwargs):
        """LV.5 MUTATION: 特征分类截取进化函数，对edge的值和数量进行优化

        @param node_data: 节点字典数据
        @param move_pb: 有至少一个edge发生移动的概率
        @param pop_pb: 当edge之间太近时，剔除分类的概率
        @param insert_pb: 当edge之间距离太远时，插入分类的概率
        @param add_zoom_mul: 当edge之间距离太远、插入分类时，后与原区间边界保持的距离（sep倍数）
        @return: True for any change happened.
        """

        mutation_tag = False

        # 1. edge 太近删除
        if pop_pb:
            poped = Tools.mutate_edge_pop(node_data, pop_pb=pop_pb)
            if poped:
                mutation_tag = True
                print('class_edge: poped.')

        # 2. edge 太远插入
        if insert_pb:
            inserted = Tools.mutate_edge_insert(node_data, insert_pb=insert_pb, add_zoom_mul=add_zoom_mul)
            if inserted:
                mutation_tag = True
                print('class_edge: inserted.')


        # 3. edge 切割点移动
        if move_pb:
            moved = Tools.mutate_edge_move(node_data, move_pb=move_pb)
            if moved:
                mutation_tag = True
                print('class_edge: moved.')

        return mutation_tag

    # LV.5 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_edge_move(node_data, *, move_pb):
        """切割边界变异：移动边界"""

        edge_list = node_data['class_args_edges']
        edge_list_new = []  # 这里的写法与 mutate_mapping_list_* 中的不同，因之前考虑到部分参数不是edge的情况（现已删除相关检验）
        zoom_of_sep = node_data['edge_mut_range']['sep']

        move_pb_each = Tools.probability_each(object_num=len(edge_list), pb_for_all=move_pb)

        for edge in edge_list:
            if random.random() < move_pb_each:
                new_edge = Tools.mutate_value(edge, sep=zoom_of_sep)
                while Tools.check_in_list(new_edge, edge_list):
                    new_edge = Tools.mutate_value(edge, sep=zoom_of_sep)  # 避免与已有的其他edge重合
                edge_list_new.append(new_edge)
            else:
                edge_list_new.append(edge)

        edge_list_new.sort()

        # 更新node
        node_updated = Tools.update_node_terminal(node_data, new_edge_list=edge_list_new)

        return node_updated

    # LV.5 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_edge_insert(node_data, *, insert_pb, add_zoom_mul=1.5):
        """切割边界变异：太远了，插入"""

        edge_list = node_data['class_args_edges']
        edge_list_new = node_data['class_args_edges'].copy()
        mapping_list = node_data['map_value_list']
        mapping_list_new = node_data['map_value_list'].copy()

        long_edge = node_data['edge_mut_range']['too_long']
        zoom_distance_edge = node_data['edge_mut_range']['too_short'] * add_zoom_mul
        zoom_of_sep = node_data['edge_mut_range']['sep']
        map_type = node_data['map_type']

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

                point1 = Tools.fit_to_minimal(zoom_line_before + zoom * random.random(), min_range=zoom_of_sep)
                point2 = Tools.fit_to_minimal(zoom_line_before + zoom * random.random(), min_range=zoom_of_sep)

                if point1 != point2:
                    edge_list_new.insert(i + add_num, max(point1, point2))
                    edge_list_new.insert(i + add_num, min(point1, point2))

                    cut_value = mapping_list[i]
                    mapping_list_new.insert(i + add_num,
                                            Tools.mutate_one_value_in_map(cut_value, map_type=map_type))
                    mapping_list_new.insert(i + add_num, cut_value)  # cut_value中间夹着一个变异后的value

                    add_num += 2  # NOTE 2 here.

        # 更新node
        node_updated = Tools.update_node_terminal(node_data, new_mapping_list=mapping_list_new,
                                                  new_edge_list=edge_list_new)

        return node_updated

    # LV.5 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_edge_pop(node_data, *, pop_pb):
        """切割边界变异：太近了，删除"""

        edge_list = node_data['class_args_edges']
        edge_list_new = node_data['class_args_edges'].copy()
        mapping_list = node_data['map_value_list']
        mapping_list_new = node_data['map_value_list'].copy()

        short_edge = node_data['edge_mut_range']['too_short']

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

        # 更新node
        node_updated = Tools.update_node_terminal(node_data, new_mapping_list=mapping_list_new,
                                                  new_edge_list=edge_list_new)

        return node_updated

    # LV.6 ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mutate_feature(node_data, *, kw_pb=0.1, refeature_pb=0.1, mul=1.5, **kwargs):
        """LV.6 特征参数进化"""

        # 1. mutate kwargs of the classifier function which is not feature data.

        key_list = []
        sep_list = []

        for key, value in node_data['class_kw_sep'].items():

            if value is None:
                continue

            key_list.append(key)
            sep_list.append(value)

        kw_pb_each = Tools.probability_each(object_num=len(key_list), pb_for_all=kw_pb)

        class_mut_flag = False
        for key, sep in zip(key_list, sep_list):

            if random.random() < kw_pb_each:

                old_value = node_data['class_kw'][key]

                # 目前这一块只有 window 一项是需要变异的，所以下方按照window的特点做出优化
                start_value = 10
                if sep is True:  # 如无提前设定sep，这里默认整数类型的sep （注意语法区别于 sep == True，混淆1）
                    if old_value > 20:
                        sep = None  # None 将默认按 5% 自动计算步长
                    else:
                        sep = 1

                new_value = Tools.mutate_value(old_value, start_value=10, sep=sep, mul=mul)

                if new_value != old_value:
                    node_data['class_kw'][key] = new_value

                    class_mut_flag = True
                    print('class_kw: "%s" value mutated.' % key)

        # 2. mutate feature

        

        # 字典结构的旧方法

        # # 提取feature计算函数
        # func_list = []
        # for value in node_data['class_kw_func'].values():
        #     if type(value) == type(Tools.agg_cal):
        #         func_list.append(value)

        # # 计算每个参数的变异概率
        # mut_arg_num = 0
        # for func in func_list:

        #     for mut_data in node_data['feature_args_range'][func]:
        #         if mut_data:
        #             if isinstance(mut_data, dict) and mut_data['sep']:  # 对于dict类别定义的参数范围
        #                 mut_arg_num += 1
        #             if isinstance(mut_data, list) and mut_data:  # 对于list类别定义的参数范围
        #                 mut_arg_num += 1  # += len(mut_data) - 1  # --太极端了，改变类别的概率不应该大于改变参数的概率

        # arg_mut_pb = Tools.probability_each(object_num=mut_arg_num, pb_for_all=refeature_pb)

        # # 参数的变异
        # feature_mut_flag = False

        # if not arg_mut_pb:
        #     return class_mut_flag, feature_mut_flag

        # for func in func_list:
        #     args_dict = node_data['feature_args'][func]  # 获得feature函数的 参数名-参数
        #     mut_data_range = node_data['feature_args_range'][func]  # 获得feature函数的 参数名-变化范围
        #     # args_dict_new = {}

        #     for param_name in args_dict.keys():
        #         if mut_data_range[param_name]:  # 可变异

        #             now_param = args_dict[param_name]
        #             new_param = None

        #             if random.random() < arg_mut_pb:  # 概率允许
        #                 choice_box = mut_data_range[param_name].copy()

        #                 if isinstance(choice_box, dict):
        #                     start_value = choice_box['start']
        #                     end_value = choice_box['end']
        #                     sep_value = choice_box['sep']
        #                     new_param = Tools.mutate_value(now_param, start_value=start_value, end_value=end_value, sep=sep_value)
                        
        #                 if isinstance(choice_box, list):
        #                     choice_box.remove(now_param)
        #                     new_param = choice_box[random.randint(0, len(choice_box) - 1)]

        #             if new_param:
        #                 node_data['feature_args'][func][param_name] = new_param
        #                 feature_mut_flag = True
        #                 print('feature_args for %s: %s value mutated.' % (func, param_name))

        return class_mut_flag, feature_mut_flag

    # terminal 生成函数 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def generate_leaf(num):
        """terminal 生成函数"""
        
        leaf = {}

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def build_leaf(*args, node_type, node_func=None, class_func, edge_sep, zoom_long, zoom_short, edge_num=5,
        leaf_kw, leaf_kw_sep, leaf_kw_func, ):

        pass

    # compare 分类序列生成 ----------------------------------------------------------------------------------------------
    @staticmethod
    def compare_distance(*sub_edge, feature1, feature2, window=0):
        """特征分类函数，差值对比，绝对距离（比大小：距离为0）  (缺点在于步长个性化太强，需要单独设置参数)

        @param sub_edge: the value which differs the subtract result. (NOTE: negative for feature1 < feature2!)
        @param window: unfunctional here.
        @return: compare classified result: 0, 1, 2 ...
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

        class_index = list(range(len(sub_edge) + 1))

        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result['diff_tag'], class_index

    # compare 分类序列生成 ----------------------------------------------------------------------------------------------
    @staticmethod
    def compare_sigma(*sigma_edge, feature1, feature2, window=0):
        """特征分类函数，差值对比，平均标准差比例对比（比大小：比例为0）

        @param sigma_edge: the sigma value which differs the subtract result. (NOTE: negative for feature1 < feature2!)
        @param window: unfunctional here.
        @return: compare classified result: 0, 1, 2 ...
        """

        pd.set_option('mode.chained_assignment', None)  # close SettingWithCopyWarning

        feature1.name = 'feature1'
        feature2.name = 'feature2'
        sigma_edge = sorted(sigma_edge)

        df_result = Tools.sig_merge(feature1, feature2)  # nan filled.
        df_result['difference'] = df_result['feature1'] - df_result['feature2']

        if window == 0:
            sigma = (feature1.std() + feature2.std()) / 2  # 这样计算用原始维度的数据作为加工后数据的sigma的标尺，含义与经典的不同
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

        class_index = list(range(len(sigma_edge) + 1))

        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result['diff_tag'], class_index

    # cut 分类序列生成 --------------------------------------------------------------------------------------------------
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

        class_index = list(range(len(cut_points) + 1))

        return df_result['cut'], class_index

    # cut 分类序列生成 --------------------------------------------------------------------------------------------------
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

        class_index = list(range(num + 1))

        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result['cut'], class_index

    # cut 分类序列生成 --------------------------------------------------------------------------------------------------
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

        class_index = list(range(num + 1))

        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result['cut'], class_index

    # cut 分类序列生成 --------------------------------------------------------------------------------------------------
    @staticmethod
    def cut_sigma(*cut_sigmas, feature, window=0):
        """特征分类函数，自切割，分布标准倍数切割  (缺点在于window长度，太短没有意义，太长初始化太慢，直接使用全部数据会有未来函数问题)

        @param feature: feature Series (timestamp index) to be cut
        @param window: rolling window for range reference
        @param cut_sigmas: float(0-1) of sigma to cut features apart
        @return: cut result: 0, 1, 2, 3...  (NO negative results here)

        NOTE: support negative-positive, positive-only and negative-only feature (但是外部的映射字典不要弄错)
        """

        pd.set_option('mode.chained_assignment', None)  # close SettingWithCopyWarning

        feature.name = 'data'
        df_result = pd.DataFrame(feature)
        df_result['cut'] = 0

        num_up = 0
        # num_down = 0
        cut_sigmas = sorted(cut_sigmas)  # NOTE
        for sigma in cut_sigmas:

            if window == 0:
                feature_mean = feature.mean()
                feature_edge = feature.std() * sigma
                df_result['cut_ref_up'] = feature_mean + feature_edge
                # df_result['cut_ref_down'] = feature_mean - feature_edge  # 可为负

            else:
                df_result['data_mean'] = df_result['data'].rolling(window).mean().fillna(method='bfill')
                df_result['sigma_value'] = df_result['data'].rolling(window).std().fillna(method='bfill') * sigma
                df_result['cut_ref_up'] = df_result['data_mean'] + df_result['sigma_value']
                # df_result['cut_ref_down'] = df_result['data_mean'] - df_result['sigma_value']

            df_result['cut'][df_result['data'] > df_result['cut_ref_up']] = num_up + 1
            # df_result['cut'][df_result['data'] < df_result['cut_ref_down']] = num_down - 1

            num_up += 1
            # num_down -= 1

        class_index = list(range(num_up + 1))

        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result['cut'], class_index

    # branch 构造函数 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def generate_branc():
        pass

    # 条件函数 ----------------------------------------------------------------------------------------------------------
    @staticmethod
    def cond_1(pos_should, cond):
        """CONDITION method: 0/1 condition
        ---在满足a(0/1)条件(1)的条件下，使用b的pos_should，其余0 【限2列】

        @param cond: condition to accept pos_should
        @param pos_should: pos_should signals Series (weighted)
        @return: pos_should signal

        NOTE: 为了保持函数可互换性，遵照Tools.divi_simple的参数顺序
        """

        pd.set_option('mode.chained_assignment', None)  # close SettingWithCopyWarning

        cond.name = 'condition'
        pos_should.name = 'signal'

        df_sig = pd.concat([cond, pos_should], axis=1, sort=False).fillna(method='ffill')
        df_sig['result_sig'] = df_sig['signal']
        df_sig['result_sig'][df_sig['condition'] <= 0] = 0

        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_sig['result_sig']

    # 条件函数 ----------------------------------------------------------------------------------------------------------
    @staticmethod
    def cond_2(pos_should, cond):
        """CONDITION method: -1/0/1 condition
        ---在满足cond方向（正负，对比0）的条件下，pos_should如同方向（正负），使用其值，其余为0 【限2列】

        @param cond: condition to accept pos_should
        @param pos_should: pos_should signals Series (weighted)
        @return: pos_should signal

        NOTE: 为了保持函数可互换性，遵循Tools.divi_same的参数顺序
        """

        df_sig = pd.concat([cond, pos_should], axis=1, sort=False).fillna(method='ffill')

        df_sig['result_ref'] = df_sig.iloc[:, 0] * df_sig.iloc[:, 1]
        df_sig['result_sig'] = df_sig.iloc[:, 1][df_sig['result_ref'] > 0]

        result_sig = df_sig['result_sig'].fillna(0)

        return result_sig

    # 增强、削弱函数 -----------------------------------------------------------------------------------------------------
    @staticmethod
    def sigmoid(x, coefficient=2):
        """增幅限制函数"""

        s = coefficient / (1 + np.exp(-x / coefficient))
        return s * 2 - coefficient

    # 增强、削弱函数 -----------------------------------------------------------------------------------------------------
    @staticmethod
    def mult_simple(vector_arr, abs_arr, coefficient=1.7):
        """增强函数：简单增强（简单相乘）。

        适用于一方是0/1类型，另一方-1/0/1类型，强化后者。如两个都是vector，将无法解释结果符号。"""

        vector_arr.name = 'vector_arr'
        abs_arr.name = 'abs_arr'

        df_mult = pd.concat([vector_arr, abs_arr], axis=1, sort=False).fillna(method='ffill')
        df_mult['ref'] = df_mult['abs_arr'] * df_mult['vector_arr']
        df_mult['result'] = df_mult['ref'].rolling(1).apply(lambda x: Tools.sigmoid(x, coefficient), raw=True)

        return df_mult['result'] * coefficient

    # 增强、削弱函数 -----------------------------------------------------------------------------------------------------
    @staticmethod
    def mult_same(arr1, arr2, coefficient=1.7):
        """增强函数：同向增强  --同方向时：保留符号，计算相乘结果；其他：0 """

        arr1.name = 'arr1'
        arr2.name = 'arr2'
        df_mult = pd.concat([arr1, arr2], axis=1, sort=False).fillna(method='ffill')  # 为了应对不同时间周期的arr数据

        arr1_abs = df_mult['arr1'].abs()
        arr2_abs = df_mult['arr2'].abs()
        arr_check_same = df_mult['arr1'] * df_mult['arr2']
        arr_check_neg = df_mult['arr1'].copy()
        arr_check_neg[arr_check_neg < 0] = -1
        arr_check_neg[arr_check_neg > 0] = 1

        df_mult['ref'] = arr1_abs * arr2_abs
        df_mult['ref'][arr_check_same <= 0] = 0
        df_mult['ref'] = df_mult['ref'] * arr_check_neg

        df_mult['result'] = df_mult['ref'].rolling(1).apply(lambda x: Tools.sigmoid(x, coefficient), raw=True)

        return df_mult['result'] * coefficient

    # 增强、削弱函数 -----------------------------------------------------------------------------------------------------
    @staticmethod
    def mult_abs(main_arr, affect_arr, coefficient=1.7):
        """增强函数：绝对增强  --主sig和副sig的绝对值相乘，改变主sig的幅度，不改变方向。"""

        main_arr.name = 'main'
        affect_arr.name = 'effect'
        df_mult = pd.concat([main_arr, affect_arr], axis=1, sort=False).fillna(method='ffill')

        effect_abs = df_mult['effect'].abs()
        ref_arr = df_mult['main'] * effect_abs

        df_mult['result'] = ref_arr.rolling(1).apply(lambda x: Tools.sigmoid(x, coefficient), raw=True)

        return df_mult['result'] * coefficient

    # 增强、削弱函数 -----------------------------------------------------------------------------------------------------
    @staticmethod
    def divi_simple(vector_arr, abs_arr, coefficient=1.5):
        """削弱函数：简单削弱（简单相除）， 且尽可能不超过dividend。

        适用于一方是0/1类型，另一方-1/0/1类型，强化后者。如两个都是vector，将无法解释结果符号。"""

        vector_arr.name = 'vector_arr'
        abs_arr.name = 'abs_arr'

        df_mult = pd.concat([vector_arr, abs_arr], axis=1, sort=False).fillna(method='ffill')
        ref_arr = df_mult['vector_arr'] / df_mult['abs_arr']
        # 避免接近0时的消极影响：
        restrict_arr = df_mult['vector_arr'].abs().rolling(1).apply(lambda x: Tools.sigmoid(x, coefficient), raw=True)

        df_mult['result'] = ref_arr.rolling(1).apply(lambda x: Tools.sigmoid(x, coefficient), raw=True)
        df_mult['result'] = df_mult['result'] * restrict_arr

        return df_mult['result'] * coefficient

    # 增强、削弱函数 -----------------------------------------------------------------------------------------------------
    @staticmethod
    def divi_same(dividend, divisor, coefficient=1.5):
        """削弱函数：同向削弱  --同方向时：保留符号，计算相除结果，且尽可能不超过dividend；其他：0 """

        dividend.name = 'dividend'  # 被除数
        divisor.name = 'divisor'
        df_mult = pd.concat([dividend, divisor], axis=1, sort=False).fillna(method='ffill')

        dividend_abs = df_mult['dividend'].abs()
        divisor_abs = df_mult['divisor'].abs()
        arr_check_same = df_mult['dividend'] * df_mult['divisor']
        arr_check_neg = df_mult['dividend'].copy()
        arr_check_neg[arr_check_neg < 0] = -1
        arr_check_neg[arr_check_neg > 0] = 1
        restrict_arr = df_mult['dividend'].abs().rolling(1).apply(lambda x: Tools.sigmoid(x, coefficient), raw=True)

        ref_arr = dividend_abs / divisor_abs
        ref_arr[arr_check_same <= 0] = 0
        ref_arr = ref_arr * arr_check_neg

        df_mult['result'] = ref_arr.rolling(1).apply(lambda x: Tools.sigmoid(x, coefficient), raw=True)
        df_mult['result'] = df_mult['result'] * restrict_arr

        return df_mult['result'] * coefficient

    # 增强、削弱函数 -----------------------------------------------------------------------------------------------------
    @staticmethod
    def divi_abs(main_arr, affect_arr, coefficient=1.5):
        """削弱函数：绝对削弱  --主sig处以副sig的绝对值，改变主sig的幅度，不改变方向。且尽可能不超过主sig。
        （当 affect_arr > 0 时，和 divi_simple 相同？"""

        main_arr.name = 'main'
        affect_arr.name = 'effect'
        df_mult = pd.concat([main_arr, affect_arr], axis=1, sort=False).fillna(method='ffill')

        effect_abs = df_mult['effect'].abs()
        # effect_abs = effect_abs.rolling(1).apply(lambda x: Tools.sigmoid(x, coefficient), raw=True)
        ref_arr = df_mult['main'] / effect_abs
        restrict_arr = df_mult['main'].abs().rolling(1).apply(lambda x: Tools.sigmoid(x, coefficient), raw=True)

        df_mult['result'] = ref_arr.rolling(1).apply(lambda x: Tools.sigmoid(x, coefficient), raw=True)
        df_mult['result'] = df_mult['result'] * restrict_arr

        return df_mult['result'] * coefficient

    # 多项合并函数 ------------------------------------------------------------------------------------------------------
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

    # 多项合并函数 ------------------------------------------------------------------------------------------------------
    @staticmethod
    def sig_delta(*signals):
        """Compare each Series line by line

        @param signals: Series
        @return: DataFrame

        note: old function name: sig_trend
        """

        df_sig = Tools.sig_merge(*signals)

        column_num = len(df_sig.columns)
        num = 0
        df_trend = pd.DataFrame()  # if len(signals) < 2: return an empty df

        while num < column_num - 1:
            df_trend[num] = df_sig.iloc[:, num + 1] - df_sig.iloc[:, num]
            num += 1

        return df_trend

    # 多项合并函数 ------------------------------------------------------------------------------------------------------
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

    # 多项合并函数 ------------------------------------------------------------------------------------------------------
    @staticmethod
    def sig_to_one(method, *signals):
        """分流函数

        @param method: method for merging signals into one
        @param signals: pos_should signals Series(weighted)f
        @return: pos_should signal
        """

        if not method in Method.ALL_NAME.value:
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

    # 多项合并函数 combination ------------------------------------------------------------------------------------------
    @staticmethod
    def comb_sum(*signals):
        """Pos_should signal merge method: comb_sum  
        ---对各列signal进行加和

        @param signals: pos_should signals Series (weighted)
        @return: pos_should signal
        """

        df_sig = Tools.sig_merge(*signals)
        result_sig = df_sig.sum(axis=1)

        return result_sig

    # 多项合并函数 combination ------------------------------------------------------------------------------------------
    @staticmethod
    def comb_vote0(*signals):
        """引用各列signal，按-1,0,1赋值（消除weight和小数点）再加和，输出为：-1/0/1"""

        df_sig = Tools.sig_merge(*signals)

        df_sig[df_sig > 0] = 1
        df_sig[df_sig < 0] = -1
        df_sig[df_sig == 0] = 0  # exclude float.

        return df_sig.sum(axis=1)

    # 多项合并函数 combination ------------------------------------------------------------------------------------------
    @staticmethod
    def comb_vote1(*signals):
        """Pos_should signal merge method: comb_vote1  
        ---使用各列signal投票，加和，输出为：-1/0/1

        @param signals: pos_should signals Series (weighted)
        @return: pos_should signal  -- -1/0/1
        """

        result_ref = Tools.comb_sum(*signals)  # NOTE this depends on comb_sum()

        df_result = pd.DataFrame(result_ref, columns=['result_ref'])
        df_result['result_sig'] = 0
        pd.set_option('mode.chained_assignment', None)  # close SettingWithCopyWarning 
        df_result['result_sig'][df_result['result_ref'] > 0] = 1
        df_result['result_sig'][df_result['result_ref'] < 0] = -1
        pd.set_option('mode.chained_assignment', 'warn')  # reopen SettingWithCopyWarning

        return df_result['result_sig']

    # 多项合并函数 combination ------------------------------------------------------------------------------------------
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

    # 多项合并函数 combination ------------------------------------------------------------------------------------------
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

    # 多项合并函数 combination ------------------------------------------------------------------------------------------
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

    # 多项合并函数 trend ------------------------------------------------------------------------------------------------
    @staticmethod
    def sig_trend_loose(*signals):
        """判断指标的趋势：一个比一个大：1, 一个比一个小：-1, 不一定：0
        比sig_trend_strict宽松：可分部分相等（部分指标数据连续相等，只要不全都相等），趋势仍成立。
        """

        df_trend = Tools.sig_delta(*signals)
        arr_temp = np.array(df_trend)

        direction_list = []
        for n in range(arr_temp.shape[0]):
            line = arr_temp[n]
            ref = 0
            if line.min() >= 0:
                ref = 1
                if Tools.check_full_in_list(0, list(line), check_type='equal'):
                    ref = 0  # 排除全部是0的
            elif line.max() <= 0:
                ref = -1
            direction_list.append(ref)

        trend = pd.Series(direction_list, index=df_trend.index)

        return trend

    # 多项合并函数 trend ------------------------------------------------------------------------------------------------
    @staticmethod
    def sig_trend_strict(*signals):
        """Check each df line to see if values (in each signal) goes straight or not

        @param signals: Series
        @return: one Series
            -1: signals go straight down
            1: signals go straight up
            0: signals don't go straight down (even/rebound within)

        note: old function name: sig_one_direction
        """

        df_trend = Tools.sig_delta(*signals)
        trend_signals = Tools.df_to_series(df_trend)
        result_ref = Tools.comb_min(*trend_signals)  # 注意这里用拆包语法

        result_ref.name = 'result_ref'
        df_sig = pd.DataFrame(result_ref)
        df_sig['result_sig'] = 0
        df_sig['result_sig'][df_sig['result_ref'] > 0] = 1  # NOTE ">=" can't be set here
        df_sig['result_sig'][df_sig['result_ref'] < 0] = -1

        return df_sig['result_sig']

    # 多项合并函数 trend ------------------------------------------------------------------------------------------------
    @staticmethod
    def sig_trend_start_end(*signals):
        """Check each df line to see if values (in each signal) eventually(start vs. end) goes up/down

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

        result_ref = Tools.sig_trend_strict(*signals)
        df_sig['one_direction'] = result_ref

        df_sig['result_ref'] = df_sig['direction'][df_sig['one_direction'] == 0]
        df_sig['result_sig'] = 0

        result_sig = df_sig['result_sig'].copy()
        result_sig[df_sig['result_ref'] > 0] = 1
        result_sig[df_sig['result_ref'] < 0] = -1

        return result_sig

    # 多项合并函数 permutation ------------------------------------------------------------------------------------------
    @staticmethod
    def perm_add(*signals):
        """Pos_should signal merge method: permutation addition
        --- 一直涨，sig值越来越大:1，否则0

        @param signals: pos_should signals Series (weighted)
        @return: signal: 0/1
        """

        result_ref = Tools.sig_trend_strict(*signals)

        result_ref.name = 'result_ref'
        df_sig = pd.DataFrame(result_ref)
        df_sig['result_sig'] = 0
        df_sig['result_sig'][df_sig['result_ref'] > 0] = 1

        return df_sig['result_sig']

    # 多项合并函数 permutation ------------------------------------------------------------------------------------------
    @staticmethod
    def perm_sub(*signals):
        """Pos_should signal merge method: permutation subtract
        --- 一直跌，sig值越来越小:1， 否则0

        @param signals: pos_should signals Series (weighted)
        @return: signal: 0/1
        """

        result_ref = Tools.sig_trend_strict(*signals)

        result_ref.name = 'result_ref'
        df_sig = pd.DataFrame(result_ref)
        df_sig['result_sig'] = 0
        df_sig['result_sig'][df_sig['result_ref'] < 0] = 1

        return df_sig['result_sig']

    # 多项合并函数 permutation ------------------------------------------------------------------------------------------
    @staticmethod
    def perm_up(*signals):
        """Pos_should signal merge method: permutation go up
        --- sig值震荡（含持平）上涨：1，否则0

        @param signals: pos_should signals Series (weighted)
        @return: signal: 0/1
        """

        result_ref = Tools.sig_trend_start_end(*signals)

        result_ref.name = 'result_ref'
        df_sig = pd.DataFrame(result_ref)
        df_sig['result_sig'] = 0
        df_sig['result_sig'][df_sig['result_ref'] > 0] = 1

        return df_sig['result_sig']

    # 多项合并函数 permutation ------------------------------------------------------------------------------------------
    @staticmethod
    def perm_down(*signals):
        """Pos_should signal merge method: permutation go up
        --- sig值震荡（含持平）下跌：1，否则0

        @param signals: pos_should signals Series (weighted)
        @return: signal: 0/1
        """

        result_ref = Tools.sig_trend_start_end(*signals)

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
            if slippage < MIN_DISTANCE and random.random() < 0.5:
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
                if random.random() < p:
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
                    if random.random() < up_pin_percentage:
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
            if slippage < MIN_DISTANCE and random.random() < 0.5:
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
                if random.random() < p:
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
                    if random.random() < down_pin_percentage:
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
                    slippage = Tools.fit_to_minimal(random.random() * random.random() * slip_range)

        else:

            if how == 'bad':
                slippage = slip_range / 2
            elif how == 'good':
                slippage = 0

        slippage = abs(Tools.fit_to_minimal(slippage, min_range=min_range))

        return slippage

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def random_fee_rate(p=0.5):
        """To return a random fee rate

        @param p: probability of being a market MAKER
        @return: fee rate value
        """
        if random.random() < p:
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

        dict_score = {'Sortino_ratio': -888.88,
                      'Calmar_ratio': -888.88,
                      'Annual_return': -888.88,
                      'Max_drawdown': -888.88,
                      'Sharpe_ratio': -888.88,
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
