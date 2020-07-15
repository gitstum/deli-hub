# --coding:utf-8--

import random
import time
import pandas as pd
import numpy as np

from tools import Tools
from private.parameters import Integrator
from leaf import Terminal


class Primitive(Tools):
    name = 'primitive'
    score_list = []
    score_num = 0
    avg_score = 0

    def __init__(self, *, node_box, intergrator_map=None, primitive_pbs=None, child_num_range=None, node_data=None,
                 lv_mut_tag=None):

        self.name = self.get_id('primitive')
        self.score_list = []
        self.score_num = 0
        self.avg_score = 0

        self.node_result = pd.Series()  # weighted_data
        self.primitive_result = pd.Series()

        self.node_box = node_box

        if not intergrator_map:
            self.intergrator_map = Integrator.intergrator_map
        else:
            self.intergrator_map = intergrator_map

        if not primitive_pbs:
            self.primitive_pbs = Integrator.primitive_pbs
        else:
            self.primitive_pbs = primitive_pbs

        if not child_num_range:
            self.child_num_range = Integrator.child_num_range
        else:
            self.child_num_range = child_num_range

        if not node_data:
            self.node_data = Integrator.node_data
        else:
            self.node_data = node_data

        if not lv_mut_tag:
            self.lv_mut_tag = Integrator.lv_mut_tag
        else:
            self.lv_mut_tag = lv_mut_tag

        self.inter_data = pd.Series()

    def __get_weighted_data(self):

        weighted_data = self.inter_data * self.node_data['weight']
        self.primitive_result = weighted_data

        return weighted_data

    def __get_random_method(self, update=True):

        all_method = self.intergrator_map['all_method']

        weight_all = []
        for weight in all_method.values():
            weight_all.append(weight)

        pb_for_weight = 1 / sum(weight_all)

        method_box = []
        while not method_box:
            for method, weight in all_method.items():
                if random.random() < weight * pb_for_weight:
                    method_box.append(method)

        method = random.choice(method_box)

        if update:
            self.node_data['inter_method'] = method

        return method

    def __fill_input_type(self):

        method = self.node_data['inter_method']
        list_input_restrict_abs = self.intergrator_map['input_restrict_abs']
        list_input_restrict_2 = self.intergrator_map['input_restrict_2']
        list_input_addable = self.intergrator_map['input_addable']

        if method in list_input_restrict_abs:
            self.node_data['input_abs'] = True
        else:
            self.node_data['input_abs'] = False

        if method in list_input_restrict_2:
            self.node_data['input_2'] = True
        else:
            self.node_data['input_2'] = False

        if method in list_input_addable:
            self.node_data['input_addable'] = True
        else:
            self.node_data['input_addable'] = False

    def __fill_output_type(self):

        method = self.node_data['inter_method']
        list_output_cond = self.intergrator_map['output_01']

        if method in list_output_cond:
            self.node_data['output_cond'] = True
        else:
            self.node_data['output_cond'] = False

        if self.node_data['output_cond']:
            self.node_data['node_type'] = 'abs_value'
        else:
            self.node_data['node_data'] = 'pos_value'

    def __get_random_child_num(self):

        keep = self.child_num_range['keep']
        start = self.child_num_range['start']
        sep = self.child_num_range['sep']
        end = self.child_num_range['end'] + sep

        child_num = random.choice(list(range(start, end, sep)))

        if not keep:
            self.child_num_range['start'] = 2  # 有意义的最低值
            self.child_num_range['end'] = 52  # 编码的最大值
            self.child_num_range['sep'] = 1

        return child_num

    def __get_random_child_nodes(self, update=True):

        method_ins = []
        child_num = self.__get_random_child_num()
        if self.node_data['input_abs']:
            instance_1 = random.choice(self.node_box['abs_value'])  # 针对条件函数
            method_ins.append(instance_1)
            child_num -= 1

        instance_2 = random.choice(self.node_box['pos_value'])
        method_ins.append(instance_2)
        child_num -= 1

        while child_num > 0:
            if self.node_data['input_addable']:
                instance = random.choice(self.node_box['pos_value'])
                method_ins.append(instance)
            child_num -= 1

        return method_ins

    def create_primitive(self):
        """create a random primitive  --与terminal不同，这里是不会计算并返还结果的。"""

        # 1. 确定合成方法
        self.node_data['inter_method'] = self.__get_random_method()

        # 2. 确定输入数据
        self.__fill_input_type()

        # 3. 选取子节点
        self.node_data['method_ins'] = self.__get_random_child_nodes()

        # 4. 登记输出数据类别
        self.__fill_output_type()

        # 5. 向tree、forest报告
        return self.node_data['method_ins']

    def get_args(self, strip=True):
        return self.node_data

    def copy(self):

        # TODO 递归，应该能全部做到深度复制 test it 

        new_instance = self.__class__(node_box=self.node_box,
                                      intergrator_map=self.intergrator_map,
                                      primitive_pbs=self.primitive_pbs,
                                      child_num_range=self.child_num_range,
                                      node_data=self.node_data,
                                      lv_mut_tag=self.lv_mut_tag)

        new_instance.__dict__ = self.__dict__.copy()
        new_instance.name = self.get_id('%s' % self.name.split('_')[0])

        new_instance.__dict__['node_data'] = self.__dict__['node_data'].copy()

        new_node_data = new_instance.__dict__['node_data']  # view
        child_instance_list = []
        for instance in new_node_data['method_ins']:
            child_instance_list.append(instance.copy())
        new_node_data['method_ins'] = child_instance_list.copy()   # 向tree、forest报告??
        
        return new_instance

    def add_score(self, score):
        pass  # TODO

    def update_score(self, sortino_score):
        pass  # TODO

    def mutate_primitive(self):

        upload_child_node = False

        reweighted = Tools.mutate_weight(self.node_data, )

        return upload_child_node

    def cal(self):

        if self.lv_mut_tag[3]:  # lv.3

            func = self.node_data['inter_method']
            instance_list = self.node_data['method_ins']

            args = []
            for instance in instance_list:
                args.append(instance.node_result)

            self.primitive_result = func(*args)

            self.node_result = Tools.cal_weight(self.primitive_result, self.node_data['weight'])

        elif self.lv_mut_tag[2]:  # lv.2

            self.node_result = Tools.cal_weight(self.primitive_result, self.node_data['weight'])

        return self.node_result
