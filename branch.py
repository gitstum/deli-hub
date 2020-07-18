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

        self.branch_depth = 0  # 下方最深有多少层node
        self.branch_width = 0  # 下方第一层有多少node
        self.branch_population = 0  # 总共有多少node（包括自身）

        self.score_list = []
        self.score_num = 0
        self.avg_score = 0

        self.node_result = pd.Series()  # weighted_data
        self.primitive_result = pd.Series()

        # ----------------------------

        self.node_box = node_box.copy()  # ~~

        if not intergrator_map:
            self.intergrator_map = Integrator.intergrator_map
        else:
            self.intergrator_map = intergrator_map

        if not primitive_pbs:
            self.primitive_pbs = Integrator.primitive_pbs
        else:
            self.primitive_pbs = primitive_pbs

        if not child_num_range:
            self.child_num_range = Integrator.child_num_range.copy()
        else:
            self.child_num_range = child_num_range

        if not node_data:
            self.node_data = Integrator.node_data.copy()  # ~~
        else:
            self.node_data = node_data

        if not lv_mut_tag:
            self.lv_mut_tag = Integrator.lv_mut_tag.copy()
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
        list_input_restrict_abs = self.intergrator_map['mutation_map']['input_restrict_abs']
        list_input_restrict_2 = self.intergrator_map['mutation_map']['input_restrict_2']
        list_input_addable = self.intergrator_map['mutation_map']['input_addable']
        list_input_pos_only = self.intergrator_map['mutation_map']['input_only_pos']

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

        if method in list_input_pos_only:
            self.node_data['input_pos_only'] = True
        else:
            self.node_data['input_pos_only'] = False

    def __fill_output_type(self):

        method = self.node_data['inter_method']
        list_output_cond = self.intergrator_map['mutation_map']['output_01']
        list_special = self.intergrator_map['mutation_map']['input_all_abs_then_output_abs']

        if method in list_output_cond:
            self.node_data['output_cond'] = True
        else:
            self.node_data['output_cond'] = False

        if method in list_special:
            all_abs = True
            for instance in self.node_data['method_ins']:
                if instance.node_data['node_type'] != 'abs_value':
                    all_abs = False
            if all_abs:
                self.node_data['output_cond'] = True

        if self.node_data['output_cond']:
            self.node_data['node_type'] = 'abs_value'
        else:
            self.node_data['node_type'] = 'pos_value'

    def __get_random_child_num(self):

        keep = self.child_num_range['keep']
        start = self.child_num_range['start']
        sep = self.child_num_range['sep']
        end = self.child_num_range['end'] + sep

        child_num = random.choice(list(range(start, end, sep)))

        if not keep:
            self.child_num_range['start'] = 2  # 有意义的最低值
            self.child_num_range['end'] = 51  # 编码的最大值
            self.child_num_range['sep'] = 1

        return child_num

    def __get_random_child_nodes(self, *, terminal_child_only=False):

        method = self.node_data['inter_method']
        node_box = self.node_box.copy()

        if terminal_child_only:
            for key in node_box.keys():
                node_box[key] = []
            for key, value in self.node_box:
                if isinstance(value, Terminal):
                    node_box[key].append(value)

        child_num = self.__get_random_child_num()
        # print('child_num', child_num)
        # print(self.node_box)

        method_ins_list = []

        # 目前下面（基于method的）分类合理。
        if self.node_data['input_pos_only']:
            while child_num > 0:
                instance = random.choice(node_box['pos_value'])
                method_ins_list.append(instance)
                child_num -= 1

        elif self.node_data['input_abs']:  # 目前这一类里面都是2个变量的。
            instance_1 = random.choice(node_box['all'])
            instance_abs = random.choice(node_box['abs_value'])
            method_ins_list = [instance_1, instance_abs]

        else:
            while child_num > 0:
                instance = random.choice(node_box['all'])
                method_ins_list.append(instance)
                if self.node_data['input_2'] and len(method_ins_list) >= 2:
                    break
                child_num -= 1

        # check
        if self in method_ins_list:
            raise ValueError('循环添加自身')

        return method_ins_list

    def __child_instance_bug(self):

        child_lv1_list = self.node_data['method_ins']

        if not child_lv1_list:
            return True  # None

        if isinstance(child_lv1_list, bool):
            print(child_lv1_list, self.node_data['method_ins'])

        if len(child_lv1_list) < 2:
            print('TypeError: missing required positional argument in %s for %s. Regenerate method_ins. 19851' % (self, self.node_data['inter_method']))
            return True

        for child_lv1 in child_lv1_list:
            if not isinstance(child_lv1, Primitive):
                continue
            child_lv2_list = child_lv1.node_data['method_ins']
            if child_lv1 in child_lv2_list:
                print('Error: 自包含 lv1 64')
                return True
            # if set(child_lv2_list) & set(child_lv1_list):
            #     print('Error: 自包含 lv2 654')   # 儿子的儿子是自己的另一个儿子，不算自包含
            #     print(child_lv1_list, child_lv2_list)
            #     return True

    def __get_branch_info(self):

        self.branch_width = len(self.node_data['method_ins'])

        self.branch_population = 1
        branch_list = []
        for instance in self.node_data['method_ins']:
            if isinstance(instance, Terminal):
                self.branch_population += 1
            elif isinstance(instance, Primitive):
                self.branch_population += instance.branch_population
                branch_list.append(instance)
            else:
                raise ValueError('instance is neither Terminal nor Primitive: %s' % instance)

        self.branch_depth = 0
        for instance in branch_list:
            self.branch_depth = max(self.branch_depth, instance.branch_depth)
        self.branch_depth += 1  # include self.  

    def __mutate_primitive_method(self):

        # TODO: test

        if not random.random() < self.primitive_pbs['refunc_pb']:
            return False

        method = self.node_data['inter_method']
        mutation_map = self.intergrator_map['mutation_map'].copy()
        child_num = len(self.node_data['method_ins'])
        input_type = self.node_data['method_ins'][0].node_data['node_type']  # abs/pos
        output_type = self.node_data['node_type']

        # 不导致计算出错的，都可以换
        if child_num > 2:
            mutation_map.pop('input_restrict_abs')
            mutation_map.pop('input_restrict_2')

        choice_box = []
        for group, method_list in mutation_map.items():

            if method in method_list:
                choice_box += method_list

                # 对于条件性质的输入输出，增加一点选中同类方法的概率：
                if input_type == 'abs_value' and group == 'input_restrict_abs':
                    choice_box += method_list
                if output_type == 'abs_value' and group == 'output_01':
                    choice_box += method_list

        while method in choice_box:
            choice_box.remove(method)

        if not choice_box:
            return False

        new_method = random.choice(choice_box)
        self.node_data['inter_method'] = new_method

        self.__fill_input_type()
        self.__fill_output_type()

        return True

    def mutate_offspring_num(self):

        pass
        return False

    # ----------------------------------------------------------------------------------------------------------------------

    def create_primitive(self, terminal_child_only=False):
        """create a random primitive  --与terminal不同，这里是不会计算并返还结果的。"""

        # 1. 确定合成方法
        self.node_data['inter_method'] = self.__get_random_method()

        # 2. 确定输入数据
        self.__fill_input_type()

        # 3. 选取子节点
        self.node_data['method_ins'] = self.__get_random_child_nodes(terminal_child_only=terminal_child_only)
        while self.__child_instance_bug():
            self.node_data['method_ins'] = self.__get_random_child_nodes()

        # 4. 登记输出数据类别
        self.__fill_output_type()

        # 5. 生成branch属性
        self.__get_branch_info()

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
        
        # TODO test it

        self.score_list.append(score)

        for instance in self.node_data['method_ins']:
            instance.add_score(score)

    def update_score(self, sortino_score):
        
        # TODO  test it

        self.avg_score = (self.avg_score * self.score_num + sortino_score) / (self.score_num + 1)
        self.score_num += 1

        for instance in self.node_data['method_ins']:
            instance.update_score(sortino_score)

    def mutate_primitive(self):

        mutation_tag = False

        self.lv_mut_tag[1] = self.mutate_offspring_num()
        self.lv_mut_tag[2] = Tools.mutate_weight(self.node_data, reweight_pb=self.primitive_pbs['reweight_pb'])
        self.lv_mut_tag[3] = self.__mutate_primitive_method()

        if self.lv_mut_tag[1] or self.lv_mut_tag[2] or self.lv_mut_tag[3]:
            mutation_tag = True

        return mutation_tag

    def recal(self):

        self.lv_mut_tag = Integrator.lv_mut_tag
        self.cal()

    def cal(self):

        # TODO: test

        if self.lv_mut_tag[3]:  # lv.3

            func = self.node_data['inter_method']
            instance_list = self.node_data['method_ins']

            args = []
            for instance in instance_list:

                sig_series = instance.node_result.copy()

                # bug 处理
                if isinstance(sig_series, pd.DataFrame):
                    print('ERROR: signal(node_result) should not be DataFrame. 654852')
                    print('Where: ', self, self.node_data['inter_method'])
                    
                    if sig_series.shape[1] > 2:
                        raise ValueError('ERROR: signal has more than 2 columns. 75249')

                    sig_series.columns = ['a', 'b']
                    if sig_series[sig_series['a'] != sig_series['b']].shape[0] > 0:
                        raise ValueError('ERROR: signal columns are not the same. 95498')
                    else:  # 出错的典型情况：Series变成两列的DataFrame
                        print('Modify 654852: signal columns are the same, keep one.')

                        # do the modify
                        instance.node_result = sig_series['a'].copy()
                        sig_series = instance.node_result.copy()

                args.append(sig_series)

            self.primitive_result = func(*args)
            self.node_result = Tools.cal_weight(self.primitive_result, self.node_data['weight'])

            print('c', end='; ')

        elif self.lv_mut_tag[2]:  # lv.2

            self.node_result = Tools.cal_weight(self.primitive_result, self.node_data['weight'])

        for key in self.lv_mut_tag.keys():
            self.lv_mut_tag[key] = False  # reset mutation_tag

        # check datatype. 即使这里做了检验，但有的 node的node_result 还是莫名其妙是两列的DataFrame，且两列内容一样。猜测是python本身的bug。
        if isinstance(self.node_result, pd.DataFrame):
            raise TypeError('ERROR: node_result should not be DataFrame.',
                            self, self.node_data['inter_method'], self.node_data['method_ins'])

        return self.node_result
