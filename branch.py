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

    def __init__(self, *, node_box, intergrator_map=None, primitive_pbs=None, child_num_range=None,
                 node_data=None, lv_mut_tag=None):

        # rank --------------------------------------------
        self.name = self.get_id('primitive')
        self.score_list = []
        self.score_num = 0
        self.avg_score = 0

        # get data ----------------------------------------
        self.node_result = pd.Series()  # weighted_data
        self.primitive_result = pd.Series()

        self.depth = 0  # 下方最深有多少层node。terminal: 0
        self.width = 0  # 下方第一层有多少node。terminal: 0
        self.population = 0  # 总共有多少node（包括自身）。terminal: 1

        # inputs ------------------------------------------
        self.node_box = node_box  # .copy()  # ~~

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
            self.child_num_range = child_num_range.copy()

        # rebuild model -----------
        if not node_data:
            self.node_data = Integrator.node_data.copy()  # ~~
        else:
            self.node_data = node_data.copy()

        if not lv_mut_tag:
            self.lv_mut_tag = Integrator.lv_mut_tag.copy()
        else:
            self.lv_mut_tag = lv_mut_tag.copy()

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

        if len(child_lv1_list) < 2:
            print('TypeError: missing required positional argument in %s for %s. Regenerate method_ins. 19851' % (
                self, self.node_data['inter_method']))
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

    def __update_branch_info(self):

        self.width = len(self.node_data['method_ins'])

        self.population = 1
        self.depth = 0
        for instance in self.node_data['method_ins']:
            self.population += instance.population
            self.depth = max(self.depth, instance.depth)
        self.depth += 1  # include self.  

    def __mutate_primitive_method(self):

        if not random.random() < self.primitive_pbs['refunc_pb']:
            return False

        method = self.node_data['inter_method']
        mutation_map = self.intergrator_map['mutation_map'].copy()
        child_num = len(self.node_data['method_ins'])
        input_type = self.node_data['method_ins'][0].node_data['node_type']  # abs/pos
        output_type = self.node_data['node_type']

        choice_box = []
        for group, method_list in mutation_map.items():

            if method in method_list:
                choice_box += method_list

                # 对于条件性质的输入输出，增加一点选中同类方法的概率：
                if input_type == 'abs_value' and group == 'input_restrict_abs':
                    choice_box += method_list
                if output_type == 'abs_value' and group == 'output_01':
                    choice_box += method_list

        # 排除自身
        while method in choice_box:
            choice_box.remove(method)

        # 排除导致计算出错的(不导致计算出错的，都可以换)
        if child_num > 2:
            for restrict_method in mutation_map['input_restrict_2']:
                while restrict_method in choice_box:
                    choice_box.remove(restrict_method)

        if not choice_box:
            print('note: choice_box empty, no mutation for inter_method: %s. 89463' % method)
            return False

        new_method = random.choice(choice_box)
        self.node_data['inter_method'] = new_method

        self.__fill_input_type()
        self.__fill_output_type()

        print('lv.3 mutation: intergrator method changed to: ', new_method)

        return True

    def __mutate_primitive_offspring(self, node_box):
        """变更子节点"""

        mutation_tag = False

        changed = self.__mutate_primitive_offspring_change(node_box)  # 变换
        if changed:
            mutation_tag = True

        if self.node_data['input_addable']:

            poped = self.__mutate_primitive_offspring_pop()  # 剔除
            added = self.__mutate_primitive_offspring_add(node_box)  # ”新增“

            if poped or added:
                mutation_tag = True

        return mutation_tag

    def __mutate_primitive_offspring_change(self, node_box):
        """子节点变更：node_box 中同类数据类型的随机替换"""

        if not random.random() < self.primitive_pbs['changechild_pb']:
            return False

        instance_list = self.node_data['method_ins']  # view

        child = random.choice(instance_list)
        node_type = child.node_data['node_type']
        new_child = random.choice(node_box[node_type])
        new_child = new_child.copy()  # ~~
        change_map = {child: new_child}

        new_instance_list = [change_map[i] if i in change_map else i for i in instance_list]

        self.node_data['method_ins'] = new_instance_list.copy()

        print('lv.3 mutation: primitive offspring changed.')

        return True

    def __mutate_primitive_offspring_pop(self):
        """子节点剔除：随机剔除"""

        instance_list = self.node_data['method_ins']  # view

        if not len(instance_list) >= 3:
            return False  # at least 2 child nodes

        if not random.random() < self.primitive_pbs['popchild_pb']:
            return False

        child = random.choice(instance_list)
        instance_list.remove(child)

        self.node_data['method_ins'] = instance_list.copy()

        print('lv.3 mutation: primitive offspring poped.')

        return True

    def __mutate_primitive_offspring_add(self, node_box):
        """子节点新增
        permutation类：随机选一个已有child，复制， 变异20次。
        combination类：随机新增 pos_value
        """

        addchild_pb = self.primitive_pbs['addchild_pb']

        if not random.random() < addchild_pb:
            return False

        instance_list = self.node_data['method_ins']  # view
        method = self.node_data['inter_method']
        mut_map = self.intergrator_map['mutation_map']

        if method in mut_map['permutation']:

            new_instance = random.choice(instance_list).copy()
            if isinstance(new_instance, Primitive):
                n = 0
                while n < (1 / addchild_pb / 2):  # 避免概率太高，死循环
                    new_instance.mutate_primitive(node_box=node_box)
                    # print('re_mutate_primitive counting: %d' % n)
                    n += 1
                new_instance.recal()  # note: recal()

            elif isinstance(new_instance, Terminal):
                n = 0
                while n < 20:  # 10次和20次时间消耗差别不大。。
                    new_instance.mutate_terminal()
                    # print('mutate terminal counting: %d' % n)
                    n += 1
                new_instance.recal()  # note: recal()

            else:
                raise TypeError('new_instance not Primitive nor Terminal. 6515')

        elif method in mut_map['combination']:
            new_instance = random.choice(node_box['pos_value'])

        else:
            new_instance = random.choice(node_box['all'])

        insert_num = random.randint(0, len(instance_list))
        self.node_data['method_ins'].insert(insert_num, new_instance)

        print('lv.3 mutation: primitive offspring added for %s' % method)

        return True

    # ------------------------------------------------------------------------------------------------------------------

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
        self.__update_branch_info()

        # 5. 向tree、forest报告
        return self.node_data['method_ins']

    def get_args(self, strip=True):
        return self.node_data

    def get_node_map():
        pass

    def copy(self):
        """深度复制. tested."""

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
        new_node_data['method_ins'] = child_instance_list.copy()  # 向tree、forest报告??

        return new_instance

    def add_score(self, score):

        self.score_list.append(score)

        for instance in self.node_data['method_ins']:
            instance.add_score(score)

    def update_score(self, sortino_score):

        self.avg_score = (self.avg_score * self.score_num + sortino_score) / (self.score_num + 1)
        self.score_num += 1

        for instance in self.node_data['method_ins']:
            instance.update_score(sortino_score)

    def mutate_primitive(self, *, node_box=None, update_node_box=True):

        if node_box:
            if update_node_box:
                self.node_box = node_box.copy()
        else:
            node_box = self.node_box

        mutation_tag = False

        self.lv_mut_tag[2] = Tools.mutate_weight(self.node_data, reweight_pb=self.primitive_pbs['reweight_pb'])

        self.lv_mut_tag[3] = False
        if self.__mutate_primitive_method() or self.__mutate_primitive_offspring(node_box):
            self.lv_mut_tag[3] = True

        if self.lv_mut_tag[2] or self.lv_mut_tag[3]:
            self.__update_branch_info()
            mutation_tag = True

        return mutation_tag


    def recal(self):

        self.lv_mut_tag = Integrator.lv_mut_tag
        self.cal()

    def cal(self):

        if self.lv_mut_tag[3]:  # lv.3

            func = self.node_data['inter_method']
            instance_list = self.node_data['method_ins']

            args = []
            for instance in instance_list:
                sig_series = instance.node_result.copy()
                args.append(sig_series)

            self.primitive_result = func(*args)
            self.node_result = Tools.cal_weight(self.primitive_result, self.node_data['weight'])

        elif self.lv_mut_tag[2]:  # lv.2

            self.node_result = Tools.cal_weight(self.primitive_result, self.node_data['weight'])

        for key in self.lv_mut_tag.keys():
            self.lv_mut_tag[key] = False  # reset mutation_tag

        return self.node_result
