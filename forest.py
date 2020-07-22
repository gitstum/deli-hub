# --coding:utf-8--
import random
import time
import pandas as pd
import numpy as np

from tools import Tools
from leaf import Terminal
from branch import Primitive
from tree import Tree

"""
初始化步骤：
1. 随机创建一堆 leaf，分类放入 node_box
2. 随机创建一堆一层的 branch，带有leaf数据，用branch_map记录。然后全部分类放入 node_box
3. 随机创建多层（限制层数）的branch，带有下级所有数据，用branch_map记录。然后全部分类放入 node_box
4. 创建一堆tree的根节点，随机选择node作为子节点。注意要copy
5. 开始计算回测结果、反馈、交叉、变异。

node_box分级末位淘汰制。其中0层保留20%，1层保留30%。
里面的内容不会变异。

branch_map不干涉内存管理，但能确定关系。


分数的添加：
所有子节点的实例本身能够添加分数
feature所属的indicator能够添加分数


"""


class Forest(Tools):

    name = 'forest'

    def __init__(self):

        self.name = Tools.get_id('forest')

        self.node_box = {
                'all': [],
                'pos_value': [],
                'abs_value': [],
            }

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def add_node_to_box(node_box, node_instance):
        """根据 node 的数据类型（pos/abs）分类添加到 node_box
        NOTE:inplace.
        """

        node_type = node_instance.node_data['node_type']  # abs/pos
        node_box[node_type].append(node_instance)
        node_box['all'].append(node_instance)

    @staticmethod
    def merge_note_box(*args):
        """将两个不同的node_box进行合并，并返还新的node_box。
        NOTE: not inplace
        """

        all_value_list = []
        pos_value_list = []
        abs_value_list = []

        for box in args:
            all_value_list += box['all']
            pos_value_list += box['pos_value']
            abs_value_list += box['abs_value']

        node_box = {
            'all': all_value_list,
            'pos_value': pos_value_list,
            'abs_value': abs_value_list,
        }

        return node_box

    @staticmethod
    def generate_leaves(terminal_num, *, df_source, node_box=None,
                        terminal_pbs=None, classifier_map=None, classifier_group=None):
        """随机生成指定数量的Terminal，返还填充后的 node_box
        NOTE: 如传入node_box，则inplace。
        """

        print('\nGenerate leaves. counting:')

        if not node_box:
            node_box = {
                'all': [],
                'pos_value': [],
                'abs_value': [],
            }

        n = 0
        while n < terminal_num:
            terminal = Terminal(df_source=df_source,
                                terminal_pbs=terminal_pbs, classifier_map=classifier_map, classifier_group=classifier_group)
            terminal.create_terminal()
            Forest.add_node_to_box(node_box, terminal)
            n += 1

            if n % 10 == 0:
                if n % 200 == 0 or n == terminal_num:
                    end = '\n'
                else:
                    end = ','
                print(n, end=end)

        print('Finish: %s leaves generated.' % n)

        return node_box

    @staticmethod
    def add_first_level_branchs(branch_num, *, df_source, node_box,
                                terminal_pbs=None, classifier_map=None, classifier_group=None,
                                intergrator_map=None, primitive_pbs=None, child_num_range=None):
        """添加首层branch，子节点为terminal。"""

        print('\nGenerate level 1 branches. counting:')

        if not node_box['pos_value'] or not node_box['abs_value']:
            print('node_box empty, generate more leaves.')
            node_box = Forest.generate_leaves(2, df_source=df_source, node_box=node_box,
                                       terminal_pbs=terminal_pbs, classifier_map=classifier_map,
                                       classifier_group=classifier_group)
            node_box = Forest.add_first_level_branchs(branch_num, df_source=df_source, node_box=node_box,
                                               terminal_pbs=terminal_pbs, classifier_map=classifier_map,
                                               classifier_group=classifier_group,
                                               intergrator_map=intergrator_map, primitive_pbs=primitive_pbs,
                                               child_num_range=child_num_range)
            return node_box

        lv1_node_box = {
            'all': [],
            'pos_value': [],
            'abs_value': [],
        }

        n = 0
        while n < branch_num:
            primitive = Primitive(node_box=node_box)
            primitive.create_primitive()
            primitive.cal()
            Forest.add_node_to_box(lv1_node_box, primitive)
            n += 1

            if n % 10 == 0:
                if n % 200 == 0 or n == branch_num:
                    end = '\n'
                else:
                    end = ','
                print(n, end=end)

        print('Finish: %s level 1 branches generated.' % n)

        node_box = Forest.merge_note_box(node_box, lv1_node_box)
        return node_box

    @staticmethod
    def add_second_level_branch(branch_num, *, node_box,
                                intergrator_map=None, primitive_pbs=None, child_num_range=None):
        """添加第二代branch，子节点为terminal或首层branch"""

        print('\nGenerate level 2 branches. counting:')

        lv2_node_box = {
            'all': [],
            'pos_value': [],
            'abs_value': [],
        }

        n = 0
        while n < branch_num:
            primitive = Primitive(node_box=node_box,
                                  intergrator_map=intergrator_map, primitive_pbs=primitive_pbs,
                                  child_num_range=child_num_range)
            primitive.create_primitive()
            primitive.cal()
            Forest.add_node_to_box(lv2_node_box, primitive)
            n += 1

            if n % 10 == 0:
                if n % 200 == 0 or n == branch_num:
                    end = '\n'
                else:
                    end = ','
                print(n, end=end)

        print('Finish: %s level 2 branches generated.' % n)

        node_box = Forest.merge_note_box(node_box, lv2_node_box)
        return node_box

    @staticmethod
    def add_limit_depth_branch(branch_num, *, node_box, depth_limit=3,
                               intergrator_map=None, primitive_pbs=None, child_num_range=None):
        """随机添加branch，限定层级深度（depth）。"""

        print('\nGenerate limit depth(%d) branches. counting:' % depth_limit)

        n = 0
        while n < branch_num:

            primitive = Primitive(node_box=node_box,
                                  intergrator_map=intergrator_map, primitive_pbs=primitive_pbs,
                                  child_num_range=child_num_range)
            primitive.create_primitive()
            if primitive.depth > depth_limit:
                continue

            primitive.cal()
            Forest.add_node_to_box(node_box, primitive)  # 生成10,000个节点大约需要3分钟
            n += 1

            if n % 10 == 0:
                if n % 200 == 0 or n == branch_num:
                    end = '\n'
                else:
                    end = ','
                print(n, end=end)

        print('Finish: %s limit depth(%d) branches generated.' % (n, depth_limit))

        return node_box

    @staticmethod
    def add_random_branchs(branch_num, *, node_box,
                           intergrator_map=None, primitive_pbs=None, child_num_range=None):
        """随机添加branch，不做层数限制"""

        print('\nGenerate branches: random branches. counting:')

        n = 0
        while n < branch_num:

            primitive = Primitive(node_box=node_box,
                                  intergrator_map=intergrator_map, primitive_pbs=primitive_pbs,
                                  child_num_range=child_num_range)
            primitive.create_primitive()
            primitive.cal()
            Forest.add_node_to_box(node_box, primitive)  # 生成10,000个节点大约需要3分钟
            n += 1

            if n % 10 == 0:
                if n % 200 == 0 or n == branch_num:
                    end = '\n'
                else:
                    end = ','
                print(n, end=end)

        print('Finish: %s random branches generated.' % n)

        return node_box
    
    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rebuild_model(model_path, df_source):

        model = np.load(model_path, allow_pickle=True)
        model = model.item()
        tree_map = model['tree_map']

        # step1: recal features

        for key, node_data in tree_map.items():

            if node_data['node_class'] == Terminal:
                func_group = node_data['class_func_group']

                if func_group == 'cut' or func_group == 'compare':

                    # rebuild terminal --dict
                    for kw in node_data['class_kw'].keys():
                        if kw[:7] == 'feature':

                            instance_class = node_data['class_kw_ins_class'][kw]
                            kwargs = node_data['class_kw_ins_kwargs'][kw]
                            instance = instance_class(df_source=df_source, kwargs=kwargs)
                            node_data['class_kw_ins'][kw] = instance
                            node_data['class_kw'][kw] = instance.cal()

                elif func_group == 'permutation' or func_group == 'trend':

                    # rebuild terminal --list
                    node_data['class_args_features_ins'] = []
                    node_data['class_args_features'] = []

                    for instance_class, kwargs in zip(node_data['class_args_features_ins_class'],
                                                      node_data['class_args_features_ins_kwargs']):

                        instance = instance_class(df_source=df_source, kwargs=kwargs)
                        node_data['class_args_features_ins'].append(instance)
                        node_data['class_args_features'].append(instance.cal())

                else:
                    raise ValueError('Uncategorized func_group: %s. 4674' % func_group)

        # step2: rebuild nodes

        waiting_room = {}  
        for num in range(model['depth']):
            waiting_room[num + 1] = []  # +1 --对应name长度

        tree_instance_map = {}
        for key in tree_map.keys():
            tree_instance_map[key] = None

        for node_tag, node_data in tree_map.items():
            room_num = len(node_tag)
            waiting_room[room_num].append([node_tag, node_data])

        room_num = model['depth']
        while room_num >= 1:

            room = waiting_room[room_num]
            for node_zip in room:
                node_tag, node_data = node_zip[0], node_zip[1]

                if node_data['terminal']:

                    if node_data['node_class'] is not Terminal:
                        raise TypeError('Node class error: terminal node not Terminal.')
                    terminal = Terminal(df_source=df_source, node_data=node_data)
                    terminal.recal()
                    tree_instance_map[node_tag] = terminal

                else:

                    child_tags = Tools.get_child_name_list(node_tag, tree_map)
                    instance_list = []
                    for tag in child_tags:
                        instance_list.append(tree_instance_map[tag])
                    node_data['method_ins'] = instance_list.copy()

                    if node_data['node_class'] is not Primitive:
                        raise TypeError('Node class error: non-terminal node not Primitive.')
                    primitive = Primitive(node_data=node_data)
                    primitive.update_branch_info()
                    primitive.recal()
                    tree_instance_map[node_tag] = primitive

            room_num -= 1

        # step3: rebuild tree

        deputy_tags = Tools.get_child_name_list('', tree_map)
        deputy_list = []
        for tag in deputy_tags:
            deputy_list.append(tree_instance_map[tag])

        tree_node_data = dict(
            tree_method=model['tree_method'],
            deputy_list=deputy_list
            )

        tree = Tree(node_data=tree_node_data)
        tree.cal()
        
        return tree

    # ----------------------------------------------------------------------------------------------------------------------
    def generate_trees(self):
        """生成用于迭代的tree： 将node_box数据深拷贝至tree实例中"""

        # TODO
        pass

    def judge_trees(self):
        # TODO
        pass

    # ----------------------------------------------------------------------------------------------------------------------
    def evolution_go(tree_box):
        
        # TODO

        pass




if __name__ == '__main__':
    pd.set_option('display.max_rows', 8)
    # df = pd.read_csv('data/bitmex_price_1hour_2020q1.csv')
    df = pd.read_csv('private/test_data(2018-2019).csv')
    df['timestamp'] = pd.to_datetime(df.timestamp)
    df.set_index('timestamp', inplace=True)

    memory_before = Tools.memory_free()
    time_before = time.time()

    node_box = Forest.generate_leaves(300, df_source=df)  # inplace node_box
    node_box = Forest.add_first_level_branchs(200, df_source=df, node_box=node_box)  # not inplace node_box
    node_box = Forest.add_second_level_branch(200, node_box=node_box)  # not inplace node_box
    node_box = Forest.add_limit_depth_branch(200, node_box=node_box)  # inplace node_box
    node_box = Forest.add_random_branchs(100, node_box=node_box)  # inplace node_box

    memory_after = Tools.memory_free()
    print('memory consumed: %.6f GB' % (memory_before - memory_after))
    print('time consumed: %.3f seconds.' % (time.time() - time_before))

    # debug
    num = 0
    while num < 100:
        instance = random.choice(node_box['pos_value'])
        while isinstance(instance, Terminal):  # or instance.node_data['inter_method'] != Tools.sig_trend_strict:
            instance = random.choice(node_box['pos_value'])

        instance = instance.copy()
        result = instance.mutate_primitive(node_box=node_box)

        n = 0
        while n < 100:
            result = instance.mutate_primitive()
            while not result:
                result = instance.mutate_primitive()

            node_result_new = instance.cal()

            n += 1

        num += 1
        print(num, '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    print('finish.')

    memory_after = Tools.memory_free()
    print('memory consumed: %.6f GB' % (memory_before - memory_after))
    print('time consumed: %.3f seconds.' % (time.time() - time_before))

    print('\nend ----------------------------------------------------------------------------------------', end='\n\n')
