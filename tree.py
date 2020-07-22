# --coding:utf-8--
import random
import pandas as pd

from tools import Tools
from leaf import Terminal
from branch import Primitive
from constant import *
from backtest.swap_engine import Judge
from features.indicators import *


class Tree(Tools):
    name = 'Tree'

    def __init__(self, *, node_box={}, tree_method=None, deputy_range=None, tree_range=None, tag_list=None,
                 node_data=None):

        self.name = Tools.get_id('tree')
        self.now_score = 0
        self.score_list = []
        self.score_num = 0
        self.avg_score = 0

        self.pos_should = pd.Series()
        self.node_map = {}

        self.depth = 0  # 下方最深有多少层node。terminal: 0
        self.width = 0  # 下方第一层有多少node。terminal: 0
        self.population = 0  # 总共有多少node（包括自身）。terminal: 1

        self.node_box = node_box.copy()

        if not tree_method:
            tree_method = Tools.comb_sum

        if deputy_range:
            self.deputy_range = deputy_range
        else:
            self.deputy_range = {'start': 3, 'end': 11}

        if tree_range:
            self.tree_range = tree_range
        else:
            self.tree_range = {'depth': 7, 'population': 256}

        if not tag_list:
            self.tag_list = NODE_TAG_LIST.copy()
        else:
            self.tag_list = tag_list.copy()

        if node_data:
            self.node_data = node_data.copy()
            self.__update_tree_map()
            self.__update_tree_info()
        else:
            self.node_data = dict(tree_method=tree_method,
                                  deputy_list=[],
                                  tree_map={}
                                  )

        self.generate_num = 0

    def __update_tree_info(self):

        self.width = len(self.node_data['deputy_list'])

        self.population = 0  # exclude tree
        self.depth = 0
        for instance in self.node_data['deputy_list']:
            self.population += instance.population
            self.depth = max(self.depth, instance.depth)
        self.depth += 1

    def __update_tree_map(self):

        node_map = {}

        tag = iter(self.tag_list)

        for deputy in self.node_data['deputy_list']:

            deputy_tag = tag.__next__()
            node_map[deputy_tag] = deputy

            if isinstance(deputy, Primitive):
                deputy_map = Tree.get_node_map(deputy, mother_tag=deputy_tag, tag_list=self.tag_list)
                node_map.update(deputy_map)

        self.node_data['tree_map'] = node_map.copy()

    def __deputy_update(self, node_tag):
        """inplace, update node_result for the line"""

        if len(node_tag) == 0:
            return  # deputy has no mother

        instance = self.node_data['tree_map'][node_tag]
        instance.update_branch_info()
        instance.recal()

        mother_node_tag = node_tag[:-1]
        self.__deputy_update(mother_node_tag)

    def __over_populated(self, new_node, *, my_node=None):

        if not self.tree_range['population']:
            return False

        if my_node:
            line_population = self.population - my_node.population
        else:
            line_population = 0
        population_delta = new_node.population

        if line_population + population_delta > self.tree_range['population']:
            return True
        else:
            return False

    def __over_deep(self, new_node, *, my_tag=None):

        if not self.tree_range['depth']:
            return False

        if my_tag:
            line_depth = len(my_tag)
        else:
            line_depth = 1
        depth_delta = new_node.depth

        if line_depth + depth_delta > self.tree_range['depth']:
            return True
        else:
            return False

    @staticmethod
    def __strip_type(value):

        if isinstance(value, Indicator):
            return value.__class__  # 父类继承自Indicator
        if isinstance(value, Primitive):
            return Primitive
        if isinstance(value, Terminal):
            return Terminal
        if isinstance(value, pd.Series):
            return pd.Series
        if isinstance(value, pd.DataFrame):
            return pd.DataFrame

        return False

    @staticmethod
    def __strip_node_data(dict_data):
        
        data = dict_data.copy()

        for key, value in data.items():

            value_type = Tree.__strip_type(value)
            if value_type:
                data[key] = value_type

            if isinstance(value, list):
                new_value_list = []
                for value2 in value:
                    value_type = Tree.__strip_type(value2)
                    if value_type:
                        new_value_list.append(value_type)
                    else:
                        if isinstance(value2, dict):
                            dict_in_list = Tree.__strip_node_data(value2)
                            new_value_list.append(dict_in_list)
                        else:
                            new_value_list.append(value2)

                data[key] = new_value_list

            if isinstance(value, dict):
                data[key] = Tree.__strip_node_data(value)

        return data


    @staticmethod
    def get_node_map(node, *, mother_tag=None, tag_list=None):
        """applicable for Primitives."""

        node_map = {}

        if not tag_list:
            tag_list = NODE_TAG_LIST

        now_tag = iter(tag_list)

        if isinstance(node, Primitive):
            for instance in node.node_data['method_ins']:

                if mother_tag:
                    instance_tag = mother_tag + now_tag.__next__()
                else:
                    instance_tag = now_tag.__next__()
                node_map[instance_tag] = instance

                if isinstance(instance, Primitive):
                    child_node_map = Tree.get_node_map(instance, mother_tag=instance_tag, tag_list=tag_list)
                    node_map.update(child_node_map)

        return node_map

    # ---------------------------------------------------------------------------------------------------------------

    def generate_tree(self):

        start = self.deputy_range['start']
        end = start + 2  # 一开始，先少一点比较好。后期再用 self.deputy_range['end']

        deputy_num = random.randint(start, end)
        deputy_list = random.sample(self.node_box['pos_value'], deputy_num)

        for instance in deputy_list:
            self.node_data['deputy_list'].append(instance.copy())

        self.__update_tree_map()
        self.__update_tree_info()

        max_depth = self.tree_range['depth']
        if not max_depth:
            max_depth = np.inf
        max_pop = self.tree_range['population']
        if not max_pop:
            max_pop = np.inf

        if self.depth > max_depth or self.population > max_pop:
            self.generate_num += 1
            if self.generate_num > 100:
                raise RuntimeError('ERROR: tree_range too small for node_box given. ', self)
            self.generate_tree()

    def get_one_node(self, node_type):

        tree_map = self.node_data['tree_map']
        choice_box = list(tree_map.keys())

        node_tag = random.choice(choice_box)
        node_instance = tree_map[node_tag]

        if node_type:

            while node_instance.node_data['node_type'] != node_type:
                node_tag = random.choice(choice_box)
                node_instance = tree_map[node_tag]

        return node_tag, node_instance

    def get_args(self):
        return self.node_data

    def get_model(self):

        model = self.node_data.copy()
        model.pop('deputy_list')
        model['avg_score'] = self.avg_score
        model['depth'] = self.depth
        model['population'] = self.population

        tree_map = model['tree_map'].copy()
        for key, value in tree_map.items():

            tree_map[key] = dict(
                node_class=model['tree_map'][key].__class__,
                avg_score=model['tree_map'][key].avg_score, 
                node_ID=model['tree_map'][key].name,
            )

            node_data = Tree.__strip_node_data(model['tree_map'][key].node_data)
            tree_map[key].update(node_data)

        model['tree_map'] = tree_map

        return model

    def cross_node(self, other_tag, other_node, *, my_tag=None, my_node=None):

        if not my_tag or not my_node:

            node_type = other_node.node_data['node_type']
            my_tag, my_node = self.get_one_node(node_type)

            # tree_range 限制  --如果指定了 my_tag, my_node，将不做限制（可能超出限制）
            num = 0
            while self.__over_deep(other_node, my_tag=my_tag) or self.__over_populated(other_node, my_node=my_node):
                my_tag, my_node = self.get_one_node(node_type)
                num += 1
                if num > 50:
                    # raise RuntimeError('ERROR: input other_node too big to fix into current tree. ', self)
                    print('ERROR: input other_node too big to fix into current tree. ', self)
                    break

        change_map = {my_node: other_node}
        mother_node_tag = my_tag[:-1]

        print('%s cross node: {%s: %s}, %s' % (self, my_tag, other_tag, change_map))

        if not mother_node_tag:  # my_node in self.node_data['deputy_list']

            instance_list = self.node_data['deputy_list']
            new_instance_list = [change_map[i] if i in change_map else i for i in instance_list]
            self.node_data['deputy_list'] = new_instance_list.copy()

        else:

            mother_node = self.node_data['tree_map'][mother_node_tag]
            instance_list = mother_node.node_data['method_ins']
            new_instance_list = [change_map[i] if i in change_map else i for i in instance_list]
            mother_node.node_data['method_ins'] = new_instance_list.copy()

            # if len(other_tag) == 1:  # 如cross不对等（deputy vs. branch)， 针对deputy进行添加与淘汰

        if len(other_tag) == 1:  # 针对deputy进行添加与淘汰 （忽略上方的对等问题）

            # 在 deputy level 添加交换来的 node
            print('deputy updating for %s' % self)
            new_deputy = other_node.copy()
            self.node_data['deputy_list'].append(new_deputy)

            if len(self.node_data['deputy_list']) > self.deputy_range['end']:  # 如deputy数量太多，淘汰分数低的

                instance_list = self.node_data['deputy_list']

                score_list = []
                for deputy in instance_list:
                    score = deputy.avg_score
                    score_list.append(score)

                min_score = min(score_list)
                min_score_num = score_list.count(min_score)

                if min_score_num == 1:
                    pop_index = score_list.index(min_score)

                else:  # 如最低分数相同，淘汰权重低的

                    weight_list = []
                    weight_dict = {}
                    weight_index = 0
                    for index, score in zip(range(len(score_list)), score_list):

                        if score == min_score:
                            weight_dict[weight_index] = index
                            weight = instance_list[index].node_data['weight']
                            weight_list.append(weight)
                            weight_index += 1

                    min_weight = min(weight_list)
                    min_weight_num = weight_list.count(min_weight)

                    if min_weight_num == 1:
                        pop_index_weight = weight_list.index(min_weight)
                        pop_index = weight_dict[pop_index_weight]

                    else:  # 如最低权重相同，随机淘汰
                        pop_index_weight = random.choice(range(len(weight_list)))
                        pop_index = weight_dict[pop_index_weight]

                self.node_data['deputy_list'].pop(pop_index)

        # update data
        self.__update_tree_map()
        self.__deputy_update(mother_node_tag)
        self.__update_tree_info()

        return my_tag, my_node

    def mutate_tree(self, *, node_box=None, update_node_box=True):

        mutation_tag = False
        node_tag, node_instance = self.get_one_node(node_type=None)  # 选定一个节点进行变异
        mother_node_tag = node_tag[:-1]

        if isinstance(node_instance, Primitive):
            if self.population < self.tree_range['population']:
                tree_addable = True
            else:
                tree_addable = False
            mutation_tag = node_instance.mutate_primitive(node_box=node_box, update_node_box=update_node_box,
                                                          tree_addable=tree_addable)

        elif isinstance(node_instance, Terminal):
            mutation_tag = node_instance.mutate_terminal()

        else:
            raise ValueError('ERROR: undefined node type. 49146')

        if mutation_tag:
            node_instance.cal()  # 更新自己的 node_result，分层根据需要计算

            self.__update_tree_map()
            self.__deputy_update(mother_node_tag)
            self.__update_tree_info()

        return mutation_tag

    def cal(self):

        args = []
        func = self.node_data['tree_method']

        for instance in self.node_data['deputy_list']:
            args.append(instance.node_result)

        self.pos_should = func(*args)

        return self.pos_should

    def judge_tree(self, df_source,
                   judge_type='kbar', save_file=None, adjust='auto'):

        pos_should = self.cal()

        df_book = pd.concat([pos_should, df_source['price_end']], axis=1)
        df_book.columns = ['pos_should', 'price']
        df_book['pos_should'] = df_book['pos_should'] * VOLUME_BASE
        df_book['order_price'] = 0

        # timestamp的处理 - 变为end  --好像不变也可以。。
        # df_book['time_end'] = df_book.index.copy()

        trade_instance = Judge(name=self.name, df_book=df_book, df_price=df_source,
                               judge_type=judge_type, save_file=save_file, adjust=adjust)
        trade_scores = trade_instance.judge_start()
        sortino_score = trade_scores['Sortino_ratio']

        self.update_score(sortino_score)

        return trade_scores

    def add_score(self, score):

        self.score_list.append(score)

        for instance in self.node_data['deputy_list']:
            instance.add_score(score)

    def update_score(self, sortino_score):

        self.avg_score = (self.avg_score * self.score_num + sortino_score) / (self.score_num + 1)
        self.score_num += 1

        for instance in self.node_data['deputy_list']:
            instance.update_score(sortino_score)
