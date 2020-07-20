#--coding:utf-8--
import random
import pandas as pd


from tools import Tools
from leaf import Terminal
from branch import Primitive
from constant import *
from backtest.swap_engine import Judge


class Tree(Tools):

    name = 'Tree'

    def __init__(self , *, node_box, tree_method=None, deputy_range=None, tree_range=None, tag_list=None, node_data=None):

        self.name = self.get_id('tree')
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

        if tree_range:  # TODO: 发挥作用
            self.tree_range = tree_range
        else:
            self.tree_range = {'depth': 7, 'population': 52}

        if not tag_list:
            self.tag_list = NODE_TAG_LIST.copy()
        else:
            self.tag_list = tag_list.copy()

        if node_data:
            self.node_data = node_data.copy()
        else:
            self.node_data = dict(tree_method=tree_method,
                                  deputy_list=[],
                                  tree_map={}
                                  )

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

    def __cal(self):

        args = []
        func = self.node_data['tree_method']

        for instance in self.node_data['deputy_list']:
            args.append(instance.node_result)

        pos_should = func(*args)

        return pos_should

    def __deputy_update(self, node_tag):
        """inplace, update node_result for the line"""

        if len(node_tag) == 0:
            return  # deputy has no mother

        instance = self.node_data['tree_map'][node_tag]
        instance.update_branch_info()
        instance.recal()

        mother_node_tag = node_tag[:-1]
        self.__deputy_update(mother_node_tag)

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

    def get_tree_data(self):
        return self.node_data

    def cross_node(self, other_tag, other_node, *, my_tag=None, my_node=None):
        
        if not my_tag or not my_node:
            node_type = other_node.node_data['node_type']
            my_tag, my_node = self.get_one_node(node_type)

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
            print('%s: deputy updating.' % self)
            new_deputy = other_node.copy()
            self.node_data['deputy_list'].append(new_deputy)
            
            # 如数量太多。。。淘汰分数低的
            if len(self.node_data['deputy_list']) > self.deputy_range['end']:

                min_score = np.inf
                pop_deputy = None
                for deputy in self.node_data['deputy_list']:
                    if deputy.avg_score < min_score:
                        min_score = deputy.avg_score
                        pop_deputy = deputy

                self.node_data['deputy_list'].remove(pop_deputy)

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
            mutation_tag = node_instance.mutate_primitive(node_box=node_box, update_node_box=update_node_box)

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

    def judge_tree(self, df_source):

        pos_should = self.__cal()

        df_book = pd.concat([pos_should, df_source['price_end']])
        df_book.columns=['pos_should', 'price']
        df_book['order_price'] = 0

        # timestamp的处理 - 变为end
        df_book['time_end'] = df_book.index.copy()
        # TODO


        trade_instance = Judge(name=instance.name, df_book=df_book, df_price=df_source)
        trade_scores = trade_instance.judge_start()
        sortino_score = trade_scores['Sortino_ratio']

        instance.update_score(sortino_score)

        return scores

    def add_score(self, score):

        self.score_list.append(score)

        for instance in self.node_data['deputy_list']:
            instance.add_score(score)

    def update_score(self, sortino_score):

        self.avg_score = (self.avg_score * self.score_num + sortino_score) / (self.score_num + 1)
        self.score_num += 1

        for instance in self.node_data['deputy_list']:
            instance.update_score(sortino_score)






