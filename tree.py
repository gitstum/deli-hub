#--coding:utf-8--

from tools import Tools
from leaf import Terminal
from branch import Primitive
from constant import *

from private.prameters import Board


class Tree(Tools):


	def __init__(self , *, node_box, tree_pbs=None, tag_list=None, node_data=None):

        self.name = self.get_id('tree')
        self.score_list = []
        self.score_num = 0
        self.avg_score = 0

		self.pos_should = pd.Series()
		self.node_map = {}

        self.depth = 0  # 下方最深有多少层node。terminal: 0
        self.width = 0  # 下方第一层有多少node。terminal: 0
        self.population = 0  # 总共有多少node（包括自身）。terminal: 1

		if not tree_pbs:
			self.tree_pbs = Board.tree_pbs.copy()
		else:
			self.tree_pbs = tree_pbs.copy()

		if not tag_list:
			self.tag_list = NODE_TAG_LIST.copy()
		else:
			self.tag_list = tag_list.copy()

		if not node_data:
			self.node_data = Board.node_data.copy()
		else:
			self.node_data = node_data.copy()
    
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
                    child_node_map = get_node_map(instance, mother_tag=instance_tag, tag_list=tag_list)
                    node_map.update(child_node_map)
                
        return node_map


    def __get_node_map(self):

    	node_map = {}

    	tag = iter(self.tag_list)

    	for deputy in self.node_data['deputy_list']:

    		deputy_tag = tag.__next__()
    		node_map[deputy_tag] = deputy

    		if isinstance(deputy, Primitive):
    			deputy_map = get_node_map(deputy, mother_tag=deputy_tag, tag_list=self.tag_list)
    			node_map.update(deputy_map)

		self.node_map = node_map.copy()









