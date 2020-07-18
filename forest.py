# --coding:utf-8--

import time
import pandas as pd

from tools import Tools
from leaf import Terminal
from branch import Primitive

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

# 存放node实例的分类的容器
node_box = {
    'all': [],
    'pos_value': [],
    'abs_value': [],
}


def add_node_to_box(node_box, node_instance):
    """根据 node 的数据类型（pos/abs）分类添加到 node_box
    NOTE:inplace.
    """

    node_type = node_instance.node_data['node_type']  # abs/pos
    node_box[node_type].append(node_instance)
    node_box['all'].append(node_instance)


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
        add_node_to_box(node_box, terminal)
        n += 1

        if n % 100 == 0:
            print(n, end=', ')
            if n % 2000 == 0:
                print('')

    print('\n%s leaves generated.' % n)

    return node_box


def add_first_level_branchs(branch_num, *, df_source, node_box):
    """"""


    print('\nGenerate first level branches. counting:')

    if not node_box['pos_value'] or not node_box['abs_value']:
        print('node_box empty, generate more leaves.')
        node_box = generate_leaves(2, df_source=df_source, node_box=node_box)
        node_box = add_first_level_branchs(branch_num, df_source=df_source, node_box=node_box)
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
        add_node_to_box(lv1_node_box, primitive)
        n += 1

        if n % 100 == 0:
            print(n, end=', ')
            if n % 2000 == 0:
                print('')

    print('\n%s first level branches generated.' % n)

    node_box = merge_note_box(node_box, lv1_node_box)
    return node_box


def add_random_branchs(branch_num, *, node_box):
    print('\nGenerate random branches. counting:')

    n = 0
    while n < branch_num:

        primitive = Primitive(node_box=node_box)
        primitive.create_primitive()
        primitive.cal()
        add_node_to_box(node_box, primitive)  # 生成10,000个节点大约需要3分钟
        n += 1

        if n % 100 == 0:
            print(n, end=', ')
            if n % 2000 == 0:
                print('')

    print('\n%s random branches generated.' % n)

    return node_box


def generate_tree(node_box):
    pass


def deep_copy(node):
    """深度复制节点内的所有instance"""
    pass


if __name__ == '__main__':
    pd.set_option('display.max_rows', 8)
    df = pd.read_csv('data/bitmex_price_1hour_2020q1.csv')
    df['timestamp'] = pd.to_datetime(df.timestamp)
    df.set_index('timestamp', inplace=True)

    node_box = generate_leaves(1000, df_source=df)  # inplace node_box

    node_box = add_first_level_branchs(500, df_source=df, node_box=node_box)  # inplace node_box

    node_box = add_random_branchs(10000, node_box=node_box)  # inplace node_box

    print('\nend ----------------------------------------------------------------------------------------', end='\n\n')
