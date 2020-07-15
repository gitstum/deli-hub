# --coding:utf-8--

import random
import talib
import pandas as pd

from indicator import Indicator
from features.indicators import *
from leaf import Terminal

# g = talib.get_function_groups()
# for x in g:
#     print(x)


class Dog(object):
    name = 'dog'
    child = 'an instance'

    def __init__(self, data, child=None):
        self.data = data
        if child:
            self.child = child

    def show(self):
        print(self)

    def __del__(self):
        print('delete %s: %s' % (self.name, self))  # 当实例从内存中删除前，会自动执行这部分的内容。




if __name__ == '__main__':

    pd.set_option('display.max_rows', 8)

    # a = Dog(random.random())
    # ba = Dog(random.random(), a)
    # ca = Dog(random.random(), a)
    # dca = Dog(random.random(), ca)
    #
    # node_box = {'node': []}
    #
    # node_box['node'].append(a)
    # node_box['node'].append(ba)
    # node_box['node'].append(ca)
    # node_box['node'].append(dca)
    #
    # print(node_box)
    #
    # del a
    # del ba
    # del ca
    # del dca
    #
    # # del node_box  # 这能删除所有实例
    #
    # node_box['node'].pop(0)  # a  -- 无任何实例被清除
    # node_box['node'].pop(0)  # ba  -- 到这里删除了b的实例
    # node_box['node'].pop(0)  # ca  -- 无任何实例被清除
    # # node_box['node'].pop(0)  # dca中虽然没有保存a的信息，但它保存的ca中保存了a的信息，所以a也不会被内存删除
    #
    # print(node_box['node'][0])
    # print(node_box['node'][0].child)
    # print(node_box['node'][0].child.child)
    #
    # print('-' * 30)
    #
    # # instance 拷贝删除的影响
    #
    # test = MA(df_source=None)
    # test2 = test.copy()
    # print(test)
    # print(test2)
    # del test


    print('-' * 30)
    # Terminal debug

    df = pd.read_csv('private/test_data(2018-2019).csv')

    instance = Terminal(df_source=df)
    node_result = instance.create_terminal()
    node_result = instance.cal()
    print(node_result)

    print('end', '-' * 40)
