# --coding:utf-8--

import time
import random
import numpy as np
import pandas as pd

from tools import Tools


class Hello(object):

    name = 'test indicator'

    def __init__(self, name=name):
        self.name = name

    def show(self, tag='test', *args, param=None, **kwargs):
        print('%s | %s, args: %s, param: %s, %s' % (tag, self.name, args, param, kwargs))


# ======================================================================================================================

class Indicator(Tools):

    name = 'Indicator'

    default_range = dict(df_source=None,
                         column_name=['price_end'],  # 字符类型：list表示，提供参数的变异选项
                         window={'start': 0, 'end': 0, 'sep': 0}  # 数值类型：dict表示，参数的变异起终点、步长 | auto sep --  sep:True
                         )

    score_list = []
    score_num = 0
    avg_score = 0

    def __init__(self, *, df_source, name=None, arg_range=None, refeature_pb=0.07):

        self.df_source = df_source  # data_source. must have.

        if name:
            self.name = name
        else:
            self.name = self.get_id(self.name)  # feature_ID

        if not arg_range:
            arg_range = self.default_range
        self.arg_range = arg_range  # 自动变异的范围

        self.score_list = []  # 得分相关记录
        self.score_num = 0
        self.avg_score = 0

        self.kwargs = self.__generate_random_args()  # 主函数除df外所需全部数据

        self.refeature_pb = refeature_pb
        self.pb_each = self.__calculate_pb_each(refeature_pb)  # 变异概率

    def __generate_random_args(self):
        """随机生成主函数的一组参数  --除了df，都放到kwargs里面来"""

        kwargs = {}
        for key, value in self.arg_range.items():
            if value:
                arg = None
                if isinstance(value, list):
                    arg = random.choice(value)
                if isinstance(value, dict):
                    start = value['start']
                    sep = value['sep']
                    end = value['end']
                    if sep == 0:
                        arg = start
                    else:
                        start = value['start']
                        if sep is True:
                            sep = None
                        else:
                            end += sep  # to include the end data, if possible.
                        arg = np.random.choice(np.arange(start, end, sep))
                kwargs[key] = arg

        return kwargs

    def __calculate_pb_each(self, refeature_pb, inplace=True):
        """计算每个参数的变异概率"""

        mut_arg_num = 0

        for value in self.arg_range.values():
            if isinstance(value, dict) and value['sep']:
                mut_arg_num += 1
            if isinstance(value, list) and len(value) > 1:
                mut_arg_num += 1
        
        if mut_arg_num == 0:
            pb_each = 0
        else:
            pb_each = self.probability_each(object_num=mut_arg_num, pb_for_all=refeature_pb)

        if inplace:
            self.refeature_pb = refeature_pb
            self.pb_each = pb_each  # 全局更新实例的变异概率

        return pb_each

    # ------------------------------------------------------------------------------------------------------------------

    def get_name(self):
        return self.name

    def get_range(self):
        return self.arg_range  # 获取变异区间信息

    def get_current_args(self):
        return self.kwargs  # 获取主函数参数

    def get_avg_score(self):
        return self.avg_score  # 获取分数统计信息

    # ------------------------------------------------------------------------------------------------------------------

    def random_start(self):
        """用于完成突变，并返还计算结果"""

        kwargs = self.__generate_random_args()  # 初始化时已经随机选取了
        result = self.cal(**kwargs)

        return result

    def add_score(self, score):
        """Add all scores, requires big memory."""

        self.score_list.append(score)
        Indicator.score_list.append(score)

    def update_score(self, sortino_score):
        """calculate average sortino_ratio"""

        self.avg_score = (self.avg_score * self.score_num + sortino_score) / (self.score_num + 1)
        Indicator.avg_score = (Indicator.avg_score * Indicator.score_num + sortino_score) / (Indicator.score_num + 1)

        self.score_num += 1
        Indicator.score_num += 1

    def mutate_args(self, mul=3.0, refeature_pb=None, update=True):
        """渐变变异函数"""

        mut_flag = False

        if refeature_pb is not None:
            pb_each = self.__calculate_pb_each(refeature_pb, inplace=update)
        else:
            pb_each = self.pb_each

        for key, value in self.arg_range.items():
            new_value = None

            if isinstance(value, list) and len(value) > 1:
                if random.random() < pb_each:
                    choice_box = value.copy()
                    # now_value = self.kwargs[key]
                    # choice_box.remove(now_value)  # 这里的意思是，执行无放回的随机抽样。暂不用。
                    new_value = random.choice(choice_box)

            if isinstance(value, dict) and value['sep'] != 0:
                if random.random() < pb_each:
                    start = value['start']
                    sep = value['sep']
                    end = value['end']
                    if sep is True:
                        sep = None
                    else:
                        end += sep
                    now_value = self.kwargs[key]

                    # NOTE: don't know why but this always return a float for the child class:
                    new_value = self.mutate_value(now_value, mul=mul,
                                                  start_value=start, end_value=end, sep=sep)

            if new_value is not None:
                self.kwargs[key] = new_value
                mut_flag = True

        return mut_flag

    def cal(self, **kwargs):
        """rewrite this in subclass """

        if not kwargs:
            kwargs = self.kwargs
        else:
            self.kwargs = kwargs  # look to data from outside

        # main calculation starts here -------------------------------------------------------
        result = 'to be done.'

        return result


# ======================================================================================================================

class MA(Indicator):

    name = 'simple moving average'
    default_range = dict(df_source=None,
                         column_name=['price_end', 'price_start'],
                         window={'start': 5, 'end': 400, 'sep': True}
                         )
    score_list = []
    score_num = 0
    avg_score = 0

    def add_score(self, score):

        super(MA, self).add_score(score)
        MA.score_list.append(score)

    def update_score(self, sortino_score):

        super(MA, self).update_score(sortino_score)

        MA.avg_score = (MA.avg_score * MA.score_num + sortino_score) / (MA.score_num + 1)
        MA.score_num += 1

    def cal(self, **kwargs):

        if not kwargs:
            kwargs = self.kwargs
        else:
            self.kwargs = kwargs

        # main calculation starts here ---------------------------------------------------------

        column_name = kwargs['column_name']
        window = kwargs['window']
        result = self.df_source[column_name].rolling(window).mean()

        return result

# ======================================================================================================================

class WEMA(Indicator):

    name = 'exponential weighted moving average'  # change indicator name
    default_range = dict(df_source=None,
                         column_name=['price_end', 'price_start'],
                         com={'start': 1.0, 'end': 400.0, 'sep': True}  # pd.ewm的com非常适合True自动变异
                         )  # change default argument range
    score_list = []
    score_num = 0
    avg_score = 0

    def add_score(self, score):

        super(WEMA, self).add_score(score)  # change subclass_name
        WEMA.score_list.append(score)  # change subclass_name

    def update_score(self, sortino_score):

        super(WEMA, self).update_score(sortino_score)  # change subclass_name

        WEMA.avg_score = (WEMA.avg_score * WEMA.score_num + sortino_score) / (WEMA.score_num + 1)  # change subclass_name
        WEMA.score_num += 1  # change subclass_name

    def cal(self, kind='com', **kwargs):

        if not kwargs:
            kwargs = self.kwargs
        else:
            self.kwargs = kwargs

        # main calculation starts here ---------------------------------------------------------

        column_name = kwargs['column_name']
        com = kwargs['com']

        result = self.df_source[column_name].ewm(com=com).mean()

        return result

if __name__ == '__main__':

    # p1 = Hello('AAA indicator')
    # p1.show('aeaee', 'bibibi', 'cicici', param=654654, additional1='dididid', add2='eieiei')

    # indicator_class_test = Indicator(df_source=None)
    # i = indicator_class_test
    # print(i)
    # print(i.get_range())
    # print(i.get_current_args())
    # print(i.random_start())
    #
    # print('-' * 30)
    #


    # 真实的使用方式：

    df = pd.read_csv('data/bitmex_price_1hour_2020q1.csv') 

    test = MA(df_source=df, refeature_pb=0.5)
    test2 = WEMA(df_source=df, refeature_pb=0.5)
    test3 = WEMA(df_source=df, refeature_pb=0.5)

    print(test.name)
    print(test2.name)
    print(test3.name)

    test.random_start()
    test2.random_start()
    test3.random_start()

    n = 0
    while n < 10:

        print(n, '------------------------')
        print('Indic', '\t',  Indicator.score_num, Indicator.avg_score, Indicator.score_list)
        print('MA  ', '\t',  MA.score_num, MA.avg_score, MA.score_list)
        print('test', '\t',  test.score_num, test.avg_score, test.score_list)
        print('WEMA', '\t',  WEMA.score_num, WEMA.avg_score, WEMA.score_list)
        print('test2', '\t',  test2.score_num, test2.avg_score, test2.score_list)
        print('test3', '\t',  test3.score_num, test3.avg_score, test3.score_list)
        print(test3.get_current_args(), test3.get_range())

        score1 = round(random.random() * 100)
        score2 = round(random.random() * 100)
        score3 = round(random.random() * 100)

        test.add_score(score1)
        test2.add_score(score2)
        test3.add_score(score3)

        test.update_score(score1)
        test2.update_score(score2)
        test3.update_score(score3)

        change_flag = test3.mutate_args(refeature_pb=0.8)
        if change_flag:
            result = test3.cal()

        n += 1
        time.sleep(0.2)

    print('-' * 30)

    # print('-' * 30)

    # print(test)
    # print(test.get_range())
    # print(test.get_current_args())

    # print('-' * 30)
    #
    # print(Indicator.name)
    # print('Indicator scores: ', Indicator.score_list)
    # print('MA scores:', MA.score_list)
    # print('instance scores:', test.score_list)
    #
    # print(test.get_range())
    # print(test.mutate_args(refeature_pb=1))
    # print(test.get_current_args())
    # print('-' * 30)
    # print(test.get_range())
    # print(test.mutate_args(refeature_pb=0.5, update=True))
    # print(test.get_current_args())
    # print('-' * 30)
    # print(test.get_range())
    # print(test.mutate_args(refeature_pb=1, update=True))
    # print(test.get_current_args())
    # print('-' * 30)
    # print(test.pb_each)
    #
    # print(test.name)

    # print('-' * 30)

    # result = test.cal()
    # print(result)