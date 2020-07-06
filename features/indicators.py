# --coding:utf-8--

import time
import random
import numpy as np
import pandas as pd

from tools import Tools

from features.indicator import Indicator


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

    data = pd.Series()  # indicator 计算出来的data

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

        # main calculation finish --------------------------------------------------------------

        self.result = result
        return result


# ======================================================================================================================

class WEMA(Indicator):
    name = 'exponential weighted moving average'
    default_range = dict(df_source=None,
                         column_name=['price_end', 'price_start'],
                         com={'start': 1.0, 'end': 400.0, 'sep': True}  # pd.ewm的com非常适合True自动变异
                         )
    score_list = []
    score_num = 0
    avg_score = 0

    data = pd.Series()

    def add_score(self, score):

        super(WEMA, self).add_score(score)
        WEMA.score_list.append(score)

    def update_score(self, sortino_score):

        super(WEMA, self).update_score(sortino_score)

        WEMA.avg_score = (WEMA.avg_score * WEMA.score_num + sortino_score) / (WEMA.score_num + 1)
        WEMA.score_num += 1

    def cal(self, **kwargs):

        if not kwargs:
            kwargs = self.kwargs
        else:
            self.kwargs = kwargs

        # main calculation starts here ---------------------------------------------------------

        column_name = kwargs['column_name']
        com = kwargs['com']

        result = self.df_source[column_name].ewm(com=com).mean()

        # main calculation finish --------------------------------------------------------------

        self.result = result
        return result


# ======================================================================================================================

class MovingSTD(Indicator):

    name = 'moving standard deviation'

    default_range = dict(df_source=None,
                         column_name=['price_end', 'price_avg'],
                         window={'start': 5, 'end': 400, 'sep': True}
                         )
    score_list = []
    score_num = 0
    avg_score = 0

    data = pd.Series()

    def add_score(self, score):

        super(MovingSTD, self).add_score(score)
        MovingSTD.score_list.append(score)

    def update_score(self, sortino_score):

        super(MovingSTD, self).update_score(sortino_score)

        MovingSTD.avg_score = (MovingSTD.avg_score * MovingSTD.score_num + sortino_score) / (MovingSTD.score_num + 1)
        MovingSTD.score_num += 1

    def cal(self, kind='com', **kwargs):

        if not kwargs:
            kwargs = self.kwargs
        else:
            self.kwargs = kwargs

        # main calculation starts here ---------------------------------------------------------

        column_name = kwargs['column_name']
        window = kwargs['window']

        result = self.df_source[column_name].rolling(window).std()

        # main calculation finish --------------------------------------------------------------

        self.result = result
        return result





# ======================================================================================================================
# feature_lv1的，直接扩展基础数据，没有什么变异空间的，不用放到这里

# class ATR(Indicator):

#     name = 'average true range'

#     default_range = dict(df_source:None,
#         price_start=['price_start'],
#         price_end=['price_end'],
#         price_max=['price_max'],
#         price_min=['price_min'],
#         window={'start': 1, 'end': 400, 'sep': True}  # if >1: moving average ATR
#         )

#     score_list = []
#     score_num = 0
#     avg_score = 0

#     data = pd.Series()

#     def add_score(self, score):

#         supper(ATR, self).add_score(score)
#         ATR.score_list.append(score)

#     def update_score(self, sortino_score):

#         super(ATR, self).update_score(sortino_score)

#         ATR.avg_score = (ATR.avg_score * ATR.score_num + sortino_score) / (ATR.score_num + 1)
#         ATR.score_num += 1

#     def cal(self, **kwargs):

#         if not kwargs:
#             kwargs = self.kwargs
#         else:
#             self.kwargs = kwargs

#         # main calculation starts here ---------------------------------------------------------

#         # TODO test it.

#         df = self.df_source.copy()

#         arr_prices = np.array([
#             df['price_start'],  # 因为没有变异的空间，所以这里就不用 df_source[kwargs['price_start']] 这样复杂的表达了
#             df['price_end'],
#             df['price_max'],
#             df['price_min']
#         ])  # for np is faster than pd

#         atr_list = []
#         before_end = arr_prices[0, 0]

#         num = 0
#         max_line = arr_prices.shape[1]
#         while num < max_line:

#             data = arr_prices[..., num]
#             now_start = data[0]
#             now_end = data[1]
#             now_max = data[2]
#             now_min = data[3]

#             atr_value = cal_atr(before_end, now_start, now_max, now_min)  # atr value of each line
#             atr_list.append(atr_value)

#             before_end = now_end
#             num += 1

#         df['atr_each'] = atr_list

#         window = kwargs['window']
#         result = df['atr_each'].rolling(window).mean()

#         # main calculation finish --------------------------------------------------------------

#         self.result = result  # update result data
#         return result


if __name__ == '__main__':

    # 真实的使用方式：

    df = pd.read_csv('../data/bitmex_price_1hour_2020q1.csv')

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
        print('Indic', '\t', Indicator.score_num, Indicator.avg_score, Indicator.score_list)
        print('MA  ', '\t', MA.score_num, MA.avg_score, MA.score_list)
        print('test', '\t', test.score_num, test.avg_score, test.score_list)
        print('WEMA', '\t', WEMA.score_num, WEMA.avg_score, WEMA.score_list)
        print('test2', '\t', test2.score_num, test2.avg_score, test2.score_list)
        print('test3', '\t', test3.score_num, test3.avg_score, test3.score_list)
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

    print(test3.data)

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
