# --coding:utf-8--

import time
import random
import numpy as np
import pandas as pd
import multiprocessing as mp

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
    result = pd.Series() 
    map_type = ['vector', 'condition', 'multiplier']

    default_range = dict(df_source=None,
                         column_name=['price_end'],  # 字符类型：list表示，提供参数的变异选项
                         window={'start': 0, 'end': 0, 'sep': 0}  # 数值类型：dict表示，参数的变异起终点、步长 | auto sep --  sep:True
                         )

    score_list = []
    score_num = 0
    avg_score = 0

    def __init__(self, *, df_source, kwargs=None, arg_range=None, refeature_pb=0.07):

        self.name = self.get_id(self.name)  # feature_ID
        self.result = pd.Series()   # indicator 计算出来的data 储存在这里

        self.df_source = df_source  # data_source. must have.

        if not arg_range:
            arg_range = self.default_range
        self.arg_range = arg_range  # 自动变异的范围

        if kwargs:
            self.kwargs = kwargs  # 导入既有参数
        else:
            self.kwargs = self.__generate_random_args()  # 主函数除df外所需全部数据

        self.score_list = []  # 得分相关记录
        self.score_num = 0
        self.avg_score = 0

        self.refeature_pb = refeature_pb
        self.pb_each = self.__calculate_pb_each(refeature_pb)  # 变异概率

    def __del__(self):
        # print('instance deleted: ', self)
        pass

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

    @classmethod
    def get_indicator_mutable_dimension_num(cls, arg_range=None):
        """indicator中可变异的参数的维度（根据columns数量决定）。"""

        if not arg_range:
            arg_range = cls.default_range

        num = 0
        for value in arg_range.values():
            if isinstance(value, dict) and value['sep']:
                num += 1
            if isinstance(value, list):
                num += len(value) - 1

        return num

    # ------------------------------------------------------------------------------------------------------------------
    def copy(self):

        new_ins = self.__class__(df_source=self.df_source,
                                 kwargs=self.kwargs,
                                 arg_range=self.arg_range,
                                 refeature_pb=self.refeature_pb)

        new_ins.__dict__ = self.__dict__.copy()
        new_ins.name = self.get_id('%s' % self.name.split('_')[0])
        new_ins.__dict__['kwargs'] = self.__dict__['kwargs'].copy()  # 注意这里，需要深拷贝的数据，要单独写一下

        return new_ins

    # ------------------------------------------------------------------------------------------------------------------

    def random_start(self):
        """用于系统生成参数的初次计算，或中途完成突变。返还计算结果"""

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

    def mutate_feature(self, mul=3.0, refeature_pb=None, update=True):
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

    def cal(self, sleep_time=0.1, *args, **kwargs):
        """rewrite this in subclass """

        if not kwargs:
            kwargs = self.kwargs
        else:
            self.kwargs = kwargs  # look to data from outside

        # main calculation starts here -------------------------------------------------------

        print(self.df_source, 'running cal() for instance', self.name)
        time.sleep(sleep_time)

        result = 'to be done.'


        # main calculation finish -------------------------------------------------------------

        self.result = result
        return result


# ======================================================================================================================

class MA(Indicator):
    """template"""

    name = 'simple moving average'   # change indicator name
    result = pd.Series()  # indicator 计算出来的data
    map_type = ['vector', 'condition']  # 本计算方式所的结果适应的分类方式

    default_range = dict(df_source=None,
                         column_name=['price_end', 'price_start'],
                         window={'start': 5, 'end': 400, 'sep': True}
                         )   # change default argument range
    score_list = []
    score_num = 0
    avg_score = 0

    def add_score(self, score):

        super(MA, self).add_score(score)  # change subclass_name
        MA.score_list.append(score)  # change subclass_name

    def update_score(self, sortino_score):

        super(MA, self).update_score(sortino_score)    # change subclass_name

        MA.avg_score = (MA.avg_score * MA.score_num + sortino_score) / (MA.score_num + 1)  # change subclass_name
        MA.score_num += 1  # change subclass_name

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

        self.result = result  # update result data
        return result


if __name__ == '__main__':

    p1 = Hello('AAA indicator')
    p1.show('aeaee', 'bibibi', 'cicici', param=654654, additional1='dididid', add2='eieiei')

    indicator_class_test = Indicator(df_source=None)
    i = indicator_class_test
    print(i)
    print(i.get_range())
    print(i.get_current_args())
    print(i.random_start())

    print('-' * 30)


    # 多进程cal计算 调用方式：
    args_list = []
    for x in range(9):
        instance = Indicator(df_source=x)
        args_for_one = (instance, 1)  # cal所需的参数（只支持位置参数）写在这tuple里，包括self实例本身
        args_list.append(args_for_one)  # 构造多进程参数列表

    Tools.agg_cal(Indicator.cal,  # 实例对应的计算函数，这里是Indicator的实例
                  *args_list,  # 参数列表
                  process_num=3  # 多进程数
                  )

    # copy --------------------------------

    print('copy issue ----------------------')

    print(i.__dict__)


