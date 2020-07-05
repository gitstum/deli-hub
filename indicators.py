# --coding:utf-8--

import random
import numpy as np
import pandas as pd


class Hello(object):

    name = 'test indicator'

    def __init__(self, name=name):
        self.name = name

    def show(self, tag='test', *args, param=None, **kwargs):
        print('%s | %s, args: %s, param: %s, %s' % (tag, self.name, args, param, kwargs))


class Indicator(object):

    name = 'Indicator'

    default_range = dict(df_source=None,
                         column_name=['price_end'],
                         window={'start': 0, 'end': 0, 'sep': 0}
                         )

    def __init__(self, *, df_source, name=None, arg_range=None, refeature_pb=0.07):

        self.df_source = df_source  # must have

        if name:
            self.name = name
        if not arg_range:
            arg_range = self.default_range
        self.arg_range = arg_range
        self.refeature_pb = refeature_pb

        self.kwargs = self.__generate_random_args()
        self.pb_each = self.__calculate_pb_each(refeature_pb)

    def get_name(self):
        return self.name

    def get_range(self):
        return self.arg_range

    def get_current_args(self):
        return self.kwargs

    def __generate_random_args(self):
        """随机生成主函数的一组参数  --除了df，都放到kwargs里面来"""

        kwargs = {}
        for key, value in self.arg_range.items():
            if value:
                arg = None
                if isinstance(value, list):
                    arg = random.choice(value, 1)
                if isinstance(value, dict):
                    sep = value['sep']
                    if sep == 0:
                        arg = value['start']
                    else:
                        start = value['start']
                        end = value['end'] + sep  # to include the end data
                        arg = np.random.choice(np.arange(start, end, sep))
                kwargs[key] = arg

        return kwargs

    def __calculate_pb_each(self, refeature_pb, inplace=True):
        """计算每个参数的变异概率"""

        mut_arg_num = 0

        for value in self.kwargs.values():
            if isinstance(value, dict) and value['sep']:
                mut_arg_num += 1
            if isinstance(value, list) and len(value) > 1:
                mut_arg_num += 1
        
        if mut_arg_num == 0:
            pb_each = 0
        else:
            pb_each = self.__probability_each(mut_arg_num, refeature_pb)

        if inplace:
            self.pb_each = pb_each  # 全局更新实例的变异概率

        return pb_each

    @staticmethod
    def __probability_each(object_num, pb_for_all):

        pb_for_each = 1 - (1 - pb_for_all) ** (1 / object_num)
        return pb_for_each

    def random_start(self):

        kwargs = self.__generate_random_args()
        result = self.cal(**kwargs)

        return result

    def mutate_args(self, refeature_pb=None, update=False):

        mut_flag = False

        if refeature_pb is not None:
            pb_each = self.__calculate_pb_each(refeature_pb, inplace=update)
        else:
            pb_each = self.pb_each

        for key, value in self.arg_range.items():

            current_value = self.kwargs[key]
            new_value = None

            if isinstance(value, list) and len(value) > 1:
                if random.random() < pb_each:
                    choice_box = value.copy()
                    choice_box.remove(current_value)
                    new_value = random.choice(choice_box)

            if isinstance(value, dict) and value['sep'] != 0:
                sep = value['sep']
                end = value['end'] + sep
                new_value = np.random.choice(np.arange(value['start'], end, sep))

            if new_value is not None:
                self.kwargs[key] = new_value
                mut_flag = True

        return mut_flag

    def cal(self, **kwargs):
        """rewrite this."""

        if not kwargs:
            kwargs = self.kwargs
        else:
            self.kwargs = kwargs  # look to data from outside

        # main calculation
        result = 'to be done.'

        return result


# ======================================================================================================================


class MA(Indicator):

    name = 'simple moving average'
    default_range = dict(df_source=None,
                         column_name=['price_end', 'price_avg'],
                         window={'start': 4, 'end': 80, 'sep': 1}
                         )

    def cal(self, **kwargs):

        if not kwargs:
            kwargs = self.kwargs
        else:
            self.kwargs = kwargs  # look to data from outside

        # main calculation
        column_name = kwargs['column_name']
        window = kwargs['window']
        result = self.df_source[column_name].rolling(window).mean()

        return result


if __name__ == '__main__':

    # p1 = Hello('AAA indicator')
    # p1.show('aeaee', 'bibibi', 'cicici', param=654654, additional1='dididid', add2='eieiei')

    indicator_class_test = Indicator(df_source=None)
    i = indicator_class_test
    print(i)
    print(i.get_range())
    print(i.get_current_args())
    print(i.random_start())

    print('-' * 30)

    df = pd.read_csv('private/test_data(2018-2019).csv')
    test = MA(df_source=df)
    result = test.cal()

    print(test)
    print(test.get_range())
    print(test.get_current_args())
    print(result)
