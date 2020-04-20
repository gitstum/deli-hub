import time
import multiprocessing as mp

from tools import Tools


def cal(arg1=888, arg2=999):
    """A test calculation of an indicator."""

    print('%s | A test indicator is being calculated on arg(%s, %s).' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), arg1, arg2))
    time.sleep(1)

    result = 'Result of args(%s, %s).' % (arg1, arg2)

    return result


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = Tools.agg_cal(cal, *args, process_num=process_num)

    return result


if __name__ == '__main__':

    cal()

    # way1 for calling agg_cal()--------------------

    r1, r2, r3 = agg_cal([11, 313],
                         [22, 414],
                         [33, 565],
                         process_num=2)
    print('-' * 40)
    print(r1)
    print(r2)
    print(r3)

    # way2 for calling agg_cal()--------------------
    print('=' * 40)

    params = (
        [1.1, 100],
        [2.2, 200],
        [3.3, 300],
        [4.4, 400],
    )
    r1, r2, r3, r4 = agg_cal(*params, process_num=3)

    print('-' * 40)
    print(r1)
    print(r2)
    print(r3)
    print(r4)

    # wrong way for calling agg_cal()--------------------
    print('=' * 40)

    params = (
        {'arg1_1': 1,
         'arg1_2': 11},
        {'arg2_1': 2,
         'arg2_2': 222},
        {'arg3_1': 3,
         'arg3_2': 333},
        {'arg4_1': 4,
         'arg4_2': 444},
    )
    r1, r2, r3, r4 = agg_cal(*params, process_num=3)

    print('-' * 40)
    print(r1)
    print(r2)
    print(r3)
    print(r4)
