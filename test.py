import os

from features import *


if __name__ == '__main__':

    test_features.cal()

    print('------')

    r1, r2, r3 = test_features.agg_cal(['test1', 121],
                         ['test2', 212],
                         ['test3', 343],
                         process_num=2)
    print('-' * 40)
    print(r1)
    print(r2)
    print(r3)

