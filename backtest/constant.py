# --coding:utf-8--

from enum import Enum
import numpy as np
import string


class Direction(Enum):
    """Order Direction"""

    LONG = 1  # 'long'
    SHORT = -1  # 'short'
    NONE = 0  # 'none'
    # UP = 'up'
    # DOWN = 'down'


class Status(Enum):
    """Order status."""

    SUBMITTING = 'submitting'
    NOTTRADED = 'not_traded'
    PARTTRADED = 'part_traded'
    ALLTRADED = 'all_traded'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'


class OrderType(Enum):
    """Order type."""

    LIMIT = 'limit'  # limit order
    LIMIT_NEXT = 'limit_next'  # limit order that can only be traded in the K period, cancelled later.
    MARKET = 'market'
    STOP = 'stop'
    LIQUIDATION = 'liquidation'


class OrderFee(Enum):
    MAKER = 0.0001
    TAKER = -0.00075


# MARKET price
MARKET_PRICE = 0  # use 0 for market price.

# Minimum price distance change range
MIN_DISTANCE = 0.5

# Slippage min and max edge ticks number
SLIPPAGE_TICK_MIN = 1800
SLIPPAGE_TICK_MAX = 5000

# Overload tick number.
OVERLOAD_EDGE = 6000

# How much value initially hold
CAPITAL = 1000

# How much value for volume base
VOLUME_BASE = 100

# How many period one year has
ANNUAL_PERIOD = 365 * 24

# At what drawdown percentage of avg_price should force a STOP loss for all.

# Fee rate in float for each order type
FEE_MAP = {
    OrderType.LIMIT: 0.0001,
    OrderType.MARKET: -0.00075,
    OrderType.STOP: -0.00075
}

# Code that map to side(direction) and order_value(volume)
# CODE_MAP = {
#     'strong_short': {'side': Direction.SHORT, 'order_value': VOLUME_BASE},
#     'short': {'side': Direction.SHORT, 'order_value': VOLUME_BASE},
#     'leave': {'side': Direction.OUT, 'order_value': np.nan},
#     'buy': {'side': Direction.LONG, 'order_value': VOLUME_BASE},
#     'strong_buy': {'side': Direction.LONG, 'order_value': VOLUME_BASE},
# }

# Position held when strategy start.
POS_START = 0

# Default distance(to tick price) for limit order
LIMIT_DISTANCE = 0.5  # the bigger the safer but harder to get traded

# Node name order in node map
NODE_LETTER_LIST = sorted(list(string.ascii_letters))  # ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz


# ---------------------------------------------------------------------------------------------------------

class Method(Enum):
    """Method Category
    - node method: 合并sig信号
    - terminal method / classifier: 将特征、指标转化为分类

    相关函数：
        Tools.sig_to_one()
        Tools.各方法名字对应的函数()

    """

    ALL_NAME = ['cond_1', 'cond_2',
                'mult_simple', 'mult_same', 'mult_abs', 'divi_simple', 'divi_same', 'divi_abs',
                'comb_sum', 'comb_vote0', 'comb_vote1', 'comb_vote2', 'comb_vote3', 'comb_min',
                'perm_cond', 'perm_add', 'perm_sub', 'perm_up', 'perm_down',
                'sig_trend_strict', 'sig_trend_loose', 'sig_trend_start_end'
                ]

    # signal 条件求解「限2列」
    CONDITION = ['cond_1',  # 在满足a(0/1)条件(1)的条件下，使用b的pos_should，其余0 【限2列】「(0/1)条件必须是terminal」
                 'cond_2'  # 在满足a方向（正负，对比0）的条件下，使用同方向（正负）的b的pos_should，其余0 【限2列】「可用于高层树枝」
                 ]

    # signal 增强、弱化(乘除) 「限2项」
    ENFORCE = ['mult_simple',  # 简单相乘。适用于一方是0/1类型，另一方-1/0/1类型，强化后者 「(0/1)类型必须是terminal」(至少要降低其他的概率。。)
               'mult_same',  # 条件相乘。同方向时：保留符号，计算相乘结果；其他：0。
               'mult_abs',  # 绝对幅度变化。主sig和副sig的绝对值相乘，改变主sig的幅度，不改变方向。
               'divi_simple',  # 简单相除。同上。「(0/1)类型必须是terminal」
               'divi_same',  # 条件相除。同上。
               'divi_abs'  # 绝对幅度变化。同上。
               ]

    # signal 组合求解
    COMBINATION = ['comb_sum',  # 对各列signal进行加和，含小数
                   'comb_vote1',  # 使用各列signal投票，加和，输出为：-1/0/1
                   'comb_vote2',  # 使用各列signal投票，须无反对票，输出为：-1/0/1
                   'comb_vote3',  # 使用各列signal投票，须全票通过，输出为：-1/0/1
                   'comb_min'  # 多/空方向：取各列signal中最小/最大的（以做多/空）。如sig含有相反符号，则返回0（可用于判断）
                   ]

    # signal/feature 排列求解：按序依次判断（/比较）
    PERMUTATION = ['perm_add',  # 一直涨，sig值越来越大:1，否则0
                   'perm_sub',  # 一直跌，sig值越来越小:1， 否则0
                   'perm_up',  # sig值震荡（含持平）上涨：1，否则0
                   'perm_down',  # sig值震荡（含持平）下跌：1，否则0
                   'sig_trend_strict',  # sig一直上涨：1， sig一直下跌：-1， 否则0
                   'sig_trend_loose',  # sig整体上涨（可局部持平）：1， sig整体下跌（可局部持平）：-1， 否则0
                   'sig_trend_start_end'  # sig震荡上涨：1，sig值震荡下跌：-1，否则0 （只考虑头尾）
                   ]

    # feature 单项计算：自切割分类函数 classifier
    cut = ['cut_number',  # 自切割，常量切割
           'cut_rank',  # 自切割，百分比排名切割
           'cut_sigma',  # 自切割，sigma比例切割
           'cut_distance'  # 自切割，全距比例切割
           ]

    # feature 双项计算：对比切割分类函数 classifier
    COMPARE = ['compare_distance',  # 比较feature差值，绝对距离（比大小：距离为0）
               'compare_sigma'  # 比较feature差值，平均标准差比例（比大小：比例为0）
               ]

    # Classifier Category
    CATES = ['origin',  # 有含义的原始数据，价格、交易量等等
             'real',  # 负无穷到正无穷
             'abs',  # 0到正无穷
             'normal_real',  # 负无穷到正无穷, 标准化
             'normal_abs'  # 0到正无穷 标准化
             ]  


#     # ------------------------------------------------------------------------
#     method_abs_restrict = dict(cond_1=Tools.cond_1,
#                                mult_simple=Tools.mult_simple,
#                                divi_simple=Tools.divi_simple
#                                )

#     method_output_01 = dict(perm_add=Tools.perm_add,
#                             perm_sub=Tools.perm_sub,
#                             perm_up=Tools.perm_up,
#                             perm_down=Tools.perm_down
#                             )  # condition

#     method_output_float = dict(cond_1=Tools.cond_1,
#                                cond_2=Tools.cond_2,
#                                mult_simple=Tools.mult_simple,
#                                mult_same=Tools.mult_same,
#                                mult_abs=Tools.mult_abs,
#                                divi_simple=Tools.divi_simple,
#                                divi_same=Tools.divi_same,
#                                divi_abs=Tools.divi_abs,
#                                comb_sum=Tools.comb_sum,
#                                comb_min=Tools.comb_min
#                                )  # 没有约束为整数，但可能为整数

#     # method_ouput_vector = 除了meaningless的，都是有方向含义的：
#     method_output_meaningless = dict(cut_number=Tools.cut_number,
#                                      cut_rank=Tools.cut_rank,
#                                      cut_sigma=Tools.cut_sigma,
#                                      cut_distance=Tools.cut_distance,
#                                      compare_distance=Tools.compare_distance,
#                                      compare_sigma=Tools.compare_sigma
#                                      )


if __name__ == '__main__':
    print(Direction.SHORT.value)
    print(Direction.SHORT)
    print(CAPITAL)
