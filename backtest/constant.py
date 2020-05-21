#--coding:utf-8--

from enum import Enum
import numpy as np



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

# ---------------------------------------------------------------------------------------------------------

# Method Category

class Method(Enum):
    """合并成pos_should信号的方法
    相关函数：
        Tools.sig_to_one()
        Tools.各方法名字对应的函数()
    """

    ALL = ['comb_sum', 'comb_vote1', 'comb_vote2', 'comb_vote3', 'comb_min',
           'perm_cond', 'perm_add', 'perm_sub', 'perm_up', 'perm_down']

    # 组合求解
    COMBINATION = ['comb_sum',  # 对各列signal进行加和
                   'comb_vote1',  # 使用各列signal投票，加和，输出为：-1/0/1
                   'comb_vote2',  # 使用各列signal投票，须无反对票，输出为：-1/0/1
                   'comb_vote3',  # 使用各列signal投票，须全票通过，输出为：-1/0/1
                   'comb_min'  # 多/空方向：取各列signal中最小/最大的（以做多/空）。如sig含有相反符号，则返回0（可用于判断）
                   ]

    # 排列求解：按序依次判断（/比较）
    PERMUTATION = ['perm_cond',  # 在满足a方向（正负，对比0）的条件下，使用同方向（正负）的b的pos_should，其余0 【限2列】
                   'perm_add',  # 一直涨，sig值越来越大:1，否则0
                   'perm_sub',  # 一直跌，sig值越来越小:1， 否则0
                   'perm_up',  # sig值震荡（含持平）上涨：1，否则0
                   'perm_down'  # sig值震荡（含持平）下跌：1，否则0
                   ]




if __name__ == '__main__':
    print(Direction.SHORT.value)
    print(Direction.SHORT)
    print(CAPITAL)






