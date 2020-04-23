#--coding:utf-8--

from enum import Enum
import numpy as np



class Direction(Enum):
    """Order Direction"""

    LONG = 'long'
    SHORT = 'short'
    NONE = 'none'
    UP = 'up'
    DOWN = 'down'


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


# Minimum price distance change range
MIN_DISTANCE = 0.5

# Slippage min and max edge ticks number
SLIPPAGE_TICK_MIN = 1500
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


if __name__ == '__main__':
    print(Direction.SHORT.value)
    print(Direction.SHORT)
    print(CAPITAL)
    print(CODE_MAP)