#--coding:utf-8--

from enum import Enum


class Direction(Enum):
    """Order Direction"""

    LONG = 'long'
    SHORT = 'short'


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

    LIMIT = 'limit'
    MARKET = 'market'
    STOP = 'stop'

