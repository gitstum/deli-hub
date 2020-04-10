import pandas as pd
import numpy as np
import tools


# TODO: test it


def cal_kstyle(price_start, price_end, price_max, price_min):
    """Categorize the K bar style"""

    color = None  # -1 for down, 0 for even, 1 for up

    len_bar = 0  # K线实体部分长度，绝对值
    len_up_pin = 0  # K线上方针
    len_down_pin = 0  # K线下方针

    rate_bar = 0  # K线实体部分长度比例
    rate_up_pin = 0  # K线上方针长度比例
    rate_down_pin = 0  # K线下方针长度比例

    bar_move = price_end - price_start
    len_bar = abs(bar_move)
    
    if bar_move > 0:
        color = 1
    elif bar_move < 0:
        color = -1
    else:
        color = 0

    line_up = max(price_start, price_end)
    line_down = min(price_start, price_end)

    len_up_pin = price_max - line_up
    len_down_pin = line_down - price_min

    rate_bar = len_bar / price_start
    rate_up_pin = len_up_pin / price_start
    rate_down_pin = len_down_pin / price_start

    return color, len_bar, len_up_pin, len_down_pin, rate_bar, rate_up_pin, rate_down_pin


def cal(df, price_start, price_end, price_max, price_min, inplace=True):
    """Categorize the K bar style, pd.DataFrame method

    @param df: pd.DataFrame which includes prices info
    @param price_start: NAME of df column for starting prices
    @param price_end: NAME of df column for ending prices
    @param price_max: NAME of df column for max prices
    @param price_min: NAME of df column for min prices
    @param inplace: True to make change on df, False for returning a new df for view.
    @return: average true range for prices, a python list

    See cal_kstyle() for more info.
    """

    df_temp = df.copy()

    df_temp['color'] = 0  # this first.

    df_temp['bar_move'] = df[price_end] - df[price_start]
    df_temp['len_bar'] = df_temp['bar_move'].abs() 
    
    df_temp['color'][df_temp.bar_move > 0] = 1
    df_temp['color'][df_temp.bar_move < 0] = -1

    df_temp['line_up'] = df_temp[price_start]
    df_temp['line_up'][df_temp.price_start < df_temp.price_end] = df_temp[price_end]  # Not sure if this works.
    df_temp['line_down'] = df_temp[price_start]
    df_temp['line_down'][df_temp.price_start > df_temp.price_end] = df_temp[price_end]  # Not sure if this works.

    df_temp['len_up_pin'] = df_temp[price_max] - df_temp['line_up']
    df_temp['len_down_pin'] = df_temp['line_down'] - df_temp[price_min]

    df_temp['rate_bar'] = df_temp['len_bar'] / df_temp[price_start]
    df_temp['rate_up_pin'] = df_temp['len_up_pin'] / df_temp[price_start]
    df_temp['rate_down_pin'] = df_temp['len_down_pin'] / df_temp[price_start]

    if inplace:
        df = df_temp
        df.drop(['bar_move', 'line_up', 'line_down'], axis=1, inplace=True)
    else:
        return df_temp


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = tools.agg_cal(cal, *args, process_num=process_num)

    return result
