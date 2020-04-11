import pandas as pd
import numpy as np


def cal_kstyle(price_start, price_end, price_max, price_min):
    """K Bar Extensions Calculator"""

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
    len_all = price_max - price_min

    bar_move_rate = 100 * bar_move / price_start
    rate_bar = 100 * len_bar / price_start
    rate_up_pin = 100 * len_up_pin / price_start
    rate_down_pin = 100 * len_down_pin / price_start

    result = {'K_color': color,  # -1 for down, 0 for even, 1 for up
              'K_move': bar_move,  # 价格移动值，正负
              'K_move_rate': bar_move_rate,  # 价格移动比例，正负
              'K_len_all': len_all,  # 区间极值
              'K_len_body': len_bar,  # K线实体部分长度，绝对值
              'K_len_up': len_up_pin,  # K线上方针长度，绝对值
              'K_len_down': len_down_pin,  # K线下方针长度，绝对值
              'K_ratio_bar': rate_bar,  # K线实体部分长度比例，绝对值
              'K_ratio_up': rate_up_pin,  # K线上方针长度比例，绝对值
              'K_ratio_down': rate_down_pin  # K线下方针长度比例，绝对值
    }

    return result


def cal(df, price_start, price_end, price_max, price_min, inplace=False):
    """K Bar Extensions, Categorized. Not inplace!

    @param df: pd.DataFrame which includes prices info
    @param price_start: NAME of df column for starting prices
    @param price_end: NAME of df column for ending prices
    @param price_max: NAME of df column for max prices
    @param price_min: NAME of df column for min prices
    @param inplace: True to make change on df, False for returning a new df for view.
    @return: a pd.DataFrame view. Never inplace! (require re-writen)

    See cal_kstyle() for more info. The reason why I don't use df mothod here:
    """

    if inplace:
        print('lv1_Kex function never works inplace.')
        return

    arr_prices = np.array([
        df[price_start],
        df[price_end],
        df[price_max],
        df[price_min]
    ])  # for np is faster than pd

    result_list = []

    num = 0
    max_line = arr_prices.shape[1]
    while num < max_line:

        data = arr_prices[..., num]
        price_start = data[0]
        price_end = data[1]
        price_max = data[2]
        price_min = data[3]

        kex = cal_kstyle(price_start, price_end, price_max, price_min)
        result_list.append(kex)

        num += 1

    df_kex = pd.DataFrame(result_list)
    df = pd.concat([df, df_kex], axis=1, sort=False)

    return df
