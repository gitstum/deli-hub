import pandas as pd
import numpy as np
import tools

# TODO: test it


def cal_atr(before_end, now_start, now_max, now_min):
    """Average True Range, Single Number Calculator

    :param each: price, number
    :return: atr, a number
    """

    value = max(now_max - now_min,
                abs(now_max - before_end),
                abs(before_end - now_min)
                )

    now_atr = 100 * value / now_start

    return now_atr


def cal(df, price_start, price_end, price_max, price_min):
    """Average True Range, Series Calculator

    :param df: pd.DataFrame which includes prices info
    :param price_start: df column name of starting prices
    :param price_end: df column name of ending prices
    :param price_max: df column name of max prices
    :param price_min: df column name of min prices
    :return: average true range for prices, a python list
    """

    arr_prices = np.array([
        df[price_start],
        df[price_end],
        df[price_max],
        df[price_min]
    ])

    atr_list = []
    before_end = arr_prices[0, 0]

    num = 0
    while num < arr_prices.shape[1]:

        data = arr_prices[..., num]
        now_start = data[0]
        now_end = data[1]
        now_max = data[2]
        now_min = data[3]

        value = max(now_max - now_min,
                    abs(now_max - before_end),
                    abs(before_end - now_min)
                    )

        atr_value = 100 * value / now_start  # atr value of each line
        atr_list.append(atr_value)

        before_end = now_end
        num += 1

    return atr_list


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = tools.agg_cal(cal, *args, process_num=process_num)

    return result
