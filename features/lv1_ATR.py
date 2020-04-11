import numpy as np


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
    :param price_start: NAME of df column for starting prices
    :param price_end: NAME of df column for ending prices
    :param price_max: NAME of df column for max prices
    :param price_min: NAME of df column for min prices
    :return: average true range for prices, a python LIST
    """

    arr_prices = np.array([
        df[price_start],
        df[price_end],
        df[price_max],
        df[price_min]
    ])  # for np is faster than pd

    atr_list = []
    before_end = arr_prices[0, 0]

    num = 0
    max_line = arr_prices.shape[1]
    while num < max_line:

        data = arr_prices[..., num]
        now_start = data[0]
        now_end = data[1]
        now_max = data[2]
        now_min = data[3]

        atr_value = cal_atr(before_end, now_start, now_max, now_min)  # atr value of each line
        atr_list.append(atr_value)

        before_end = now_end
        num += 1

    return atr_list

