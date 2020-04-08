import pandas as pd
import tools


def cal(prices, index_value, kind='com'):
    """Exponential Weighted Moving Average Calculation for Prices

    :param prices: prices, sequence
    :param index_value: parameter of 'kind', com / alpha
    :param kind: number, 'com' (> 0) / 'alpha' (0 < alpha ≤ 1)
    """

    if not isinstance(prices, pd.Series):
        series = pd.Series(prices)

    if kind == 'alpha':
        exponential_weighted_moving_average = prices.ewm(alpha=index_value).mean()
    else:
        exponential_weighted_moving_average = prices.ewm(com=index_value).mean()

    return exponential_weighted_moving_average


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = tools.agg_cal(cal, *args, process_num=process_num)

    return result
