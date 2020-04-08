import pandas as pd
import tools


def cal(prices, window):
    """Moving Average Calculation for Standard Deviations

    :param prices: prices, sequence
    :param window: rolling window, int
    :return: moving average prices, Series
    """

    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)

    moving_average = prices.rolling(window).std()

    return moving_average


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = tools.agg_cal(cal, *args, process_num=process_num)

    return result
