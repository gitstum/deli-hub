import pandas as pd
import tools


def cal(prices, window, quantile):
    """"Quantile Sequence in Window

    :param prices: prices, sequence
    :param window: rolling window, int
    :param quantile: float, 0 < q < 1
    :return: quantile values of prices, Series
    """

    if not isinstance(prices, pd.Series):
        series = pd.Series(prices)

    quantile_values = prices.rolling(window).quantile(quantile, interpolation='linear')

    return quantile_values


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = tools.agg_cal(cal, *args, process_num=process_num)

    return result
