import pandas as pd
import tools


def cal(series, param, kind='com'):
    """Exponential Weighted Moving Average
    :param series: sequence
    :param param: parameter of 'kind'
    :param kind: 'com' (> 0) / 'alpha' (0 < alpha ≤ 1)
    """

    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    if kind == 'alpha':
        exponential_weighted_moving_average = series.ewm(elpha=param).mean()
    else:
        exponential_weighted_moving_average = series.ewm(com=param).mean()

    return exponential_weighted_moving_average


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = tools.agg_cal(cal, *args, process_num=process_num)

    return result
