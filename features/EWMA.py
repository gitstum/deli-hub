import pandas as pd
import tools


def cal(the_series, index_value, kind='com'):
    """Exponential Weighted Moving Average Calculation

    :param the_series: sequence, which can be:
        price,
        STDs (within each line, which is counted by smaller units),
        volumes,
        order numbers(1, plus/minus, 5, all),
        etc.
    :param index_value: parameter of 'kind', com / alpha
    :param kind: number, 'com' (> 0) / 'alpha' (0 < alpha ≤ 1)
    """

    if not isinstance(the_series, pd.Series):
        the_series = pd.Series(the_series)

    if kind == 'alpha':
        exponential_weighted_moving_average = the_series.ewm(alpha=index_value).mean()
    else:
        exponential_weighted_moving_average = the_series.ewm(com=index_value).mean()

    return exponential_weighted_moving_average


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = tools.agg_cal(cal, *args, process_num=process_num)

    return result
