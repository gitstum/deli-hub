import pandas as pd
import tools


def cal(STDs, index_value, kind='com'):
    """Exponential Weighted Moving Average Calculation
    for Standard Deviations by Each STDs

    :param STDs: standard deviations within each line (which is
                probably counted by tick data), sequence.
                (useful for big range groupby)
    :param index_value: parameter of 'kind', com / alpha
    :param kind: number, 'com' (> 0) / 'alpha' (0 < alpha ≤ 1)
    """

    if not isinstance(STDs, pd.Series):
        STDs = pd.Series(STDs)

    if kind == 'alpha':
        exponential_weighted_moving_average = STDs.ewm(alpha=index_value).mean()
    else:
        exponential_weighted_moving_average = STDs.ewm(com=index_value).mean()

    return exponential_weighted_moving_average


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = tools.agg_cal(cal, *args, process_num=process_num)

    return result
