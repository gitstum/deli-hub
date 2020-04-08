import pandas as pd
import tools


def cal(STDs, window):
    """Moving Average Calculation for Standard Deviations by Each STDs

    :param STDs: standard deviations within each line (which is
                probably counted by tick data), sequence.
                (useful for big range groupby)
    :param window: rolling window, int
    :return: moving average prices, Series
    """

    if not isinstance(STDs, pd.Series):
        STDs = pd.Series(STDs)

    moving_average = STDs.rolling(window).mean()  # note it's mean() here.

    return moving_average


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = tools.agg_cal(cal, *args, process_num=process_num)

    return result
