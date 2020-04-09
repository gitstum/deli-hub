import pandas as pd
import tools


def cal(the_series, window):
    """Moving Average Calculation for the_series

    :param the_series: sequence, which can be:
        price,
        STDs (within each line, which is counted by smaller units),
        volumes,
        order numbers(1, plus/minus, 5, all),
        etc.
    :param window: rolling window, int
    :return: moving average the_series, Series
    """

    if not isinstance(the_series, pd.Series):
        the_series = pd.Series(the_series)

    moving_average = the_series.rolling(window).mean()

    return moving_average


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = tools.agg_cal(cal, *args, process_num=process_num)

    return result
