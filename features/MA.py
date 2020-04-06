import pandas as pd
import tools


def cal(series, window):
    """Moving Average Calculation"""

    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    moving_average = series.rolling(window).mean()

    return moving_average


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = tools.agg_cal(cal, *args, process_num=process_num)

    return result
