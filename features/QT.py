import pandas as pd
import tools


def cal(series, window, quantile):
    """"Quantile Sequence in Window

    :param series: sequence
    :param window: rolling window
    :param quantile: 0 < q < 1
    :return: quantile value series
    """

    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    quantile_values = series.rolling(window).quantile(quantile, interpolation='linear')

    return quantile_values


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = tools.agg_cal(cal, *args, process_num=process_num)

    return result
