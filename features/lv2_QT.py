import pandas as pd
import tools


def cal(the_series, window, quantile):
    """"Quantile Sequence within a Rolling Window

    :param the_series: sequence
    :param window: rolling window, int
    :param quantile: float, 0 < q < 1
    :return: quantile values of the_series, Series
    """

    if not isinstance(the_series, pd.Series):
        the_series = pd.Series(the_series)

    quantile_values = the_series.rolling(window).quantile(quantile, interpolation='linear')

    return quantile_values


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = tools.agg_cal(cal, *args, process_num=process_num)

    return result
