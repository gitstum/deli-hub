import pandas as pd
from tools import Tools
import numpy as np

arg_range = dict(df_source=None,
                 column_name=['price_end', 'price_avg'],
                 window={'start': 4, 'end': 80, 'sep': 1}
                 )

def cal(*, df_source, column_name, window=20):
    """Moving Average Calculation for the_series

    :param window: rolling window, int
    :return: moving average the_series, Series
    """

    result = df_source[column_name].rolling(window).mean()

    return result


def agg_cal(*args, process_num=None):
    """Multi-processing calculator."""

    result = Tools.agg_cal(cal, *args, process_num=process_num)

    return result
