import os
import numpy as np
import pandas as pd
import features

if __name__ == '__main__':

    df = pd.read_csv('data/bitmex_price_1hour_2020q1.csv')
    data_series = df.price_end

    result = features.QT.cal(data_series, 24, 0.05)

    print(result)

    print('------')

    r1, r2, r3 = features.QT.agg_cal((data_series, 24, 0.05),
                            (data_series, 48, 0.5),
                            (data_series, 72, 0.95),
                            )
    print('-' * 40)
    print(r1)
    print('-' * 40)
    print(r2)
    print('-' * 40)
    print(r3)

