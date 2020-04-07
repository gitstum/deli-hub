import os
import numpy as np
import pandas as pd
import features

if __name__ == '__main__':

    data_series = np.random.rand(200)

    result = features.EWMA.cal(data_series, 20)

    print(result)

    print('------')

    r1, r2, r3 = features.EWMA.agg_cal((data_series, 20),
                            (data_series, 40),
                            (data_series, 80),
                            )
    print('-' * 40)
    print(r1)
    print('-' * 40)
    print(r2)
    print('-' * 40)
    print(r3)

