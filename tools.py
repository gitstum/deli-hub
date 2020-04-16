import os
import time
import pandas as pd
import numpy as np

import multiprocessing as mp


def agg_cal(func, *args, process_num):
    """Multi-processing function for feature calculation."""

    result_list1 = []
    result_list2 = []

    if not process_num:
        pool = mp.Pool()
    else:
        pool = mp.Pool(processes=process_num)

    for i in args:
        result1 = pool.apply_async(func, i)
        result_list1.append(result1)

    pool.close()
    pool.join()

    for r in result_list1:
        result2 = r.get()
        result_list2.append(result2)

    return result_list2


def get_tick_comp_price(file_path, to_path=None, start_date=None, end_date=None, exchange='BITMEX', symbol='XBTUSD'):
    """To get the compressed traded tick price.

    @param file_path: path that contains all(all will be dealt with) the trade csv files
        support bitmex only, for now.
    @param to_path: path to save csv file. None for not saving.
    @param start_date: str, format: '20150925'
    @param end_date: str, format: '20200101'. Note: end_day will NOT be included.
    @param exchange: support bitmex only, for now.
    @param symbol: Trading target
    @return: compressed tick price DataFrame with columns:
        timestamp
        price
    """

    if not exchange == 'BITMEX':
        return

    # 关闭 SettingWithCopyWarning 警告
    pd.set_option('mode.chained_assignment', None)

    t0 = time.time()

    list_tick_comp_price = []

    # get all file names
    file_list = sorted(os.listdir(file_path))
    for x in file_list.copy():
        if x[-4:] != '.csv':
            file_list.remove(x)
    if start_date:
        if pd.to_datetime(file_list[0][6:-4]) < pd.to_datetime(start_date):
            while len(file_list) > 0:
                if file_list[0] == 'trade_%s.csv' % start_date:
                    break
                file_list.pop(0)
    if end_date:
        if pd.to_datetime(file_list[-1][6:-4]) > pd.to_datetime(end_date):
            while len(file_list) > 0:
                if file_list[-1] == 'trade_%s.csv' % end_date:
                    file_list.pop(-1)
                    break
                file_list.pop(-1)

    # read and compress all the files in file_path
    for file in file_list:

        df = pd.read_csv('%s/%s' % (file_path, file), usecols=['timestamp', 'symbol', 'price', 'tickDirection'])
        if symbol not in list(df['symbol']):
            continue

        df['timestamp'] = df['timestamp'].str[:10] + ' ' + df['timestamp'].str[11:]  # change the unfriendly format!
        df = df[df.symbol == symbol]

        df = df[(df.tickDirection == 'PlusTick') | (df.tickDirection == 'MinusTick')]  # 仅保留有价格变动的行

        # 仅保留同方向连续吃两笔以上的tick --因为 limit order 成交的判断是：越过limit价格
        df['tickDirection_old'] = np.append(np.nan, df['tickDirection'][:-1])
        df['keep'] = 1
        df['keep'][df.tickDirection != df.tickDirection_old] = 0
        df.iloc[0, -1] = 1  # keep first line
        df = df[df['keep'] == 1]

        # 仅保留时间、价格列
        df.drop(['symbol', 'tickDirection', 'tickDirection_old', 'keep'], axis=1, inplace=True)

        list_tick_comp_price.append(df)
        print('%.3f | file "%s" included.' % ((time.time() - t0), file))

        # 储存
        if to_path:
            df.to_csv(('%s/%s' % (to_path, file)), index=False)  # the same file name.
            print('%.3f | file "%s" saved to "%s".' % ((time.time() - t0), file, to_path))

    df_tick_comp_price = pd.concat(list_tick_comp_price)
    df_tick_comp_price.reset_index(drop=True, inplace=True)

    print('%.3f | finish compressing traded tick price.' % (time.time() - t0))

    # 重新打开警告
    pd.set_option('mode.chained_assignment', 'warn')

    return df_tick_comp_price


def get_kbar(unit, file_path, to_path=None, to_name=None, start_date=None, end_date=None):
    """To get the K price DataFrame

    @param unit: compress unit: example: '30s', 't', '15t', 'h', 'd'. No more than 1 Day.
    @param file_path: path that contains the compressed csv files by get_tick_comp_price()
    @param to_path: path to save kbar csv file
    @param to_name: csv file name
    @param start_date: str, format: '20150925'
    @param end_date: str, format: '20200101'. Note: end_day will NOT be included.
    @return: compressed tick price DataFrame with columns:
        timestamp  --note that it's the period start timestamp
        price_start
        price_end
        price_max
        price_min
    """

    # get all file names
    file_list = sorted(os.listdir(file_path))
    for x in file_list.copy():
        if x[-4:] != '.csv':
            file_list.remove(x)
    if start_date:
        if pd.to_datetime(file_list[0][6:-4]) < pd.to_datetime(start_date):
            while len(file_list) > 0:
                if file_list[0] == 'trade_%s.csv' % start_date:
                    break
                file_list.pop(0)
    if end_date:
        if pd.to_datetime(file_list[-1][6:-4]) > pd.to_datetime(end_date):
            while len(file_list) > 0:
                if file_list[-1] == 'trade_%s.csv' % end_date:
                    file_list.pop(-1)
                    break
                file_list.pop(-1)

    # k线生成函数（最大单位：日）
    def compress_go(df, unit=unit):

        df['timestamp'] = pd.to_datetime(df.timestamp)
        df.set_index('timestamp', inplace=True)
        group_price = df.groupby(pd.Grouper(freq=('%s' % unit))).price
        price_start = group_price.first()
        price_end = group_price.last()
        price_max = group_price.max()
        price_min = group_price.min()

        df2 = pd.DataFrame(price_start)
        df2.rename({'price': 'price_start'}, axis=1, inplace=True)
        df2['price_end'] = price_end
        df2['price_max'] = price_max
        df2['price_min'] = price_min

        return df2

    # 执行
    df_kbar = pd.DataFrame()
    t0 = time.time()

    for file in file_list:

        df_new = pd.read_csv('%s/%s' % (file_path, file))
        print('%.3f | dealing with "%s"' % ((time.time() - t0), file))

        if df_kbar.empty:
            df_kbar = compress_go(df_new)
            continue

        df_new = compress_go(df_new)
        df_kbar = pd.concat([df_kbar, df_new], sort=False)

    print('%.3f | finish kbar.' % (time.time() - t0))

    if to_path and to_name:
        df_kbar.to_csv('%s/%s.csv' % (to_path, to_name))

    return df_kbar
