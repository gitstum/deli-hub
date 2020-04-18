import os
import time
import pandas as pd
import numpy as np

import multiprocessing as mp


def agg_cal(func, *args, process_num=None):
    """Multi-processing framework for functions.

    How to call:
    result1, result2 = func.agg_cal((param1,), (param2,))
    """

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


def get_csv_names(file_path, start_date=None, end_date=None, cat='BITMEX.trade'):
    """To get all target file names. Date range support only for bitmex TRADE csv, for now

    @param file_path: path that contains all the csv files(each day, no break)
    @param start_date: str, format: '20150925'
    @param end_date: str, format: '20200101'. Note: end_day will NOT be included.
    @param cat: exchange.csv_type
    @return: file names list
    """

    file_list = sorted(os.listdir(file_path))
    for x in file_list.copy():
        if x[-4:] != '.csv':
            file_list.remove(x)

    if not cat[:6] == 'BITMEX':
        return file_list

    if start_date:
        if pd.to_datetime(file_list[0][6:-4]) < pd.to_datetime(start_date):
            print(888)
            while len(file_list) > 0:
                if file_list[0][6:-4] == start_date:
                    break
                file_list.pop(0)
    if end_date:
        if pd.to_datetime(file_list[-1][6:-4]) > pd.to_datetime(end_date):
            while len(file_list) > 0:
                if file_list[-1][6:-4] == end_date:
                    file_list.pop(-1)  # end date is not included
                    break
                file_list.pop(-1)

    return file_list


def bitmex_time_format_change(bitmex_time):
    """Change the unfriendly bitmex time format!

    @param bitmex_time: original timestamp Series from bitmex trade/quote data
    @return: datetime64 timestamp Series.
    """

    bitmex_time = bitmex_time.str[:10] + ' ' + bitmex_time.str[11:]  # change the unfriendly time format!
    timestamp_series = pd.to_datetime(bitmex_time)

    return timestamp_series


def get_kbar(file_path_trade, file_path_quote, unit, to_path=None, to_name=None, start_date=None, end_date=None,
             exchange='BITMEX', symbol='XBTUSD'):
    """To get the K price DataFrame. NO more than 1 day

    @param file_path_trade: path that contains the trade csv files(fromt bitmex)
    @param unit: compress unit: example: '30s', 't', '15t', 'h', 'd'. No more than 1 Day.
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

    if not exchange == 'BITMEX':
        return

    t0 = time.time()

    # Two process for two data source
    df_kbar, df_ticks = agg_cal(compress_distribute,
                                ('trade', file_path_trade, unit, start_date, end_date, symbol),
                                ('quote', file_path_quote, unit, start_date, end_date, symbol),
                                process_num=2)  # 我承认这种传参方式很奇葩。。。

    df_kbar = pd.concat([df_kbar, df_ticks], axis=1, sort=False)
    df_kbar.rename({'symbol': 'ticks'}, axis=1, inplace=True)

    print('%.3f | kbar generated successfully.' % (time.time() - t0))

    if to_path and to_name:
        df_kbar.to_csv('%s/%s.csv' % (to_path, to_name))
        print('%.3f | "%s" saved successfully to "%s".' % ((time.time() - t0), to_name, to_path))

    return df_kbar


def compress_trade(file_path_trade, unit, start_date, end_date, symbol):
    """trade csv dealing: get basic price info"""

    list_kbar = []
    t0 = time.time()

    file_list_trade = get_csv_names(file_path_trade, start_date=start_date, end_date=end_date, cat='BTIMEX.trade')
    for file in file_list_trade:

        df_new = pd.read_csv('%s/%s' % (file_path_trade, file),
                             usecols=['timestamp', 'symbol', 'side', 'size', 'price'])
        if symbol not in list(df_new['symbol']):
            continue
        df_new = df_new[df_new.symbol == symbol]
        df_new.reset_index(drop=True, inplace=True)

        df_new['timestamp'] = bitmex_time_format_change(df_new['timestamp'])
        group_price = df_new.groupby(pd.Grouper(key='timestamp', freq=('%s' % unit))).price
        price_start = group_price.first()
        price_end = group_price.last()
        price_max = group_price.max()
        price_min = group_price.min()
        volume = df_new.groupby(pd.Grouper(key='timestamp', freq=('%s' % unit)))['size'].sum()

        df_g = pd.DataFrame(price_start)
        df_g.rename({'price': 'price_start'}, axis=1, inplace=True)
        df_g['price_end'] = price_end
        df_g['price_max'] = price_max
        df_g['price_min'] = price_min
        df_g['volume'] = volume

        list_kbar.append(df_g)

        print('%.3f | "%s" included.' % ((time.time() - t0), file))

    df_kbar = pd.concat(list_kbar, sort=False, ignore_index=False)

    return df_kbar


def compress_quote(file_path_quote, unit, start_date, end_date, symbol):
    """quote csv dealing: get ticks count for each line(for flipage estimation)"""

    list_ticks = []
    t0 = time.time()

    file_list_quote = get_csv_names(file_path_quote, start_date=start_date, end_date=end_date, cat='BTIMEX.quote')
    for file in file_list_quote:

        df_new = pd.read_csv('%s/%s' % (file_path_quote, file),
                             usecols=['timestamp', 'symbol'])
        if symbol not in list(df_new['symbol']):
            continue
        df_new = df_new[df_new.symbol == symbol]
        df_new.reset_index(drop=True, inplace=True)

        df_new['timestamp'] = bitmex_time_format_change(df_new['timestamp'])
        ticks = df_new.groupby(pd.Grouper(key='timestamp', freq=('%s' % unit))).symbol.count()
        df_ticks = pd.DataFrame(ticks)
        list_ticks.append(df_ticks)

        print('%.3f | "%s" ticks counted.' % ((time.time() - t0), file))

    df_ticks = pd.concat(list_ticks, sort=False, ignore_index=False)

    return df_ticks


def compress_distribute(direct_to, file_path, unit, start_date, end_date, symbol):

    result = None
    if direct_to == 'trade':
        result = compress_trade(file_path, unit, start_date, end_date, symbol)
    elif direct_to == 'quote':
        result = compress_quote(file_path, unit, start_date, end_date, symbol)

    return result


def tick_comp_price(file_path, start_date=None, end_date=None):
    """Get the compressed traded tick price from file_path.

    @param file_path: path that contains all the csv files(each day, no break) with columns:
        timestamp
        price
    @param start_date: str, format: '20150925'
    @param end_date: str, format: '20200101'. Note: end_day will NOT be included.
    @param exchange: support only bitmex, for now
    @return: compressed tick price DataFrame with columns:
        timestamp
        price
    """

    # get all target file names
    file_list = get_csv_names(file_path, start_date=start_date, end_date=end_date)

    list_df = []
    for file in file_list:
        df_add = pd.read_csv('%s/%s' % (file_path, file))
        list_df.append(df_add)

    df_tick_comp_price = pd.concat(list_df, ignore_index=True, sort=False)

    return df_tick_comp_price


def get_tick_comp_price(file_path, to_path=None, start_date=None, end_date=None, exchange='BITMEX', symbol='XBTUSD'):
    """To get the compressed traded tick price from raw trade data in file_path.

    @param file_path: path that contains all the raw trade csv files(support bitmex only, for now).
    @param to_path: path to save csv file. None for not saving.
    @param start_date: str, format: '20150925'
    @param end_date: str, format: '20200101'. Note: end_day will NOT be included.
    @param exchange: support bitmex only, for now.
    @param symbol: trading target
    @return: compressed tick price DataFrame with columns:
        timestamp
        price
    """

    if not exchange == 'BITMEX':
        return

    # close SettingWithCopyWarning
    pd.set_option('mode.chained_assignment', None)

    t0 = time.time()

    list_tick_comp_price = []

    # get all target file names
    file_list = get_csv_names(file_path, start_date=start_date, end_date=end_date)

    # read and compress all the files in file_path
    for file in file_list:

        df = pd.read_csv('%s/%s' % (file_path, file), usecols=['timestamp', 'symbol', 'price', 'tickDirection'])
        if symbol not in list(df['symbol']):
            continue

        df = df[df.symbol == symbol]
        df.reset_index(inplace=True, drop=True)
        df['timestamp'] = bitmex_time_format_change(df['timestamp'])
        df = df[
            (df.tickDirection == 'PlusTick') | (df.tickDirection == 'MinusTick')]  # keep only lines that price changed

        # 仅保留同方向连续吃两笔以上的tick --因为后续 limit order 成交的判断依据是：越过limit价格
        df['tickDirection_old'] = np.append(np.nan, df['tickDirection'][:-1])
        df['keep'] = 1
        df['keep'][df.tickDirection != df.tickDirection_old] = 0
        df.iloc[0, -1] = 1  # keep first line
        df = df[df['keep'] == 1]

        df.drop(['symbol', 'tickDirection', 'tickDirection_old', 'keep'], axis=1, inplace=True)

        list_tick_comp_price.append(df)
        print('%.3f | file "%s" included.' % ((time.time() - t0), file))

        # save csv files to to_path(folder)
        if to_path:
            df.to_csv(('%s/%s' % (to_path, file)), index=False)  # the same file name.
            print('%.3f | file "%s" saved to "%s".' % ((time.time() - t0), file, to_path))

    df_tick_comp_price = pd.concat(list_tick_comp_price)
    df_tick_comp_price.reset_index(drop=True, inplace=True)

    print('%.3f | trade tick price compressed successfully.' % (time.time() - t0))

    # reopen SettingWithCopyWarning
    pd.set_option('mode.chained_assignment', 'warn')

    return df_tick_comp_price
