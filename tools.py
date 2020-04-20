import os
import time
import datetime
import pandas as pd
import numpy as np
import empyrical

import multiprocessing as mp

from backtest.constant import *


class Tools(object):

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def agg_cal(func, *args, process_num=None):
        """Multi-processing framework for functions.

        How to call:
            result1, result2 = func.agg_cal((param1,), (param2,))
            result1, result2 = Tools.agg_cal(func,
                                             (param1, ), (param2),
                                             process_num=4
                                            )
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

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def str_time():
        now = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        return now

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_score(trading_record, df_kbar, capital=CAPITAL, annual_period=ANNUAL_PERIOD, save_path=None):
        """Get performance from trading records and prices.

        @param trading_record: Dict of traded result(and no other like cancelled), must include these keys:
            timestamp  --of the event
            side  --order direction
            price  --object price in fiat
            order_value  --fiat value, volume in fiat, to detect if traded in this line.
            fee_rate  --in float

        @param df_kbar: 1 minute / 1 hour close price df, includes:
            timestamp  --period time start
            price  --period end price

        @param annual_period: How many periods you want to cut one year into.
            Max: 365*24*60
            hour: 365*24
            day: 365
            week:  52
            month: 12
        @param save_path: to store results on disk.  --NOTE: file's max time period is equal to df_kbar!
        @return: annual score, python dict
        """

        pd.set_option('mode.chained_assignment', None)  # 关闭 SettingWithCopyWarning 警告

        # 1. 生成交易记录df_traded

        df_traded = pd.DataFrame(trading_record)
        if df_traded.shape[0] == 0:
            print('No trading orders recorded.')
            return Tools.bad_score()
        df_traded['timestamp'] = pd.to_datetime(df_traded.timestamp)

        # 2.合成运算所需df

        df_kbar['timestamp'] = pd.to_datetime(df_kbar.timestamp)
        timedelta = df_kbar.loc[1, 'timestamp'] - df_kbar.loc[0, 'timestamp']
        # 让groupby的last()方法找到对应到price_end：
        df_kbar['timestamp'] += timedelta - pd.to_timedelta(1, unit='ns')

        start_time = df_traded.timestamp.min() - timedelta
        end_time = df_traded.timestamp.max() + timedelta
        df_kbar_pare = df_kbar[(df_kbar.timestamp > start_time) & (df_kbar.timestamp < end_time)]

        df = pd.concat([df_traded, df_kbar_pare], sort=False, ignore_index=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        # df.pos.fillna(method='ffill', inplace=True)
        # df.avg_price.fillna(method='ffill', inplace=True)
        # df.traded_vol_all.fillna(method='ffill', inplace=True)

        # 3. 更换计算单位

        df['re_direction'] = np.nan
        df['re_direction'][df.side == Direction.SHORT] = 1
        df['re_direction'][df.side == Direction.LONG] = -1
        df['re_price'] = 1 / df.price
        df['re_size'] = df.order_value * df.re_price

        # 4. 遍历

        re_pos = 0
        re_pos_old = 0
        re_avg_price = 0
        re_avg_price_old = 0
        list_re_pos = []  # 时刻持仓
        list_re_avg_price = []  # 时刻均价
        close_profit_list = []  # 时刻平仓获利

        arr_temp = np.array([
            df['re_direction'],
            df['re_price'],
            df['order_value']  # 注意这里是用order_value
        ])

        num = 0
        while num < arr_temp.shape[1]:

            data = arr_temp[..., num]
            re_direction = data[0]
            price = data[1]
            volume = data[2]

            re_avg_price_old = re_avg_price  # 注意要写在循环开头（右侧更新前）
            re_pos_old = re_pos

            # 计算均价、持仓 (成熟的算法)
            if re_direction == 1:
                re_pos += abs(volume)
                if re_pos > 0:
                    if re_pos > volume:
                        # 当前订单同向影响了均价
                        re_avg_price = ((re_pos - abs(volume)) * re_avg_price
                                        + volume * price) / re_pos
                    else:
                        # 当前订单导致仓位发生正负变化(或变为非0)
                        re_avg_price = price
                elif re_pos == 0:
                    # 当前订单导致仓位归零
                    re_avg_price = 0
                else:
                    # 当前订单给空单减仓，均价未变化
                    pass
            elif re_direction == -1:
                re_pos -= abs(volume)
                if re_pos < 0:
                    if abs(re_pos) > abs(volume):
                        re_avg_price = ((abs(re_pos) - abs(volume)) * re_avg_price
                                        + abs(volume) * price) / abs(re_pos)
                    else:
                        re_avg_price = price
                elif re_pos == 0:
                    re_avg_price = 0
                else:
                    pass

            # 平仓利润计算
            if re_pos * re_pos_old <= 0:
                if re_pos_old == 0:
                    # 此时是全新开仓
                    close_profit = 0
                else:
                    # 此时是平掉了一个方向的全部仓位
                    close_profit = re_pos_old * (price - re_avg_price_old)  # 这里有修改
            else:
                if abs(re_pos) < abs(re_pos_old):
                    # 此时是部分平仓
                    if re_pos > 0:
                        close_profit = - (re_pos - re_pos_old) * (price - re_avg_price_old)  # 这里的符号曾经错了
                    else:
                        close_profit = - abs(re_pos - re_pos_old) * (price - re_avg_price_old)
                else:
                    # 此时是单向加仓
                    close_profit = 0

            list_re_pos.append(re_pos)
            list_re_avg_price.append(re_avg_price)
            close_profit_list.append(close_profit)

            num += 1

        # 5. 向量化运算赋值

        df['re_pos'] = list_re_pos
        df['re_avg_price'] = list_re_avg_price
        df['re_avg_price'][df['re_avg_price'] == 0] = np.nan

        df['re_hold_return'] = df['re_pos'] * (df['re_price'] - df['re_avg_price'])
        df['re_hold_return'].fillna(0, inplace=True)
        df['holdings_value'] = df['re_hold_return'] * df['price']

        df['re_close'] = close_profit_list
        df['close_accumulate'] = df['re_close'].cumsum()

        df['re_fee'] = df['re_size'] * df['fee_rate']
        df['re_fee'].fillna(0, inplace=True)
        df['re_profit'] = df['re_close'] + df['re_fee']

        df['fee'] = df['re_fee'] * df['price']  # 这里对上面fee的单位做了转换。假设套保。
        df['fee_acc'] = df['fee'].cumsum()
        df['close_profit'] = df['re_close'] * df['price']  # 这里假设对每一笔交易的平仓后利润进行套保
        df['profit_acc'] = df['close_profit'].cumsum()

        df['re_returns'] = df['re_close'] + df['re_fee'] + df['re_hold_return']
        df['returns'] = df['close_profit'] + df['fee'] + df['holdings_value']

        df['re_returns_acc'] = df['re_close'].cumsum() + df['re_fee'].cumsum() + df['re_hold_return']
        df['returns_acc'] = df['profit_acc'] + df['fee_acc'] + df['holdings_value']

        # 6. 结果评估

        if df['returns_acc'].min() < - capital:
            return Tools.bad_score()  # lost all.

        # 合成基于固定时间单位的表格
        if annual_period == 365 * 24 * 60:
            freq = 't'
        elif annual_period == 365 * 24:
            freq = 'h'
        elif annual_period == 365:
            freq = 'd'
        else:
            minutes = 365 * 24 * 60 / annual_period  # valid for time period within one day
            freq = '%dt' % int(minutes)

        period_balance = df.groupby(pd.Grouper(key='timestamp', freq=freq)).returns_acc.last() + capital
        period_returns_ratio = period_balance.pct_change()  # key input.
        holds_yes = df.groupby(pd.Grouper(key='timestamp', freq=freq)).re_avg_price.sum()
        period_num = period_returns_ratio.shape[0] - 1
        annual_orders = df_traded.shape[0] * annual_period / period_num
        annual_volumes = df_traded.order_value.sum() * annual_period / period_num
        hold_time_ratio = holds_yes[holds_yes > 0].count() / period_num
        period_profit = period_returns_ratio[period_returns_ratio > 0].count()
        period_loss = period_returns_ratio[period_returns_ratio < 0].count()
        profit_time_ratio = period_profit / (period_profit + period_loss)

        dict_score = {'Sortino_ratio': empyrical.sortino_ratio(period_returns_ratio, annualization=annual_period),
                      'Calmar_ratio': empyrical.calmar_ratio(period_returns_ratio, annualization=annual_period),
                      'Annual_return': empyrical.annual_return(period_returns_ratio, annualization=annual_period),
                      'Max_drawdown': empyrical.max_drawdown(period_returns_ratio),
                      'Sharpe_ratio': empyrical.sharpe_ratio(period_returns_ratio, annualization=annual_period),
                      '年化交易次数': annual_orders,
                      '年化成交量': annual_volumes,
                      '持仓时间比': hold_time_ratio,
                      '浮盈时间比(如持仓)': profit_time_ratio
                      }

        # 7. save
        if save_path:
            df.to_csv('%s' % save_path, index=False)

        pd.set_option('mode.chained_assignment', 'warn')  # 重新打开警告

        return dict_score

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def bad_score():
        """Returns a very bad score, for:

        1. No trading record
        2. Results be so bad as to cause calculation issue.
        """

        dict_score = {'Sortino_ratio': -888,
                      'Calmar_ratio': -888,
                      'Annual_return': -888,
                      'Max_drawdown': -888,
                      'Sharpe_ratio': -888,
                      '年化交易次数': 0,
                      '年化成交量': 0,
                      '持仓时间比': 0,
                      '浮盈时间比(如持仓)': 0
                      }

        return dict_score

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
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

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def bitmex_time_format_change(bitmex_time):
        """Change the unfriendly bitmex time format!

        @param bitmex_time: original timestamp Series from bitmex trade/quote data
        @return: datetime64 timestamp Series.
        """

        bitmex_time = bitmex_time.str[:10] + ' ' + bitmex_time.str[11:]  # change the unfriendly time format!
        timestamp_series = pd.to_datetime(bitmex_time)

        return timestamp_series

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_kbar(df_kbar=pd.DataFrame(), data_path=None, unit='5t'):
        """To get the K price DataFrame from a smaller unit K price csv.

        @param df_kbar: a smaller unit K price DataFrame
        @param data_path: a smaller unit K price csv
        @param unit: pd.groupby() unit
        @return: kbar DataFrame
        """

        if not df_kbar.empty:
            if df_kbar.index.name == 'timestamp':
                df_kbar.reset_index(inplace=True)
        else:
            if data_path:
                df_kbar = pd.read_csv(data_path)
            else:
                print('Either df_kbar or data_path is required.')
                return

        df_kbar['timestamp'] = pd.to_datetime(df_kbar.timestamp)
        df = df_kbar.groupby(pd.Grouper(key='timestamp', freq=unit)).agg({'price_start': 'first',
                                                                          'price_end': 'last',
                                                                          'price_max': 'max',
                                                                          'price_min': 'min',
                                                                          'volume': 'sum',
                                                                          'ticks': 'sum'
                                                                          })

        return df

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_kbar_from_tick(file_path_trade, file_path_quote, unit, to_path=None, to_name=None, start_date=None,
                           end_date=None, exchange='BITMEX', symbol='XBTUSD'):
        """To get the K price DataFrame from tick price. Bitmex only. NO more than 1 day

        @param file_path_trade: path that contains the trade csv files(fromt bitmex)
        @param file_path_quote: path that contains the quote csv files(fromt bitmex)
        @param unit: compress unit: example: '30s', 't', '15t', 'h', 'd'. No more than 1 Day.
        @param to_path: path to save kbar csv file
        @param to_name: csv file name
        @param start_date: str, format: '20150925'
        @param end_date: str, format: '20200101'. Note: end_day will NOT be included.
        @param exchange: exchange
        @param symbol: symbol
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
        df_kbar, df_ticks = Tools.agg_cal(Tools.compress_distribute,
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

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def compress_trade(file_path_trade, unit, start_date, end_date, symbol):
        """trade csv dealing: get basic price info"""

        list_kbar = []
        t0 = time.time()

        file_list_trade = Tools.get_csv_names(file_path_trade, start_date=start_date, end_date=end_date, cat='BITMEX.trade')
        for file in file_list_trade:

            df_new = pd.read_csv('%s/%s' % (file_path_trade, file),
                                 usecols=['timestamp', 'symbol', 'side', 'size', 'price'])
            if symbol not in list(df_new['symbol']):
                continue
            df_new = df_new[df_new.symbol == symbol]
            df_new.reset_index(drop=True, inplace=True)

            df_new['timestamp'] = Tools.bitmex_time_format_change(df_new['timestamp'])
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

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def compress_quote(file_path_quote, unit, start_date, end_date, symbol):
        """quote csv dealing: get ticks count for each line(for flipage estimation)"""

        list_ticks = []
        t0 = time.time()

        file_list_quote = Tools.get_csv_names(file_path_quote, start_date=start_date, end_date=end_date, cat='BITMEX.quote')
        for file in file_list_quote:

            df_new = pd.read_csv('%s/%s' % (file_path_quote, file),
                                 usecols=['timestamp', 'symbol'])
            if symbol not in list(df_new['symbol']):
                continue
            df_new = df_new[df_new.symbol == symbol]
            df_new.reset_index(drop=True, inplace=True)

            df_new['timestamp'] = Tools.bitmex_time_format_change(df_new['timestamp'])
            ticks = df_new.groupby(pd.Grouper(key='timestamp', freq=('%s' % unit))).symbol.count()
            df_ticks = pd.DataFrame(ticks)
            list_ticks.append(df_ticks)

            print('%.3f | "%s" ticks counted.' % ((time.time() - t0), file))

        df_ticks = pd.concat(list_ticks, sort=False, ignore_index=False)

        return df_ticks

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def compress_distribute(direct_to, file_path, unit, start_date, end_date, symbol):
        result = None
        if direct_to == 'trade':
            result = Tools.compress_trade(file_path, unit, start_date, end_date, symbol)
        elif direct_to == 'quote':
            result = Tools.compress_quote(file_path, unit, start_date, end_date, symbol)

        return result

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
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
        file_list = Tools.get_csv_names(file_path, start_date=start_date, end_date=end_date)

        list_df = []
        for file in file_list:
            df_add = pd.read_csv('%s/%s' % (file_path, file))
            list_df.append(df_add)

        df_tick_comp_price = pd.concat(list_df, ignore_index=True, sort=False)

        return df_tick_comp_price

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_tick_comp_price(file_path, to_path=None, start_date=None, end_date=None, exchange='BITMEX',
                            symbol='XBTUSD'):
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
        file_list = Tools.get_csv_names(file_path, start_date=start_date, end_date=end_date)

        # read and compress all the files in file_path
        for file in file_list:

            df = pd.read_csv('%s/%s' % (file_path, file), usecols=['timestamp', 'symbol', 'price', 'tickDirection'])
            if symbol not in list(df['symbol']):
                continue

            df = df[df.symbol == symbol]
            df.reset_index(inplace=True, drop=True)
            df['timestamp'] = Tools.bitmex_time_format_change(df['timestamp'])
            df = df[
                (df.tickDirection == 'PlusTick') | (
                            df.tickDirection == 'MinusTick')]  # keep only lines that price changed

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


if __name__ == '__main__':

    # 准备虚拟的交易记录字典
    df1 = pd.read_csv('data/trading_record_example.csv')
    trading_record = {
        'timestamp': list(df1.timestamp.values),
        'side': list(df1.side.values),
        'price': list(df1.price.values),
        'order_value': list(df1.order_value.values),
        'fee_rate': list(df1.fee_rate.values),
        'pos': list(df1.pos.values),
        'avg_price': list(df1.avg_price.values),
        'traded_vol_all': list(df1.traded_vol_all)
    }

    # 准备历史记录df
    df_price = pd.read_csv('data/bitmex_price_1hour_2020q1.csv', usecols=['timestamp', 'price_end'])
    df_price.rename({'price_end': 'price'}, axis=1, inplace=True)

    # TEST IT

    t0 = time.time()

    score = Tools.get_score(trading_record,
                            df_price,
                            capital=1000,
                            annual_period=(365 * 24),
                            # save_path='trading_record_test(after).csv'
                            )
    print(score)

    print('time: %s' % (time.time() - t0))