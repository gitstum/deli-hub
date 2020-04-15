#--coding:utf-8--
import time

import pandas as pd
import numpy as np
import empyrical

from backtest.constant import Direction


def get_score(trading_record, df_price, capital=1000, annual_period=(365*24), save_path=None):
    """Get performance from trading records and prices.

    @param trading_record: Dict of traded result(and no other like cancelled), must include these keys:
        timestamp  --of the event
        side  --order direction
        price  --object price in fiat
        traded_value  --fiat value, volume in fiat, to detect if traded in this line.
        fee_rate  --in float

    @param df_price: 1 hour close price df, includes:
        timestamp  --period time start
        price  --period end price

    @param annual_period: How many periods you want to cut one year into.
        Max: 365*24  --for each hour
        day: 365
        week:  52
        month: 12
    @param save_path: to store results on disk
    @return: annual score, python dict
    """

    pd.set_option('mode.chained_assignment', None)  # 关闭 SettingWithCopyWarning 警告

    # 1. 生成交易记录df_traded

    df_traded = pd.DataFrame(trading_record)
    if df_traded.shape[0] == 0:
        print('No trading orders recorded.')
        return bad_score()
    df_traded['timestamp'] = pd.to_datetime(df_traded.timestamp)

    # 2.合成运算所需df

    df_price['timestamp'] = pd.to_datetime(df_price.timestamp)
    # 让groupby的last()方法找到对应到price_end：
    df_price['timestamp'] += pd.to_timedelta(1, unit='h') - pd.to_timedelta(1, unit='ns')

    start_time = df_traded.timestamp.min() - pd.to_timedelta(1, unit='h')
    end_time = df_traded.timestamp.max() + pd.to_timedelta(1, unit='h')
    df_price_pare = df_price[(df_price.timestamp > start_time) & (df_price.timestamp < end_time)]

    df = pd.concat([df_traded, df_price_pare], sort=False, ignore_index=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.pos.fillna(method='ffill', inplace=True)
    df.avg_price.fillna(method='ffill', inplace=True)
    df.traded_vol_all.fillna(method='ffill', inplace=True)

    # 3. 更换计算单位

    df['re_direction'] = np.nan
    df['re_direction'][df.side == Direction.SHORT.value] = 1
    df['re_direction'][df.side == Direction.LONG.value] = -1
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
                    re_avg_price = ((re_pos - abs(volume)) * re_avg_price \
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
                    re_avg_price = ((abs(re_pos) - abs(volume)) * re_avg_price \
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
        return bad_score()  # lost all.

    # 合成基于固定时间单位的表格
    if annual_period == 365*24:
        freq = 'h'
    elif annual_period == 365:
        freq = 'd'
    else:
        hour = 365*24 / annual_period  # valid for time period within one day
        freq = '%dh' % int(hour)

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


if __name__ == '__main__':

    # 准备虚拟的交易记录字典
    df1 = pd.read_csv('../data/trading_record.csv')
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
    df_price = pd.read_csv('../data/bitmex_price_1hour_2020q1.csv', usecols=['timestamp', 'price_end'])
    df_price.rename({'price_end': 'price'}, axis=1, inplace=True)

    # TEST IT

    t0 = time.time()

    score = get_score(trading_record,
                      df_price,
                      capital=1000,
                      annual_period=(365 * 24),
                      save_path='trading_record_test(after).csv'
                      )
    print(score)

    print('time: %s' % (time.time() - t0))
