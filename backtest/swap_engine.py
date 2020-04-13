#--coding:utf-8--

import pandas as pd
import empyrical


def get_score(df_traded, df_price, annual_period=(365*24)):
    """Get performance from trading records and prices.

    @param df_traded: trading result, include these columns:
        timestamp  --period time start
        order_value  --fiat value, to detect if traded in this line.
        close_profit  --realized profit for the line, fee counted in
        holds  --holdings(asset) for each line
        balance  --cumsum of returns + holds value(USD) of each line

    @param df_price: 1 hour close price df, includes:
        timestamp  --period time start
        price_end  --period end price

    @param annual_period: How many periods you want to cut one year into.
        Max: 365*24  --for each hour
        day: 365
        stock trading day: 252  # not support yet
        week:  52
        month: 12

    @return: annual score, python dict
    """

    df_traded = df_traded[df_traded.order_value != 0].copy()
    if df_traded.index.name == 'timestamp':
        df_traded.reset_index(inplace=True)
    df_traded['timestamp'] = pd.to_datetime(df_traded.timestamp)
    df_traded.set_index('timestamp', inplace=True)

    if df_price.index.name == 'timestamp':
        df_price.reset_index(inplace=True)
    df_price['timestamp'] = pd.to_datetime(df_price.timestamp)
    df_price.set_index('timestamp', inplace=True)

    # Change into time period summary

    if annual_period == 365*24:
        freq = 'h'
    elif annual_period == 365:
        freq = 'd'
    elif annual_period == 52:
        freq = 'w'
    elif annual_period == 12:
        freq = 'y'
    else:
        hour = 365*24 / annual_period
        freq = '%d' % int(hour)

    group = df_traded.groupby(pd.Grouper(freq=freq))
    df_g = pd.DataFrame()
    df_g['profit'] = group.close_profit.sum()
    df_g['holds_end'] = group.holds.last()
    df_g['holds_yes'] = group.holds.apply(lambda x: sum(abs(x)))
    df_g['price_end'] = df_price.groupby(pd.Grouper(freq=freq)).price_end.last()
    df_g['holds_return'] = df_g['price_end'] * df_g['holds_end']
    df_g['returns'] = df_g['holds_return'] + df_g['profit']
    df_g['balance'] = df_g['holds_return'].cumsum() + df_traded['order_value'].max()
    df_g['returns_ratio'] = df_g['balance'].pct_change()

    df_g.to_csv('get_score_test.csv')

    # scores

    dict_score = {'sortino_ratio': empyrical.sortino_ratio(df_g.returns_ratio, annualization=annual_period),
                  'calmar_ratio': empyrical.calmar_ratio(df_g.returns_ratio, annualization=annual_period),
                  'sharpe_ratio': empyrical.sharpe_ratio(df_g.returns_ratio, annualization=annual_period),
                  'annual_return': empyrical.annual_return(df_g.returns_ratio, annualization=annual_period),
                  'max_drawdown': empyrical.max_drawdown(df_g.returns_ratio),
                  '年化交易次数': df_traded.shape[0] * annual_period / df_g.shape[0],
                  '持仓时间比': df_g[df_g.holds_yes > 0].shape[0] / df_g.shape[0],
                  '盈亏时间比(环比)': df_g[df_g.returns_ratio > 0].shape[0] / df_g[df_g.returns_ratio < 0].shape[0]
                  }

    return dict_score
