# --coding:utf-8--

"""
for bitmex swap, XBTUSD mainly.
"""

import time
import datetime
import os
import pandas as pd
import numpy as np
import empyrical

from backtest.constant import *
from tools import Tools


class Decision(Tools):
    """From each strategy outcome to df_book
    1. Create the big strategy demand DataFrame. Index: timestamp, columns:
        - strategy_IDs
        - each strategy demands: pos_should, price(None for marketprice)
    2. Merge all strategy demands into one df_book.
    """


    def __init__(self, name, structure, data):

        self.name = name



    def make_decision(self):

        pass





class Judge(Tools):
    """Judgement: from df_book to strategy score.
    1. mock trading
    2. portfolio
    2. performance evaluation
    """

    def __init__(self, name, df_book, df_price,
                 judge_type='kbar', save_file=None, adjust='auto',
                 exchange='BITMEX', symbol='XBTUSD'):
        """Create judgement object for a strategy.

        @param df_book: booking DataFrame which must include:
            timestamp  --when the order is submitted. NOTE it's NOT the start(tag) timestamp of the period!
            # exchange
            # symbol
            price  --decision tag price(at end of period)
            order_price  --limit price to trade. 0 for market price.
            pos_should  --what pos should be hold (shortly) after the timestamp
        @param df_price: price info to judging if order in df_book can be filled(traded), should be longer than df_book.
        @param judge_type:
            'kbar'  -- to specify df_price a time based kbar DataFrame, includes:
                timestamp  -- start time of the period. NOTE this is different than timestamp in df_book!
                price_start
                price_end
                price_max
                price_end
                ticks  --how many L1 changes the period has.
            'tick'  -- to specify df_price a tick price DataFrame, from:
                tools.get_tick_comp_price()
                tools.tick_comp_price()

        @param save_file: path & file name to save trading result.
        @param adjust: adjust order_price to: 'auto', 'market', None for not adjusting.

        NOTE:
            'kbar' judgement Assumptions:
                1. OrderType.MARKET
                    - orders can be traded in the next K period.
                    - slippage: according to ticks
                2. OrderType.LIMIT_NEXT
                    - order that can only be traded in the K period, cancelled latter.
                3. OrderType.LIMIT:
                    - orders can be submitted at next kbar
                    - slippage happens only at the first kbar judgement, according to price difference and ticks.
            'tick' judgement:
                Not yet surported.

        """

        print('%s | %s | Creating judgement instance: %s.' % (self.str_time(), name, name))

        self.name = name
        self.df_book = df_book
        self.df_price = df_price
        self.judge_type = judge_type
        self.save_file = save_file
        self.adjust = adjust
        self.symbol = exchange + '.' + symbol

        print('%s | %s | Preparing data.' % (self.str_time(), self.name))

        self.arr_orders = self.arr_orders_prepare(df_book, adjust=adjust)
        self.order_num = 0
        self.order_num_max = self.arr_orders.shape[1]
        self.arr_price = self.arr_price_prepare(df_price)
        self.bar_num = 0
        self.bar_num_max = self.arr_price.shape[1]

        self.bar_line = None
        self.order_book = {}  # {order_ID: {'order_timestamp': , 'order_side': , 'order_price': , 'order_value: }, ...}
        self.trading_record = {'order_ID': [],
                               'timestamp': [],  # traded timestamp
                               'side': [],
                               'price': [],  # traded price
                               'order_value': [],
                               'fee_rate': []
                               }
        self.order_record = {'order_ID': [],
                             'timestamp_kick': [],  # time tick the order
                             'price_kick': [],
                             'order_type': [],  # limit/market...
                             'order_price': [],
                             'traded': []  # if this order get traded
                             }  # TODO: implementation

        print('%s | %s | Instance initialized.' % (self.str_time(), self.name))

    # -----------------------------------------------------------------------------------------------------------------

    def judge_start(self):

        if self.judge_type == 'kbar':
            pass
        elif self.judge_type == 'tick':
            print('Not supported. Code 65498.')
            return  # tick price judgement is precise, but slow. Not yet supported.

        print('%s | %s | Starting mock trading.' % (self.str_time(), self.name))
        self.mock_trading()

        print('%s | %s | Starting evaluation.' % (self.str_time(), self.name))
        self.trading_record.pop('order_ID')  # TODO get_score() overridden
        print(self.trading_record)
        result = self.get_score(self.trading_record, self.df_price, annual_period=365*24*60, save_file=self.save_file)

        print('%s | %s | Judgement instance ended for %s.' % (self.str_time(), self.name, self.name))

        return result

    # -----------------------------------------------------------------------------------------------------------------

    def mock_trading(self):
        """Get the orders traded if possible."""

        t0 = time.time()
        self.bar_line = self.get_next_bar()
        bar_timestamp = self.bar_line[0]  # to make if faster, only timestamp, others latter. same for below.

        while True:

            # get new order data
            order_ID, order_timestamp, order_side, order_price, order_value = self.get_next_order()

            if not order_ID:
                # for no more new orders, deal with old orders(if any) until the end
                while self.order_book:
                    if self.next_bar():
                        self.bar_line = self.get_next_bar()
                        self.trade_old()
                    else:
                        print('%s | %s | NOTE: No more bars while order_book is not empty.' % (self.str_time(), self.name))
                        break  # no more bars
                break  # no old orders not traded when no more new orders/ no more bars

            # get new bar data
            if not self.next_bar():
                break  # no next bar data. this is not suppose to happen as df_price should be longer than df_book

            while bar_timestamp <= order_timestamp:
                # dealing with old limit orders.
                if self.order_book:
                    self.trade_old()

                self.bar_line = self.get_next_bar()
                bar_timestamp = self.bar_line[0]

            # get full bar data
            price_start = self.bar_line[1]
            price_end = self.bar_line[2]
            price_max = self.bar_line[3]
            price_min = self.bar_line[4]
            ticks = self.bar_line[5]

            # dealing with old limit orders.
            if self.order_book:
                self.trade_old()

            # dealing with new orders(limit/market/stop/liquid)
            trade_price, fee_rate = self.trade_kbar(price_start, price_end, price_max, price_min,
                                                    ticks=ticks,
                                                    order_side=order_side,
                                                    order_price=order_price,
                                                    first_time=True)

            if np.isnan(trade_price):
                self.order_book[order_ID] = ({'order_timestamp': order_timestamp,
                                              'order_side': order_side,
                                              'order_price': order_price,
                                              'order_value': order_value
                                              })
            else:
                self.trading_record['order_ID'].append(order_ID)
                self.trading_record['timestamp'].append(bar_timestamp)
                self.trading_record['side'].append(order_side)
                self.trading_record['price'].append(trade_price)
                self.trading_record['order_value'].append(order_value)
                self.trading_record['fee_rate'].append(fee_rate)

        print('%s | %s | %d orders traded in %.3f seconds.' % (
            self.str_time(), self.name, len(self.trading_record['order_ID']), (time.time() - t0)))

    # -----------------------------------------------------------------------------------------------------------------

    def trade_old(self):
        """Judging old limit orders"""

        # get full bar data for old limit order trading
        bar_timestamp = self.bar_line[0]
        price_start = self.bar_line[1]
        price_end = self.bar_line[2]
        price_max = self.bar_line[3]
        price_min = self.bar_line[4]
        ticks = self.bar_line[5]

        order_book = self.order_book.copy()  # important.
        for order_ID_old in order_book:
            # order_timestamp = self.df_book[order_ID]['order_timestamp']
            order_side_old = self.order_book[order_ID_old]['order_side']
            order_price_old = self.order_book[order_ID_old]['order_price']
            order_value_old = self.order_book[order_ID_old]['order_value']

            trade_price, fee_rate = self.trade_kbar(price_start, price_end, price_max, price_min,
                                                    ticks=ticks,
                                                    order_side=order_side_old,
                                                    order_price=order_price_old,
                                                    first_time=False
                                                    )

            if not np.isnan(trade_price):

                self.trading_record['order_ID'].append(order_ID_old)
                self.trading_record['timestamp'].append(bar_timestamp)
                self.trading_record['side'].append(order_side_old)
                self.trading_record['price'].append(trade_price)
                self.trading_record['order_value'].append(order_value_old)
                self.trading_record['fee_rate'].append(fee_rate)

                self.order_book.pop(order_ID_old)

    # -----------------------------------------------------------------------------------------------------------------

    # def get_score(self):
    #     """Get performance from trading records and prices."""

    #     """@note
    #     添加 limit_order 的成功概率、时差，market_order 的滑点情况
    #     """
    #     pass
    #     return

    # -----------------------------------------------------------------------------------------------------------------

    def next_bar(self):
        """To judge if there is next bar."""

        if self.bar_num < self.bar_num_max:
            return True

        return False

    # -----------------------------------------------------------------------------------------------------------------

    def get_next_bar(self):
        """get next bar data
        @return: bar_timestamp, price_start, price_end, price_max, price_min, ticks"""

        bar_line = self.arr_price[..., self.bar_num]
        self.bar_num += 1

        return bar_line

    # -----------------------------------------------------------------------------------------------------------------

    def next_order(self):
        """To judge if there is next new order."""

        if self.order_num < self.order_num_max:
            return True

        return False

    # -----------------------------------------------------------------------------------------------------------------

    def get_next_order(self):
        """get next order data
        @return: order_ID, order_timestamp, order_side, order_price, order_value"""

        if not self.order_num < self.order_num_max:
            return None, np.nan, np.nan, np.nan, np.nan

        line = self.arr_orders[..., self.order_num]

        order_timestamp = line[0]
        order_side = line[1]
        order_price = line[2]
        order_value = line[3]
        order_ID = self.symbol + '.' + str(order_timestamp)  # as no order has the same timestamp.

        self.order_num += 1

        return order_ID, order_timestamp, order_side, order_price, order_value



if __name__ == '__main__':
    # 准备虚拟的交易记录字典
    df1 = pd.read_csv('../data/trading_record_example.csv')
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

    # test it.

    i = Judge('test', df1, df_price)
    i.judge_start()
