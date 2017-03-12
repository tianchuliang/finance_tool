"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
import copy
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", _start_date=None, _end_date=None, start_val = 1000000):

    orderfile = open(orders_file).readlines()
    start_ = orderfile[1].split(',')[0].split('-')
    end_ = orderfile[-1].split(',')[0].split('-')
    start_date = dt.datetime(int(start_[0]),int(start_[1]),int(start_[2]))
    end_date = dt.datetime(int(end_[0]),int(end_[1]),int(end_[2]))
    
    dates = pd.date_range(start_date,end_date)
    symbols = set([])

    for order in orderfile[1:]:
        symbols.add(order.split(',')[1])
    symbols = list(symbols)

    prices = get_data(symbols, dates, False)
    prices = prices.dropna()
    date_indices = prices.index
    column_names = list(prices.columns.values)
    column_names.append('Cash')
    prices = np.array(prices)
    prices = np.hstack((prices,np.ones((prices.shape[0],1))))
    prices = pd.DataFrame(prices)
    prices.rename(columns = dict(zip(prices.columns.values, np.array(column_names))), inplace=True)
    prices.index = date_indices
    
    template_table = copy.deepcopy(prices)
    
    for column in column_names:
        template_table[column] = 0

    trades = copy.deepcopy(template_table)
    for order in orderfile[1:]:

        order = order.split(',')
        order_date = order[0]
        order_on = order[1]
        order_action = order[2]
        order_amount = order[3]
        if order_action == 'BUY':
            trades.loc[order_date][order_on] = int(order_amount) + trades.loc[order_date][order_on]
            trades.loc[order_date]['Cash'] = -prices.loc[order_date][order_on] * int(order_amount) + trades.loc[order_date]['Cash']
        elif order_action == 'SELL':
            trades.loc[order_date][order_on] = -int(order_amount) + trades.loc[order_date][order_on]
            trades.loc[order_date]['Cash'] = prices.loc[order_date][order_on] * int(order_amount) + trades.loc[order_date]['Cash']

    holdings = copy.deepcopy(template_table)
    cumulative_values = np.zeros(prices.shape[1])
    for i,row in enumerate(trades.iterrows()):
        if i == 0:
            # compute corresponding trade info
            new_trade_info = copy.deepcopy(trades.loc[start_date])
            new_trade_info['Cash'] = start_val + new_trade_info['Cash']
            # compute temporary cumulative_values
            temp_cumulative_values = np.array(new_trade_info * prices.loc[start_date])
            # check if this cumulative values is acceptable
            if check_leverage(temp_cumulative_values) <= 3:
                cumulative_values = temp_cumulative_values
                holdings.loc[start_date] = trades.loc[start_date]
                holdings.loc[start_date]['Cash'] = start_val + trades.loc[start_date]['Cash']
            else: 
                holdings.loc[start_date]['Cash'] = start_val
            previous_date_index = start_date
        else:
            date_index = row[0]
            row = row[1]
            # compute corresponding trade info
            new_trade_info = copy.deepcopy(row)
            # compute temporary cumulative_values
            temp_cumulative_values = cumulative_values + np.array(new_trade_info * prices.loc[date_index])
            # check if this cumulative values is acceptable:
            if check_leverage(temp_cumulative_values) <= 3:
                cumulative_values = temp_cumulative_values
                holdings.loc[date_index] = holdings.loc[previous_date_index] + row
            else:
                holdings.loc[date_index] = holdings.loc[previous_date_index]
            previous_date_index = date_index
    # print holdings
    values = copy.deepcopy(template_table)
    for i,row in enumerate(holdings.iterrows()):
        date_index = row[0]
        values.loc[date_index] = prices.loc[date_index] * holdings.loc[date_index]
    
    daily_vals = values.sum(axis=1)
    return daily_vals
    # rfr = 0.0
    # sf = 252.0
    # cr, adr, sddr, sr = compute_portfolio_stats(daily_vals, rfr, sf)
    # ev = daily_vals[-1]
    # print daily_vals
    # return cr,adr,sddr,sr,ev

def check_leverage(cumulative_values):
    numerator = sum(abs(cumulative_values[:-1]))
    denominator = sum(cumulative_values)
    return numerator/float(denominator)

def compute_portfolio_stats(port_val, rfr = 0.0, sf = 252.0):
    cr = (port_val[-1]/port_val[0]) - 1
    dr = port_val.copy()
    dr[1:] = (port_val[1:]/port_val[:-1].values) -1 
    dr.ix[0] = 0
    dr = dr[1:]
    adr = dr.mean()
    sddr = dr.std()
    sr = np.sqrt(252)*(dr - rfr).mean()/(dr - rfr).std()
    return cr, adr, sddr, sr 

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    print compute_portvals("./testcases_mc2p1/orders-leverage-2.csv")[-1]
