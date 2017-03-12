import numpy as np
import pandas as pd
import datetime as dt
import math
import time
import util
import matplotlib.pyplot as plt
from indicators import tech_indicators
import marketsim 
from termcolor import colored

class rule_based_strategy:

	def __init__(self,tech_indicators,verbose=False):
		self.tech_indicators = tech_indicators
		self.orders=None
		self.verbose = verbose

	def build_orders(self):
		# oversold, should buy: 
		# sma<0.95 and bbp <0 and rsi<30 and rsi_spy>30 and so<0.20 and (mm bottoms out and starts raising)
		# overbought, should see: 
		# sma>1.05 and bbp >1 and rsi>70 and rsi_spy<70 and so>0.80 and (mm peaks out and starts decreasing)

		# Overall strategy: 
		# use sma, bb, rsi, and so to compute a list of order_advices
		#, where +1 means should buy, 0 stay put, -1 should sell. 

		# Then, based on sma, bb, rsi, and so, we modify order_advice 
		# to be contain overfactors

		# Finally, we loop through order_advice and 
		# based on momentum change AND potential gain to decide
		# whether to turn a particular advice to a real order
		spy_rsi = self.tech_indicators.rsi.copy()
		spy_rsi.values[:,:] = spy_rsi.ix[:,['SPY']]
		allprice = self.tech_indicators.price
		alldates = [date.date() for date in allprice.index]
		sma = self.tech_indicators.sma
		bb = self.tech_indicators.bb 
		rsi = self.tech_indicators.rsi
		mm = self.tech_indicators.mm 
		so = self.tech_indicators.so 
		price = self.tech_indicators.price

		order_advice = price.copy()
		order_advice.ix[:,:] = np.nan

		sma_cross = pd.DataFrame(0, index=sma.index, columns=sma.columns)
		sma_cross[sma >= 1] = 1
		sma_cross[1:] = sma_cross.diff()
		sma_cross.ix[0] = 0
		
		mm_change = mm.copy()
		mm_rightshift1=mm_change.shift(1)
		mm_rightshift2=mm_change.shift(2)

		mm_change[(mm<mm_rightshift1) & (mm_rightshift2<mm_rightshift1)] = 1 # mm out of a peak
		mm_change[(mm>mm_rightshift1) & (mm_rightshift2>mm_rightshift1)] = -1 # mm out of a bottom
		mm_change.fillna(0,inplace=True)
		mm_change[(mm_change!=1) & (mm_change!=-1) & (mm_change!=0)] = 0		

		order_advice[(sma < 1.0) & (bb < 0.2) & \
		(rsi < 30) & (spy_rsi > 30) & (so < 0.20)\
		& (mm_change>0)] = 500 # oversold
		
		order_advice[(sma > 1.0) & (bb > 0.2) & \
		(rsi > 50) & (spy_rsi < 80) & (so > 0.80)\
		&(mm_change<0)] = -500 # overbought

		order_advice[(sma_cross != 0)] = 0

		order_advice.ffill(inplace=True)
		order_advice.fillna(0,inplace=True)
		order_advice[1:] = order_advice.diff()
		order_advice.ix[0] = 0

		self.orders = order_advice
		
		del self.orders['SPY']

		self.orders = self.orders.loc[(self.orders!=0).any(axis=1)]
		self.order_list = []
		counter = 0

		for day in self.orders.index:
			if self.orders.ix[day,'IBM'] > 0:
				if counter>0:
					previous_date = self.order_list[counter-1][0]
					if len(allprice.ix[previous_date:day.date()]) > 10:
						self.order_list.append([day.date(),'IBM','BUY',500])
						counter = counter + 1
				else:
					self.order_list.append([day.date(),'IBM','BUY',500])
					counter = counter + 1

			elif self.orders.ix[day,'IBM'] < 0:
				if counter>0:
					previous_date = self.order_list[counter-1][0]
					if len(allprice.ix[previous_date:day.date()]) > 10:
						self.order_list.append([day.date(),'IBM','SELL',500])
						counter = counter + 1
				else:
					self.order_list.append([day.date(),'IBM','SELL',500])
					counter = counter + 1
		
		# add in the exit action: exit buy by selling, exit sell by buying
		temp_list = []	
		long_entry_dates = []
		short_entry_dates = []
		exit_dates = []
		counter2=0
		
		for order in self.order_list:
			temp_list.append(order)
			prev_date = order[0]
			prev_action = order[2]
			new_date_index = alldates.index(prev_date)+9
			if new_date_index >= len(alldates):
				continue
			newdate = alldates[new_date_index]

			if prev_action == 'BUY':
				long_entry_dates.append(prev_date)
				temp_list.append([newdate,'IBM','SELL',500])
				exit_dates.append(newdate)
			else:
				short_entry_dates.append(prev_date)
				temp_list.append([newdate,'IBM','BUY',500])
				exit_dates.append(newdate)

		self.order_list = temp_list
		self.long_entry_dates = long_entry_dates
		self.short_entry_dates = short_entry_dates
		self.exit_dates = exit_dates
		self.alldates = alldates
		
		if self.verbose:
			print len(self.long_entry_dates)
			print len(self.short_entry_dates)
			print len(self.exit_dates)
	
	def write_to_csv(self,test=False):
		# TODO for TA: Modify the 'base_path' variable before testing: 
		base_path = "/Users/tianchuliang/Documents/GT_Acad/CS7646/ML4T_2016Fall/mc3_p3/"
		if not test:
			fo = open(base_path+"rule_order.csv", "w")
		else: 
			fo = open(base_path+"rule_order_test.csv", "w")
		
		for trade in self.order_list:
			time=trade[0]
			stock=trade[1]
			order=trade[2]
			amount=trade[3]
			string = str(time)+','+str(stock)+','+str(order)+','+str(amount)+'\n'
			fo.write(string)
		
		fo.close()		

	def make_plot(self, symbols, start_date, end_date,start_val):
		remaining_val, benchmark_values = self.benchmark(symbols, start_date, end_date,start_val)
		cr,adr,sddr,sr,ev, daily_vals = self.market_sim()
		benchmark_values =benchmark_values + remaining_val
		# normalize values: 
		benchmark_values = benchmark_values/benchmark_values[0]
		daily_vals = daily_vals/daily_vals[0]

		# ax1 main plot
		# ax2 subplot containing IBM prices and SPY prices
		fig, (ax1, ax2) = plt.subplots(2,1,figsize=(18,9),sharex=True,\
						gridspec_kw = {'height_ratios':[2, 1]})
		ax1.plot(daily_vals,label='Portfolio Values', color='blue')
		ax1.plot(benchmark_values,label='Benchmark Values (IBM)', color='black')
		ymin, ymax = ax1.get_ylim()
		ax1.vlines(x=self.long_entry_dates, ymin=ymin, ymax=ymax, color='g')
		ax1.vlines(x=self.short_entry_dates, ymin=ymin, ymax=ymax, color='r')
		ax1.vlines(x=self.exit_dates,ymin=ymin,ymax=ymax,color='black')

		ax2.plot(self.tech_indicators.price['IBM'],label='IBM stock price', color='magenta')
		ax2.plot(self.tech_indicators.price['SPY'],label='SPY index', color='black')
		ax1.legend(loc='lower right')
		ax1.set_title('Rule Based Strategy')
		ax1.set_ylabel('Normalized value')

		ax2.set_xlabel('Year')
		ax2.legend(loc='lower right')

		print '_________Manual Strategy Stats_________'
		print '| (normalized)'
		print '| cr: ', colored(cr,'blue')
		print '| adr: ', adr
		print '| sddr: ', sddr
		print '| sr: ', sr
		print '| last day portfolio value: ', colored(daily_vals[-1],'blue')
		print '======================================='
		print '| IBM Benchmark: '
		print '| stockprice: ', colored(self.tech_indicators.price['IBM'][-1]/self.tech_indicators.price['IBM'][0] - 1,'magenta')
		print '| portfolio value: ', colored(benchmark_values[-1],'magenta')
		print '======================================='
		return daily_vals

	def make_plot_test(self, symbols, start_date, end_date,start_val):
		remaining_val, benchmark_values = self.benchmark(symbols, start_date, end_date,start_val)
		cr,adr,sddr,sr,ev, daily_vals = self.market_sim(True)
		benchmark_values =benchmark_values + remaining_val
		# normalize values: 
		benchmark_values = benchmark_values/benchmark_values[0]
		daily_vals = daily_vals/daily_vals[0]

		# ax1 main plot
		# ax2 subplot containing IBM prices and SPY prices
		fig, (ax1, ax2) = plt.subplots(2,1,figsize=(18,9),sharex=True,\
						gridspec_kw = {'height_ratios':[2, 1]})
		ax1.plot(daily_vals,label='Portfolio Values', color='blue')
		ax1.plot(benchmark_values,label='Benchmark Values (IBM)', color='black')
		ymin, ymax = ax1.get_ylim()
		ax1.vlines(x=self.long_entry_dates, ymin=ymin, ymax=ymax, color='g')
		ax1.vlines(x=self.short_entry_dates, ymin=ymin, ymax=ymax, color='r')
		ax1.vlines(x=self.exit_dates,ymin=ymin,ymax=ymax,color='black')

		ax2.plot(self.tech_indicators.price['IBM'],label='IBM stock price', color='magenta')
		ax2.plot(self.tech_indicators.price['SPY'],label='SPY index', color='black')
		ax1.legend(loc='lower right')
		ax1.set_title('Rule Based Strategy')
		ax1.set_ylabel('Normalized value')

		ax2.set_xlabel('Year')
		ax2.legend(loc='lower right')

		print '_________Manual Strategy Stats_________'
		print '| (normalized)'
		print '| cr: ', colored(cr,'blue')
		print '| adr: ', adr
		print '| sddr: ', sddr
		print '| sr: ', sr
		print '| last day portfolio value: ', colored(daily_vals[-1],'blue')
		print '======================================='
		print '| IBM Benchmark: '
		print '| stockprice: ', colored(self.tech_indicators.price['IBM'][-1]/self.tech_indicators.price['IBM'][0] - 1,'magenta')
		print '| portfolio value: ', colored(benchmark_values[-1],'magenta')
		print '======================================='

	def market_sim(self,test=False):
		# TODO for TA: Modify the 'base_path' variable before testing: 
		base_path = "/Users/tianchuliang/Documents/GT_Acad/CS7646/ML4T_2016Fall/mc3_p3/"
		if not test: 
			start_date = dt.datetime(2006,01,01)
			end_date = dt.datetime(2009,12,31)
			daily_vals = marketsim.compute_portvals(base_path+"rule_order.csv",start_date,end_date,100000)
		else: 
			start_date = dt.datetime(2010,01,01)
			end_date = dt.datetime(2010,12,31)
			daily_vals = marketsim.compute_portvals(base_path+"rule_order_test.csv",start_date,end_date,100000)

		earlierst_date = self.tech_indicators.price.index[0].date()
		latest_date = self.tech_indicators.price.index[-1].date()
		
		endpt_1 = daily_vals.index[0].date()
		endpt_2 = daily_vals.index[-1].date()
		prices = self.tech_indicators.price.copy()
		first_third = prices.ix[earlierst_date:endpt_1,'IBM']
		last_third = prices.ix[endpt_2:latest_date,'IBM']

		first_third.values[:] = 100000
		last_third.values[:] = daily_vals[-1]

		daily_vals = pd.concat([first_third,daily_vals,last_third])

		rfr = 0.0
		sf = 252.0
		cr, adr, sddr, sr = marketsim.compute_portfolio_stats(daily_vals, rfr, sf)
		ev = daily_vals[-1]
		return cr,adr,sddr,sr,ev, daily_vals
	
	def dump_order_list(self):
		total_holding = 0 
		for i,order in enumerate(self.order_list):
			date = order[0]
			if order[2] == 'BUY':
				total_holding = total_holding + 500 
			elif order[2] == 'SELL':
				total_holding = total_holding - 500 
			if date in self.long_entry_dates:
				print '|',str(date),'		', colored('LONG 500','green')
				print '|                  			Holdings: ', colored(total_holding,'magenta')
			elif date in self.short_entry_dates:
				print '|',str(date),'		', colored('SHORT 500','red')
				print '|                  			Holdings: ', colored(total_holding,'magenta')

			if date in self.exit_dates:
				print '|---',str(date),'		', colored('EXIT','magenta')
				print '|                  			Holdings: ', colored(total_holding,'magenta')
				print '|______________________________________________________'
	
	def check_holdings(self):
		holdings = 0
		holdings_list = []
		for i,order in enumerate(self.order_list):
			print 'Current Holdings: ', holdings
			print 'Order date: ', order[0]
			holdings_list.append(holdings)
			if order[2]=='BUY':
				holdings = holdings + order[3]
			else:
				holdings = holdings - order[3]
		self.holdings_list = holdings_list
	
	def benchmark(self, symbols, start_date, end_date,start_val):
		dates = pd.date_range(start_date,end_date)
		prices = util.get_data(symbols, dates, False)
		prices = prices.dropna()

		remaining_val = start_val - 500 * prices.ix[0,'IBM']
		
		benchmark_values = 500 * prices.ix[:,'IBM']
		
		return remaining_val,benchmark_values


if __name__=="__main__":
# 	print ''
	start_date = dt.datetime(2006,01,01)
	end_date = dt.datetime(2009,12,31)
	symbols = ['IBM']
	lookback=14
	tech_inds= tech_indicators(symbols, start_date, end_date, lookback, False)
	tech_inds.compute_indicators()
	
	rule_based = rule_based_strategy(tech_inds, False)
	rule_based.build_orders()
	rule_based.write_to_csv()
	print 'final portfolio stats:'
	print rule_based.market_sim()[-1]
	# rule_based.make_plot(symbols,start_date,end_date,100000)
