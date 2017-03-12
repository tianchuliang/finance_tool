import numpy as np
import pandas as pd
import datetime as dt
import math
import time
import util
import matplotlib.pyplot as plt
from indicators import tech_indicators
import marketsim 
from RTLearner import RTLearner
from termcolor import colored

class ML_based_strategy:

	def __init__(self,tech_indicators,learner, ybuy=0, ysell=0, verbose=False):
		self.tech_indicators = tech_indicators
		self.verbose = verbose
		self.ybuy=ybuy
		self.ysell=ysell
		self.X = None
		self.Y = None 
		self.learner = learner
		self.prices = self.tech_indicators.price['IBM']
		
	def gen_X(self):
		so = self.tech_indicators.so['IBM'].values
		sma = self.tech_indicators.sma['IBM'].values
		mm = self.tech_indicators.mm['IBM'].values
		rsi = self.tech_indicators.rsi['IBM'].values
		bb = self.tech_indicators.bb['IBM'].values

		so.shape=(so.shape[0],1)
		sma.shape=(sma.shape[0],1)
		mm.shape=(mm.shape[0],1)
		rsi.shape=(rsi.shape[0],1)
		bb.shape=(bb.shape[0],1)

		self.X = np.hstack((so,sma))
		self.X = np.hstack((self.X,mm))
		self.X = np.hstack((self.X,rsi))
		self.X = np.hstack((self.X,bb))
	
	def gen_Y(self):
		self.Y = self.prices.copy()
		self.Y[:-10]=0
		ret = self.prices.copy()
		more = self.prices.copy()
		less = self.prices.copy()

		ret.values[:-10] = self.prices.values[10:]/self.prices.values[:-10] - 1 
		
		more[:-10] = (ret[:-10] > self.ybuy) * 1
		
		less[:-10] = (ret[:-10] < self.ysell) * -1
		more[-10:] = 0 
		less[-10:] = 0
		self.Y = more + less
		self.Y = self.Y.values

	def train(self):
		np.random.seed(5)
		self.learner.leaf_size=13
		self.learner.addEvidence(self.X,self.Y)

	def gen_order(self, testX):
		return self.learner.query(testX)

	def gen_test_data(self):
		start_date = dt.datetime(2010,01,01)
		end_date = dt.datetime(2010,12,31)
		symbols = ['IBM']
		lookback=14
		tech_inds= tech_indicators(symbols, start_date, end_date, lookback, False)
		tech_inds.compute_indicators()
		so = tech_inds.so['IBM'].values
		sma = tech_inds.sma['IBM'].values
		mm = tech_inds.mm['IBM'].values
		rsi = tech_inds.rsi['IBM'].values
		bb = tech_inds.bb['IBM'].values

		so.shape=(so.shape[0],1)
		sma.shape=(sma.shape[0],1)
		mm.shape=(mm.shape[0],1)
		rsi.shape=(rsi.shape[0],1)
		bb.shape=(bb.shape[0],1)

		testX = np.hstack((so,sma))
		testX = np.hstack((testX,mm))
		testX = np.hstack((testX,rsi))
		testX = np.hstack((testX,bb))
		
		return testX

	def build_orders(self,test=False):

		if not test:
			allprice = self.tech_indicators.price
			alldates = [date.date() for date in allprice.index]
			self.orders = self.prices.copy()
			self.orders.values[:14]=0
			self.orders.values[14:]= self.gen_order(self.X[14:])

		else: 
			testX = self.gen_test_data()
			start_date_test = dt.datetime(2010,01,01)
			end_date_test = dt.datetime(2010,12,31)
			dates_test = pd.date_range(start_date_test, end_date_test)
			price_test = util.get_data(['IBM'], dates_test)
			price_test =price_test['IBM']
			allprice = price_test
			alldates = [date.date() for date in allprice.index]			
			self.orders = price_test.copy()
			self.orders.values[:14]=0
			self.orders.values[14:] = self.gen_order(testX[14:])

		# assemble order_list from self.orders
		self.orders.values[:] = self.orders.values[:] * 500 
		self.orders = self.orders[14:]
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

		return 0

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

	def write_to_csv(self,test=False):
		# TODO for TA: Modify the 'base_path' variable before testing: 
		base_path = "/Users/tianchuliang/Documents/GT_Acad/CS7646/ML4T_2016Fall/mc3_p3/"
		if not test:
			fo = open(base_path+"ml_order.csv", "w")
		else:
			fo = open(base_path+"ml_order_test.csv", "w")
		
		for trade in self.order_list:
			time=trade[0]
			stock=trade[1]
			order=trade[2]
			amount=trade[3]
			string = str(time)+','+str(stock)+','+str(order)+','+str(amount)+'\n'
			fo.write(string)
		fo.close()

	def benchmark(self, symbols, start_date, end_date,start_val):
		dates = pd.date_range(start_date,end_date)
		prices = util.get_data(symbols, dates, False)
		prices = prices.dropna()

		remaining_val = start_val - 500 * prices.ix[0,'IBM']
		
		benchmark_values = 500 * prices.ix[:,'IBM']
		
		return remaining_val,benchmark_values

	def make_plot(self, symbols, start_date, end_date,start_val, rule_based_values=None):
		remaining_val, benchmark_values = self.benchmark(symbols, start_date, end_date,start_val)
		cr,adr,sddr,sr,ev, daily_vals = self.market_sim(test=False)
		benchmark_values =benchmark_values + remaining_val
		# normalize values: 
		benchmark_values = benchmark_values/benchmark_values[0]
		daily_vals = daily_vals/daily_vals[0]

		# ax1 main plot
		# ax2 subplot containing IBM prices and SPY prices
		fig, (ax1, ax2) = plt.subplots(2,1,figsize=(25,9),sharex=True,\
						gridspec_kw = {'height_ratios':[2, 1]})
		ax1.plot(daily_vals,label='Portfolio Values', color='green')
		ax1.plot(benchmark_values,label='Benchmark Values (IBM)', color='black')
		ymin, ymax = ax1.get_ylim()
		ax1.plot(rule_based_values, label='Portfolio Values (Rule Based)', color='blue')
		ax1.vlines(x=self.long_entry_dates, ymin=ymin, ymax=ymax, color='g')
		ax1.vlines(x=self.short_entry_dates, ymin=ymin, ymax=ymax, color='r')
		ax1.vlines(x=self.exit_dates,ymin=ymin,ymax=ymax,color='black')

		ax2.plot(self.tech_indicators.price['IBM'],label='IBM stock price', color='magenta')
		ax2.plot(self.tech_indicators.price['SPY'],label='SPY index', color='black')
		ax1.legend(loc='lower right')
		ax1.set_title('Machine Learning Based Strategy')
		ax1.set_ylabel('Normalized value')

		ax2.set_xlabel('Year')
		ax2.legend(loc='lower right')

		print '___________ML Strategy Stats___________'
		print '| (normalized)'
		print '| cr: ', colored(cr,'green')
		print '| adr: ', adr
		print '| sddr: ', sddr
		print '| sr: ', sr
		print '| last day portfolio value: ', colored(daily_vals[-1],'green')
		print '======================================='
		print '| IBM Benchmark: '
		print '| stockprice: ', colored(self.tech_indicators.price['IBM'][-1]/self.tech_indicators.price['IBM'][0] - 1,'magenta')
		print '| portfolio value: ', colored(benchmark_values[-1],'magenta')
		print '======================================='

	def make_plot_test(self, symbols, start_date, end_date,start_val):
		remaining_val, benchmark_values = self.benchmark(symbols, start_date, end_date,start_val)
		cr,adr,sddr,sr,ev, daily_vals = self.market_sim(test=True)
		benchmark_values =benchmark_values + remaining_val
		# normalize values: 
		benchmark_values = benchmark_values/benchmark_values[0]
		daily_vals = daily_vals/daily_vals[0]
		start_date_test = dt.datetime(2010,01,01)
		end_date_test = dt.datetime(2010,12,31)
		dates_test = pd.date_range(start_date_test, end_date_test)
		price_test = util.get_data(['IBM'], dates_test)
		
		# ax1 main plot
		# ax2 subplot containing IBM prices and SPY prices
		fig, (ax1, ax2) = plt.subplots(2,1,figsize=(25,9),sharex=True,\
						gridspec_kw = {'height_ratios':[2, 1]})
		ax1.plot(daily_vals,label='Portfolio Values', color='blue')
		ax1.plot(benchmark_values,label='Benchmark Values (IBM)', color='black')
		ymin, ymax = ax1.get_ylim()
		ax1.vlines(x=self.long_entry_dates, ymin=ymin, ymax=ymax, color='g')
		ax1.vlines(x=self.short_entry_dates, ymin=ymin, ymax=ymax, color='r')
		ax1.vlines(x=self.exit_dates,ymin=ymin,ymax=ymax,color='black')

		ax2.plot(price_test['IBM'],label='IBM stock price', color='magenta')
		ax2.plot(price_test['SPY'],label='SPY index', color='black')
		ax1.legend(loc='lower right')
		ax1.set_title('Machine Learning Based Strategy')
		ax1.set_ylabel('Normalized value')

		ax2.set_xlabel('Year')
		ax2.legend(loc='lower right')
		print '___________ML Strategy Stats___________'
		print '| (normalized)'
		print '| cr: ', colored(cr,'green')
		print '| adr: ', adr
		print '| sddr: ', sddr
		print '| sr: ', sr
		print '| last day portfolio value: ', colored(daily_vals[-1],'green')
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
			daily_vals = marketsim.compute_portvals(base_path+"ml_order.csv",start_date,end_date,100000)
		else:
			start_date = dt.datetime(2010,01,01)
			end_date = dt.datetime(2010,12,31)
			daily_vals = marketsim.compute_portvals(base_path+"ml_order_test.csv",start_date,end_date,100000)
		
		if not test:
			earlierst_date = self.tech_indicators.price.index[0].date()
			latest_date = self.tech_indicators.price.index[-1].date()
			prices = self.tech_indicators.price.copy()
		else:
			start_date_test = dt.datetime(2010,01,01)
			end_date_test = dt.datetime(2010,12,31)
			dates_test = pd.date_range(start_date_test, end_date_test)
			price_test = util.get_data(['IBM'], dates_test)
			allprice = price_test

			earlierst_date = allprice.index[0].date()
			latest_date = allprice.index[-1].date()
			prices = allprice

		endpt_1 = daily_vals.index[0].date()
		endpt_2 = daily_vals.index[-1].date()

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

# if __name__=="__main__":
# 	start_date = dt.datetime(2006,01,01)
# 	end_date = dt.datetime(2009,12,31)
# 	symbols = ['IBM']
# 	lookback=14
# 	tech_inds= tech_indicators(symbols, start_date, end_date, lookback, False)
# 	tech_inds.compute_indicators()

# 	# ML = ML_based_strategy(tech_inds, RTLearner(verbose=False))

# 	# ML.gen_Y()
# 	# ML.gen_X()
# 	# ML.train()
# 	# ML.build_orders()
# 	# ML.write_to_csv()
# 	# print ML.market_sim()
# 	start_date_test = dt.datetime(2010,01,01)
# 	end_date_test = dt.datetime(2010,12,31)
# 	ML_test = ML_based_strategy(tech_inds, RTLearner(verbose=False))
# 	ML_test.gen_Y()
# 	ML_test.gen_X()
# 	ML_test.train()
# 	ML_test.build_orders(test=True)
# 	ML_test.write_to_csv(test=True)
# 	ML_test.market_sim(test=True)