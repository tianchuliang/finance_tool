import numpy as np
import pandas as pd
import datetime as dt
import math
import time
import util
import matplotlib.pyplot as plt

# Four Indicators:

class tech_indicators:

	def __init__(self, symbols, start_date, end_date, lookback, verbose=False):
		self.dates = pd.date_range(start_date, end_date)
		self.price = util.get_data(symbols, self.dates)
		self.lookback = lookback
		self.start_date = start_date
		self.end_date = end_date
		self.symbols = symbols
		self.symbols.append('SPY')
		self.verbose = verbose

	def compute_indicators(self):
		# sma, bb
		self.sma_ = self.price.rolling(window=self.lookback,min_periods=self.lookback).mean()
		rolling_std = self.price.rolling(window=self.lookback,min_periods=self.lookback).std()
		self.top_band = self.sma_ + (2 * rolling_std)
		self.bot_band = self.sma_ - (2 * rolling_std)
		self.bb = (self.price - self.bot_band)/(self.top_band - self.bot_band)
		self.sma = self.price / self.sma_

		# rsi
		rs = self.price.copy()
		self.rsi = self.price.copy()
		daily_rets = self.price.copy()
		daily_rets.values[1:,:] = self.price.values[1:,:] - self.price.values[:-1,:]
		daily_rets.values[0,:] = np.nan

		up_rets = daily_rets[daily_rets>=0].fillna(0).cumsum()
		do_rets = -1 * daily_rets[daily_rets<0].fillna(0).cumsum()
		up_gain = self.price.copy()
		up_gain.ix[:,:] = 0
		up_gain.values[self.lookback:,:] = up_rets.values[self.lookback:,:]\
									-up_rets.values[:-self.lookback,:]
		do_loss = self.price.copy()
		do_loss.ix[:,:] = 0
		do_loss.values[self.lookback:,:] = do_rets.values[self.lookback:,:]\
									- do_rets.values[:-self.lookback,:]
		self.rs = (up_gain/self.lookback)/(do_loss/self.lookback)
		self.rsi = 100 - (100/(1 + self.rs))
		self.rsi.ix[:self.lookback,:] = np.nan
		self.rsi[self.rsi == np.inf] = 100

		# # momentum[t] = price[t]/price[t-n] - 1
		self.mm = self.price.copy()
		self.mm.values[self.lookback:,:] = self.mm.values[self.lookback:,:]/\
				self.mm.values[:-self.lookback,:]
		self.mm.values[self.lookback:,:] = self.mm.values[self.lookback:,:] - 1
		self.mm.values[:self.lookback,:] = np.nan

		# # stochastic oscillator
		self.so = self.price.copy()
		self.so_D = self.price.copy() # D is a 3-day moving avg of self.so
										# it is used as a helper for self.so
		lows = self.price.rolling(window=self.lookback,min_periods=self.lookback).min()
		highs = self.price.rolling(window=self.lookback,min_periods=self.lookback).max()
		high_low_diff = highs - lows
		price_low_diff = self.price - lows
		self.so = price_low_diff/high_low_diff
		self.so_D = self.so.rolling(window=3,min_periods=3).mean()

		if self.verbose:
			print self.so * 100

	def plot_sma(self):
		# Plot sma
			# overlay price and sma
		fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,8),sharex=True,\
						gridspec_kw = {'height_ratios':[3, 1]})

		ax1.plot(self.price['IBM'], label='IBM price', color ='magenta')
		ax1.plot(self.sma_['IBM'],label='IBM SMA',color='blue')
		ax1.plot(self.price['SPY'],label='SPY price', color='grey')

		ax2.plot(self.sma['IBM'], label='IBM sma-price ratio', color='magenta')
		ax2.plot(self.sma['SPY'],label='SPY sma-price ratio', color='grey')
		oversold_line = self.price.copy()
		overbought_line = self.price.copy()
		oversold_line.values[:,:]=0.95
		overbought_line.values[:,:]=1.05
		ax2.plot(oversold_line['IBM'],label='oversold',color='green')
		ax2.plot(overbought_line['IBM'],label='overbought',color='red')

		ax1.legend(loc='lower right')
		ax1.set_title('SMA and Price (IBM)')
		ax1.set_ylabel('Dollars')

		ax2.set_xlabel('Year')
		ax2.legend(loc='lower right')

	def plot_rsi(self):
		# Plot rsi
			# overlay price, line of oversold and overbought(<30,>70), and rsi

		fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,9),sharex=True,\
						gridspec_kw = {'height_ratios':[2, 1]})

		ax1.plot(self.price['IBM'], label='IBM price', color ='magenta')
		ax1.plot(self.rs['IBM'],label='IBM Relative Strength',color='blue')
		ax1.plot(self.price['SPY'],label='SPY price', color='grey')

		ax2.plot(self.rsi['IBM'], label='IBM Relative Strength Index', color='magenta')
		ax2.plot(self.rsi['SPY'], label='SPY RSI', color='grey')
		oversold_line = self.price.copy()
		overbought_line = self.price.copy()
		oversold_line.values[:,:]=30
		overbought_line.values[:,:]=70
		ax2.plot(oversold_line['IBM'],label='oversold',color='green')
		ax2.plot(overbought_line['IBM'],label='overbought',color='red')

		ax1.legend(loc='lower right')
		ax1.set_title('RSI and Price (IBM)')
		ax1.set_ylabel('Dollars')

		ax2.set_xlabel('Year')
		ax2.legend(loc='lower right')


	def plot_bb(self):
		# Plot bb
			# overlay price, avg, top band and bottom band
		fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,6),sharex=True,\
						gridspec_kw = {'height_ratios':[2, 1]})

		ax1.plot(self.price['IBM'], label='IBM price', color ='magenta')
		ax1.plot(self.price['SPY'],label='SPY price', color='grey')
		ax1.plot(self.sma_['IBM'], label='IBM SMA', color='blue')

		ax1.plot(self.top_band['IBM'], label='BB band-top', color='green')
		ax1.plot(self.bot_band['IBM'], label='BB band-bottom', color='red')

		separate_line = self.price.copy()
		separate_line.values[:,:] = 1
		ax2.plot(self.bb['IBM'], label='BB value', color='magenta')
		ax2.plot(separate_line['IBM'],label='1', color='green')

		ax1.legend(loc='lower right')
		ax1.set_title('BB and Price (IBM)')
		ax1.set_ylabel('Dollars')

		ax2.set_xlabel('Year')
		ax2.legend(loc='lower right')

	def plot_mm(self):
		# Plot mm
			# overlay price and momentum
		fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,6),sharex=True,\
						gridspec_kw = {'height_ratios':[2, 1]})

		ax1.plot(self.price['IBM'], label='IBM price', color ='magenta')
		ax1.plot(self.price['SPY'],label='SPY price', color='grey')
		ax2.plot(self.mm['IBM'], label='IBM momentum', color='blue')

		ax1.legend(loc='lower right')
		ax1.set_title('Momentum and Price (IBM)')
		ax1.set_ylabel('Dollars')

		ax1.set_xlabel('Year')
		ax1.legend(loc='lower right')

	def plot_so(self):
		# Plot so
			# overlay price, so, and so_D
		fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,9),sharex=True,\
						gridspec_kw = {'height_ratios':[2, 1]})

		ax1.plot(self.price['IBM'], label='IBM price', color ='magenta')
		ax1.plot(self.price['SPY'],label='SPY price', color='grey')
		ax2.plot(self.so['IBM']*100, label='IBM Stochastic Oscillator', color='blue')
		# ax2.plot(self.so_D['IBM']*100, label='IBM Stochastic Oscillator D%', color='green')

		ax1.legend(loc='lower right')
		ax1.set_title('Stochastic Oscillator and Price (IBM)')
		ax1.set_ylabel('Dollars')

		ax2.set_xlabel('Year')
		ax2.set_ylabel('Percentage')
		ax2.set_ylim([0,110])
		ax2.legend(loc='lower right')

if __name__=="__main__":

	start_date = dt.datetime(2005,01,01)
	end_date = dt.datetime(2011,12,31)
	symbols = ['IBM']
	lookback=14
	data = util.get_data(symbols,pd.date_range(start_date,end_date))

	tech_inds= tech_indicators(symbols, start_date, end_date, lookback, True)

	tech_inds.compute_indicators()
	tech_inds.sma['IBM'].values.sort()
	some_vals = np.array([[0.1],[0.5],[1.0]])
	
	print some_vals <= tech_inds.sma['IBM'].values
