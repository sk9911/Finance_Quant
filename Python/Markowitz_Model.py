import numpy as np 
import yfinance as yf 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

start_date = '2012-01-01'
end_date = '2017-01-01'

NUM_TRADING_DAYS = 252

def download_data():
	stock_data = {}

	for stock in stocks:
		ticker = yf.Ticker(stock)
		stock_data[stock] = ticker.history( start=start_date, end=end_date)['Close']

	return pd.DataFrame(stock_data)

def show_data(data):
	data.plot(figsize = (10,5))
	plt.show()

def calculate_return(data):
	log_return = np.log(data/data.shift(1))
	return log_return[1:]

def show_statistics(returns):
	print("Expected annual return=\n",returns.mean() * NUM_TRADING_DAYS)
	print("\nCovariance=\n", returns.cov() * NUM_TRADING_DAYS)

if __name__ == '__main__':
	dataset = download_data()
	lg_ret = calculate_return(dataset)
	show_statistics(lg_ret)
	show_data(dataset)