import numpy as np 
import yfinance as yf 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

start_date = '2012-01-01'
end_date = '2017-01-01'

NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000

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

def show_mean_variance(returns, weights):
	portfolio_return = np.sum(returns.mean()*weights)* NUM_TRADING_DAYS
	portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()* NUM_TRADING_DAYS, weights)))
	print("Expected portfolio mean (return)", portfolio_return)
	print("Expected portfolio volatility (stdev)", portfolio_volatility)

def generate_portfolios(returns):
	portfolio_means = []
	portfolio_risks = []
	portfolio_weights = []

	for _ in range(NUM_PORTFOLIOS):
		w = np.random.random(len(stocks))
		w /= np.sum(w)
		portfolio_weights.append(w)
		portfolio_means.append(np.sum(returns.mean()*w)* NUM_TRADING_DAYS)
		portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov()* NUM_TRADING_DAYS, w))))

	return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)

def show_portfolios(means, risks):
	plt.figure(figsize=(10,6))
	plt.scatter(risks, means, c=means/risks, marker='o')
	plt.grid(True)
	plt.xlabel('Expected Volatility')
	plt.ylabel('Expected Returns')
	plt.colorbar(label='Sharpe Ratio')
	plt.show()

def statistics(weights, returns):
	portfolio_return = np.sum(returns.mean()*weights)* NUM_TRADING_DAYS
	portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()* NUM_TRADING_DAYS, weights)))

	return np.array([portfolio_return, portfolio_volatility, portfolio_return/portfolio_volatility])

#scipy optimize module
def min_function_sharpe(weights, returns):
	return -statistics(weights, returns)[2]

#constraints sum weights = 1 IE sum weights -1 = 0
def optimize_portfolio(weights, returns):
	constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) -1}
	bounds = tuple((0,1) for _ in range(len(stocks)))
	return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)

def print_optimal_portfolio(optimum, returns):
	print("\nOptimal portfolio: ", optimum['x'].round(3))
	print("\nExpected return, volatility and Sharpe ratio: ", statistics(optimum['x'].round(3), returns))

def show_optimum_portfolio(optimum, means, risks, returns):
	opt_stat = statistics(optimum['x'], returns)

	plt.figure(figsize=(10,6))
	plt.scatter(risks, means, c=means/risks, marker='o')
	plt.grid(True)
	plt.xlabel('Expected Volatility')
	plt.ylabel('Expected Returns')
	plt.colorbar(label='Sharpe Ratio')
	plt.plot(opt_stat[1], opt_stat[0], 'g*', markersize='15')
	plt.show()

if __name__ == '__main__':
	dataset = download_data()
	lg_ret = calculate_return(dataset)
	# show_statistics(lg_ret)
	# show_data(dataset)

	pweights, means, risks = generate_portfolios(lg_ret)
	optimum = optimize_portfolio(pweights, lg_ret)
	print_optimal_portfolio(optimum, lg_ret)
	show_optimum_portfolio(optimum, means, risks, lg_ret)