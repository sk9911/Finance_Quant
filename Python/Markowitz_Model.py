import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as optimization
import streamlit as st
import datetime
import pickle
import gdown

NUM_TRADING_DAYS = 252
gdown '1xkC4WsHJ4wRH06Lhgu14JQ2zPOWX01uj'
TICKER_FILE = r"./Tickers.txt"

def download_data(stocks, start_date, end_date):
    """Downloads the historical price data for the given
	
	Args:
	    stocks (list): List of stocks for which data shall be downloaded
	    start_date (date): Start date for which historical prices are to be downloaded
	    end_date (date): Last date for which historical prices are to be downloaded
	
	Returns:
	    pd.DataFrame: DataFrame containing dates on axis 0, asset name on axis 1.
	"""
    stock_data = {}

    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)["Close"]

    return pd.DataFrame(stock_data)


def show_data(data):
    """Plot the input data using matplotlib
	
	Args:
	    data (pd.DataFrame): DataFrame to be plotted
	"""
    data.plot(figsize=(10, 5))
    plt.show()


def calculate_return(data):
    """Calculates and returns log daily returns of input historical prices. First row is omitted.
	
	Args:
	    data (pd.DataFrame): DataFrame to be plotted
	
	Returns:
	    pd.DataFrame: DataFrame containing dates on axis 0, asset name on axis 1.
	"""
    log_return = np.log(data / data.shift(1))
    return log_return[1:]


def show_statistics(returns):
    """Prints the Expected annual returns and plots the correlation matrix for given daily returns input
	
	Args:
	    returns (pd.DataFrame): DataFrame containing daily returns
	"""
    print("Expected annual return=\n", returns.mean() * NUM_TRADING_DAYS)
    # print("\nCovariance=\n", returns.cov() * NUM_TRADING_DAYS)
    fig = plt.figure()
    fig.suptitle("Correlation Matrix")
    sns.heatmap(returns.corr() * NUM_TRADING_DAYS, cmap="flare")
    plt.show()


def show_mean_variance(returns, weights):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights))
    )
    print("Expected portfolio mean (return)", portfolio_return)
    print("Expected portfolio volatility (stdev)", portfolio_volatility)


def generate_portfolios(returns, stocks, NUM_PORTFOLIOS=10000):
    """Generate NUM_PORTFOLIOS number of portfolios. Creates and returns means, risks and weights of random portfolios
	
	Args:
	    returns (pd.DataFrame): Daily returns of stocks
	    stocks (List): List of stocks
	    NUM_PORTFOLIOS (int, optional): NUM_PORTFOLIOS for which Monte Carlo Simulation in carried out
	
	Returns:
	    Tuple(3): Return tuple of lists containing weights [dim=len(stocks), ], means [dim=len(stocks), ], risks
	"""
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(
            np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w)))
        )

    return (
        np.array(portfolio_weights),
        np.array(portfolio_means),
        np.array(portfolio_risks),
    )


def show_portfolios(means, risks):
    """Plots the efficient frontier
	
	Args:
	    means (List): Portfolio means/ expected returns
	    risks (List): Portfolio risks/ expected volatility
	"""
    plt.figure(figsize=(10, 6))
    plt.scatter(risks, means, c=means / risks, marker="o")
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Returns")
    plt.colorbar(label="Sharpe Ratio")


def statistics(weights, returns, risk_free_rate=0.06768):
    """Returns statistics portfolio return, portfolio volatility and sharpe ratio
	
	Args:
	    weights (List): List of portfolio weights
	    returns (List): List of expected returns for individual stocks
	
	Returns:
	    np.array: [portfolio return, portfolio volatility, sharpe ratio]
	"""
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights))
    )

    return np.array(
        [
            portfolio_return,
            portfolio_volatility,
            (portfolio_return - 0.06768) / portfolio_volatility,
        ]
    )


# scipy optimize module
def min_function_sharpe(weights, returns):
    """In order to maximise Sharpe ratio, we minimize the negative of Sharpe ratio
	
	Args:
	    weights (List): List of portfolio weights
	    returns (List): List of expected returns for individual stocks
	
	Returns:
	    Array(int): Negative of Sharpe Ratio
	"""
    return -statistics(weights, returns)[2]


# constraints sum weights = 1 IE sum weights -1 = 0
def optimize_portfolio(weights, returns):
    """Function to optimize weights to maximise Sharpe Ratio using scipy
	
	Args:
	    weights (List): List of portfolio weights
	    returns (List): List of expected returns for individual stocks
	
	Returns:
	    scipy.optimize.optimize.OptimizeResult: SciPy optimization result
	"""
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(
        fun=min_function_sharpe,
        x0=weights[0],
        args=returns,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )


def print_optimal_portfolio(optimum, returns):
    """Print expected portfolio return, expected porfolio variance and weights of individual assets in optimized portfolio
	
	Args:
	    optimum (TYPE): Scipy optimization result
	    returns (TYPE): List of expected returns for individual stocks
	"""
    print("----------------------------------------------")
    print("Optimal portfolio: ", optimum["x"].round(3))
    for i in range(len(stocks)):
        st.write(stocks[i], (optimum["x"][i] * 100).round(3), "%")
    print(
        "\nExpected return, volatility and Sharpe ratio: ",
        statistics(optimum["x"].round(3), returns),
    )


def show_optimum_portfolio(optimum, means, risks, returns):
    """Plot the optimal portfolio along with Monte Carlo simulation points on the efficient frontier
	
	Args:
	    optimum (scipy.optimize.optimize.OptimizeResult): Scipy optimization result
	    means (List): Portfolio returns of the random portfolios
	    risks (List): Portfolio risks of the random portfolios
	    returns (List): List of expected returns for individual stocks
	"""
    opt_stat = statistics(optimum["x"], returns)

    opt_fig = plt.figure(figsize=(10, 6))
    plt.scatter(risks, means, c=means / risks, marker="o")
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Returns")
    plt.colorbar(label="Sharpe Ratio")
    plt.plot(opt_stat[1], opt_stat[0], "g*", markersize="15")
    plt.show()
    st.pyplot(opt_fig)


if __name__ == "__main__":

    with open(TICKER_FILE, "rb") as fp:  # Unpickling
        tickers = pickle.load(fp)

    stocks = st.multiselect("Pick stocks", tickers)

    NUM_PORTFOLIOS = st.number_input(
        "Number of portfolios to simulate", value=int(10000)
    )
    start_date = st.date_input("Enter Start Date for MPT", datetime.date(2020, 1, 9))
    st.write("Start Date Entered:", start_date)

    end_date = st.date_input("Enter Start Date for MPT", datetime.date(2022, 1, 1))
    st.write("End Date Entered:", end_date)

    if st.button("Get Optimal Portfolio"):

        if len(stocks) > 0:

            stocks = [f"{x}.NS" for x in stocks]
            dataset = download_data(stocks, start_date, end_date)
            lg_ret = calculate_return(dataset)
            # show_statistics(lg_ret)
            # show_data(dataset)

            pweights, means, risks = generate_portfolios(lg_ret, stocks, NUM_PORTFOLIOS)

            optimum = optimize_portfolio(pweights, lg_ret)
            print_optimal_portfolio(optimum, lg_ret)
            show_optimum_portfolio(optimum, means, risks, lg_ret)

