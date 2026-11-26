import yfinance as yf
import matplotlib.pyplot as plt

prices = yf.download(['MSFT', 'AAPL', 'GOOG'],period='1mo')["Close"]
print (prices.tail())

daily_returns = prices.pct_change()
print("\nBiggest AAPL down days:")
print(daily_returns["AAPL"].nsmallest(5))

(prices / prices.iloc[0]).plot(title="AAPL vs SPY (growth of $1)")
plt.show()