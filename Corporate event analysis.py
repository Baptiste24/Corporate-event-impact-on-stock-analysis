# -*- coding: utf-8 -*-

##############################################################################
############################## PYTHON PROJECT 2 ##############################
##############################################################################

################################## LIBRARIES ##################################
from sklearn.linear_model import LinearRegression # to calculate linear regression
import statistics
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

################################## FUNCTIONS ##################################

def trading_days(start_date, end_date, holidays):
    """
    Function used to count the trading days between 2 dates and taking in count holidays

    Parameters
    ----------
    start_date : string (format date as "YYYY-MM-DD")
    end_date : string (format date as "YYYY-MM-DD")
    holidays : list of dates (format as [datetime(YYYY, MM, DD), ...])

    Returns
    -------
    trading_days : int

    """
    trading_days = 0

    # Convert dates in datetime objects 
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # count each day between 2 dates
    current_date = start_date
    while current_date <= end_date:
        # Check if current date is on week-end or holiday
        if current_date.weekday() < 5 and current_date not in holidays:
            # If it is not week-end or holiday, add 1 to the number of trading days
            trading_days += 1

        # Next day
        current_date += timedelta(days=1)

    return trading_days

################################ MAIN PROGRAM ################################

# Question 1-7
# Used to define the numbers of trading days around the event date
"""
start_date = "2022-05-10"
end_date = "2022-11-05"
holidays = [datetime(2022, 1, 1), datetime(2022, 2, 21), datetime(2022, 5, 30), datetime(2022, 6, 19), datetime(2022, 7, 4), datetime(2022, 9, 5), datetime(2022, 11, 11), datetime(2022, 11, 24), datetime(2022, 12, 26)]  # 2022 hollidays
trading_days = trading_days(start_date, end_date, holidays)
print(f"Il y a {trading_days} jours de trading entre {start_date} et {end_date}.")
"""


SP500_ticker = "^GSPC"          # S&P500 ticker
amazon_ticker = "AMZN"          # Amazon ticker
event_date = "2022-10-28"       # Event date
start_date = "2022-05-10"       # event date -120 trading days
end_date = "2022-11-08"         # event date +5 trading days +1 because the download function exclude this date

print("=======================================")
print("Corporation : Amazon")
print("Ticker :", amazon_ticker, "\n")
print("Event nature : Q3 results revenue")
print("Event date : ", event_date)
print("=======================================\n\n\
=========\n\
Summary\n\
=========\n")

#extraction of the data from Yahoo Finance
sp500 = yf.download(SP500_ticker, start=start_date, end=end_date)
amazon = yf.download(amazon_ticker, start=start_date, end=end_date)


# Question 8 : We plot the Adj close price and the volume over price on the same graph
fig, ax1 = plt.subplots()

ax1.plot(amazon['Adj Close'], color='b', label='Price')
ax2 = ax1.twinx()
ax2.plot(amazon['Volume'], color='orange', label='Volume')

ax1.set_xlabel('Date')
ax1.set_ylabel('Stock Price', color='b')
ax2.set_ylabel('Volume', color='orange')
plt.title("Amazon Stock price and Volume over time")

plt.show()


# Question 9 : Calculate the stock returns from t-120 to t+5
# We compute the daily returns on the adjusted close price
# For this, we use pct_change() function

amazon_daily_returns = amazon['Adj Close'].pct_change()
print("\n\nAmazon daily returns from t-120 to t+5\n\
-------------------------\n", amazon_daily_returns)

# And we also compute the sp500 daily returns because we will need it for the linear regression
sp500_daily_returns = sp500['Adj Close'].pct_change()
print("\n\nS&P500 daily returns from t-120 to t+5\n\
-------------------------\n", sp500_daily_returns)


# Question 10
# control window = [t-120, t-6] : 120-6+1 = 115 values
print("\n\nControl window = [t-120, t-6]\n\
-------------------------")

# we create a list for control window
returns_cw = amazon_daily_returns[1:116]

average_return = sum(returns_cw) / len(returns_cw)
print("average_return : {0:.2%}.".format(average_return))

volatility = statistics.stdev(returns_cw)
volatility = round(volatility, 2)
print("volatility : ", volatility)


# Question 11
# Event window = [t, t+5] : 5-0+1 = 6 values
print("\n\nEvent window = [t, t+5]\n\
-------------------------")

# we create a list for event window
returns_ew = amazon_daily_returns[120:]

average_return = sum(returns_ew) / len(returns_ew)
print("average_return : {0:.2%}.".format(average_return))

volatility = statistics.stdev(returns_ew)
volatility = round(volatility, 2)
print("volatility : ", volatility)


# Question 12
print("\n\nMarket model regression\n\
-------------------------")

# Define variables for amazon and S&P500 returns only over the control window (cw)
returns_cw = amazon_daily_returns[1:116]
sp500_cw = sp500_daily_returns[1:116]
                             
#To delete the first line of returns (= NaN)
X = returns_cw.iloc[1:].to_numpy().reshape(-1,1)
Y = sp500_cw.iloc[1:].to_numpy().reshape(-1,1)

#We want to plot the linear regression of the market model
lin_regr = LinearRegression()
lin_regr.fit(X, Y)

Y_pred = lin_regr.predict(X)

alpha = lin_regr.intercept_[0]
beta = lin_regr.coef_[0, 0]

fig, ax = plt.subplots()
ax.scatter(X, Y)
ax.plot(X, Y_pred, c = 'r')
ax.set_xlabel('Amazon daily returns')
ax.set_ylabel('S&P500 daily returns')
plt.title("Market model regression over the control window")
plt.show()

print("Alpha hat = ", alpha)
print("Beta hat = ", beta)


# Question 13
R_mt_ew = sp500_daily_returns[120:]
AR_t_ew = returns_ew - (alpha + beta * R_mt_ew)
print("\n\nAbnormal returns of the event window\n\
-------------------------\n", AR_t_ew)


# Question 14
CAR_ew = sum(AR_t_ew)
print("\n\nCumulative abnormal returns\n\
-------------------------\n", CAR_ew)


# Question 15
AR_t_cw = returns_cw - (alpha + beta * sp500_cw)

sigma_ar = AR_t_cw.std() # standard deviation of the abnormal returns over the control window (not event window!).
stat = CAR_ew / (6**(1/2) * sigma_ar)

print("\n\nTest statistics\n\
-------------------------\n", stat)


# Question 16
left_tail_critical_value = -1.645

# Conclusion
print("\n\nConclusion\n\
-------------------------")
print("Amazon missed the earnings call, hence we believe that this event would have a negative impact on the stock price on the event window.\n\
Furthermore, Q14 displayed negative abnormal returns.\n\
Hence we believe that the one-tail test on the left is appropriate which gives us a critical value of", left_tail_critical_value, "for a confidence level = 95%.")
print("\nOur null hypothesis is : Our event didn't impact the stock price.")
print("Our alternative hypothesis is : Our event had a negative impact on the stock price.\n")

print("Test statistic = ", stat)
print("left tail critical value = ", left_tail_critical_value)
if stat < left_tail_critical_value:
    print("Test statistic < left tail critical value\n\
We reject our null hypothesis for a confidence level = 95%.\n\
Hence the CAR of the event window is significantly different from 0.")
else:
    print("Test statistic >= left tail critical value\n\
We accept our null hypothesis for a confidence level = 95%.\n\
Hence the CAR of the event window is not significantly different from 0.")