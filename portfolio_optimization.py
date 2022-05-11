import pandas as pd
from binance.client import Client
import config
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, Bounds
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

TRADE_SYMBOL = ['BTCUSDT', 'BNBUSDT', 'ETHUSDT', 'LTCUSDT', 'ATOMUSDT', 'XRPUSDT', 'DOTUSDT']
init_coef = np.ones(len(TRADE_SYMBOL))


def get_data(symbol):
    client = Client(config.ApiKey, config.SecretKey, tld='com')
    df = pd.DataFrame(
        client.get_historical_klines(symbol, client.KLINE_INTERVAL_1DAY, "30 Dec, 2020", "30 March, 2022"))
    df = df.iloc[:, :6]
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df.set_index('Time')
    df.index = pd.to_datetime(df.index, unit='ms')
    df = df.astype(float)
    return df


def get_daily_return(df) -> np.ndarray:
    # daily_return = df['Close'].copy()
    daily_return = (df['Close'][1:].values - df['Close'][:-1].values) / df['Close'][:-1].values
    daily_return = np.concatenate(([0.], daily_return))
    return daily_return


def get_cumulative_return(df):
    return 100 * (df['Close'][-1] - df['Close'][0]) / df['Close'][0]


def get_balance(closes, trade_symbols, coeff):
    balance = np.zeros(len(closes))
    for k in range(len(trade_symbols)):
        for j in range(len(closes)):
            balance[j] = balance[j] + closes[trade_symbols[k]].values[j] * coeff[k]

    balance = pd.DataFrame(balance)
    balance.index = closes.index
    balance.columns = ['Close']
    balance = balance / balance.values[0]

    return balance


def get_sharp_ratio(df):
    return get_cumulative_return(df) * len(df['Close']) / (get_daily_return(df).std() * np.sqrt(len(df['Close'])))


def sharp_for_optimizer(coeff, trade_symbol, closes):
    balance = get_balance(closes, trade_symbol, coeff)
    return -1 * get_sharp_ratio(balance)


def std_for_optimization(coeff, trade_symbol, closes):
    balance = get_balance(closes, trade_symbol, coeff)
    return get_daily_return(balance).std()


daily_return = []
closes = pd.DataFrame()

for i in range(len(TRADE_SYMBOL)):
    df = get_data(TRADE_SYMBOL[i])
    daily_return.append(get_daily_return(df))
    closes = pd.concat([closes, df['Close']], axis=1)
    # plt.hist(daily_return[i], bins=50, density=True)
    # print('рост', TRADE_SYMBOL[i], '=', 100 * get_cumulative_return(df), '%')
    # print('среднее отклонение', TRADE_SYMBOL[i], '=', 100 * daily_return[i].std(), '%')
    # print(get_sharp_ratio(df))
closes.columns = TRADE_SYMBOL

closes = (closes) / closes.values[0]

# closes = norm(closes, TRADE_SYMBOL)
# ax = df['Close'].plot()


# correlation = np.corrcoef(daily_return)
# print(correlation)

# plt.plot(closes)

con = LinearConstraint([np.ones(len(TRADE_SYMBOL))], 1., 1)
bnds = Bounds(np.zeros(len(TRADE_SYMBOL)), np.ones(len(TRADE_SYMBOL)))

# constraints={'type': 'ineq',
#                                'fun': lambda x: x[0] - 0.5}

optimum = minimize(std_for_optimization, init_coef, args=(TRADE_SYMBOL, closes,), method='trust-constr',
                   constraints=con, bounds=bnds)

balance = get_balance(closes, TRADE_SYMBOL, optimum.x)

mu = mean_historical_return(balance)
S = CovarianceShrinkage(balance).ledoit_wolf()

print(optimum.x)
closes.plot()
balance.plot()
# plt.plot(df.index, closes)
plt.show()
