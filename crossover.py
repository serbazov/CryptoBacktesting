import time
import pandas as pd
from binance.client import Client
import config
from backtest import Backtrading
import matplotlib.pyplot as plt
import ta.volatility as tavol
from datetime import datetime, date

from scipy.optimize import optimize

NOW = int(round(time.time())) * 1000
CASH = 100.
TRADE_QUANTITY = 25
RSI_OVERSELL = 30
RSI_OVERBOUGHT = 70
STOP_LOSS = 0.8
TRADE_SYMBOL = 'ADAUSDT'
COMMISSION = 0.00075
DEVIATION = 2
SMA_LEN = 1000
KLINE_INTERVAL = 5
START = datetime(2022, 4, 5)
STOP = datetime(2022, 3, 30)
START = int(round(START.timestamp() * 1000))
STOP = int(round(STOP.timestamp() * 1000))


def get_data(Client, start, stop):
    client = Client(config.ApiKey, config.SecretKey, tld='com')
    df = pd.DataFrame(
        client.get_historical_klines(TRADE_SYMBOL, client.KLINE_INTERVAL_5MINUTE, start, stop))
    df = df.iloc[:, :6]
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df.set_index('Time')
    df.index = pd.to_datetime(df.index, unit='ms')
    df = df.astype(float)
    return df


def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    if n > len(values):
        n = len(values) - 3
    sma = pd.Series(values).rolling(n).mean()
    sma = sma.dropna()
    sma = sma.reset_index(drop=True)
    sma.pop(0)
    return sma


def RSI(array, n):
    """Relative strength index"""
    # Approximate; good enough
    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
    return 100 - 100 / (1 + rs)


def get_sma(start, stop, sma_len):
    additional_start = start - sma_len * KLINE_INTERVAL * 60 * 1000
    df = get_data(Client, additional_start, stop)
    sma = SMA(df['Close'], sma_len)
    return sma


df = get_data(Client, START, NOW)

balance = []
closes = df['Close']
rsi = RSI(closes, 14)
rsi[0] = 0

sma = get_sma(START, NOW, SMA_LEN)
up = tavol.bollinger_hband(closes, window=20, window_dev=DEVIATION)
low = tavol.bollinger_lband(closes, window=20, window_dev=DEVIATION)
bbp = (closes - low) / (up - low)
backtrading = Backtrading(CASH, COMMISSION)
buy_trades_x = []
buy_trades_y = []
sell_trades_x = []
sell_trades_y = []
equity = []
stop_loss = []
stop_loss.append(0)

for i in range(len(closes)):
    if i > 0:
        stop_loss.sort()
        if stop_loss[-1] > closes[i] and backtrading.equity >= TRADE_QUANTITY:
            # backtrading.sell(TRADE_QUANTITY, closes[i])
            stop_loss.pop()
            # sell_trades_x.append(df.index[i])
            # sell_trades_y.append(closes[i])

        if rsi[i] < RSI_OVERSELL and backtrading.cash >= TRADE_QUANTITY * closes[i] and i > 20 and bbp[i - 1] < 0. and \
                bbp[i] > 0.:
            backtrading.buy(TRADE_QUANTITY, closes[i])
            buy_trades_x.append(df.index[i])
            buy_trades_y.append(closes[i])
            stop_loss.append(STOP_LOSS * closes[i])

        if rsi[i] > RSI_OVERBOUGHT and backtrading.equity >= TRADE_QUANTITY and bbp[i - 1] > 1. and bbp[i] < 1 and \
                closes[i] >= sma[i]:
            backtrading.sell(TRADE_QUANTITY, closes[i])
            sell_trades_x.append(df.index[i])
            sell_trades_y.append(closes[i])

    equity.append(backtrading.equity)
    balance.append(backtrading.cash + backtrading.equity * closes[i] * (1 - COMMISSION))

print(balance[-1])
print(max(balance))

plt.figure(1)
plt.subplot(311)
plt.plot(df.index, balance)
plt.ylabel('balance')
plt.subplot(312)
plt.plot(df.index, closes)
plt.plot(buy_trades_x, buy_trades_y, 'o')
plt.plot(sell_trades_x, sell_trades_y, 'D')
plt.plot(df.index, sma)
plt.plot(df.index, up)
plt.plot(df.index, low)
plt.ylabel('closes price')
plt.subplot(313)
plt.plot(df.index, equity)
plt.ylabel('equity')
plt.xlabel('Time')
plt.show()
