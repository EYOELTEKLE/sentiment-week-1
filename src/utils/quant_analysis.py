import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TA-Lib and PyNance imports are assumed to be available
try:
    import talib
except ImportError:
    talib = None
try:
    import pynance as pn
except ImportError:
    pn = None

class QuantitativeAnalyzer:
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path, parse_dates=['Date'])
        self.data.sort_values('Date', inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def add_sma(self, period=20, price_col='Close'):
        if talib:
            self.data[f'SMA_{period}'] = talib.SMA(self.data[price_col], timeperiod=period)
        else:
            self.data[f'SMA_{period}'] = self.data[price_col].rolling(window=period).mean()

    def add_ema(self, period=20, price_col='Close'):
        if talib:
            self.data[f'EMA_{period}'] = talib.EMA(self.data[price_col], timeperiod=period)
        else:
            self.data[f'EMA_{period}'] = self.data[price_col].ewm(span=period, adjust=False).mean()

    def add_rsi(self, period=14, price_col='Close'):
        if talib:
            self.data[f'RSI_{period}'] = talib.RSI(self.data[price_col], timeperiod=period)
        else:
            delta = self.data[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            self.data[f'RSI_{period}'] = 100 - (100 / (1 + rs))

    def add_macd(self, fastperiod=12, slowperiod=26, signalperiod=9, price_col='Close'):
        if talib:
            macd, macdsignal, macdhist = talib.MACD(self.data[price_col], fastperiod, slowperiod, signalperiod)
            self.data['MACD'] = macd
            self.data['MACD_Signal'] = macdsignal
            self.data['MACD_Hist'] = macdhist
        else:
            exp1 = self.data[price_col].ewm(span=fastperiod, adjust=False).mean()
            exp2 = self.data[price_col].ewm(span=slowperiod, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=signalperiod, adjust=False).mean()
            self.data['MACD'] = macd
            self.data['MACD_Signal'] = signal
            self.data['MACD_Hist'] = macd - signal

    def add_pynance_metrics(self, ticker=None):
        if pn and ticker:
            # Example: fetch returns and volatility
            try:
                returns = pn.data.get(ticker).returns()
                self.data['Pn_Returns'] = returns.values
            except Exception as e:
                print(f"PyNance error: {e}")

    def plot_price_with_indicators(self, indicators=['SMA_20', 'EMA_20']):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['Date'], self.data['Close'], label='Close Price')
        for ind in indicators:
            if ind in self.data:
                plt.plot(self.data['Date'], self.data[ind], label=ind)
        plt.title('Price with Indicators')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_rsi(self, period=14):
        col = f'RSI_{period}'
        if col in self.data:
            plt.figure(figsize=(14, 3))
            plt.plot(self.data['Date'], self.data[col], label=col)
            plt.axhline(70, color='red', linestyle='--')
            plt.axhline(30, color='green', linestyle='--')
            plt.title('RSI')
            plt.xlabel('Date')
            plt.ylabel('RSI')
            plt.legend()
            plt.grid(True)
            plt.show()

    def plot_macd(self):
        if 'MACD' in self.data and 'MACD_Signal' in self.data:
            plt.figure(figsize=(14, 4))
            plt.plot(self.data['Date'], self.data['MACD'], label='MACD')
            plt.plot(self.data['Date'], self.data['MACD_Signal'], label='Signal')
            plt.bar(self.data['Date'], self.data['MACD_Hist'], label='Hist', color='gray', alpha=0.3)
            plt.title('MACD')
            plt.xlabel('Date')
            plt.ylabel('MACD')
            plt.legend()
            plt.grid(True)
            plt.show() 