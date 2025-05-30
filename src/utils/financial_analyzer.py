import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Union, Optional

class FinancialAnalyzer:
    """A class for performing financial analysis using TA-Lib and PyNance."""
    
    def __init__(self):
        """Initialize the FinancialAnalyzer."""
        pass

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate various technical indicators using TA-Lib.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            Dict[str, pd.Series]: Dictionary containing calculated indicators
        """
        indicators = {}
        
        # Moving Averages
        indicators['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        indicators['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        indicators['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
        
        # RSI
        indicators['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            df['Close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        indicators['MACD'] = macd
        indicators['MACD_Signal'] = macd_signal
        indicators['MACD_Hist'] = macd_hist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df['Close'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        indicators['BB_Upper'] = upper
        indicators['BB_Middle'] = middle
        indicators['BB_Lower'] = lower
        
        return indicators
    
    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate historical volatility.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            window (int): Rolling window for volatility calculation
            
        Returns:
            pd.Series: Historical volatility
        """
        returns = np.log(df['Close'] / df['Close'].shift(1))
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return volatility
    
    def calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate various risk metrics.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            Dict[str, float]: Dictionary containing risk metrics
        """
        daily_returns = df['Close'].pct_change()
        
        metrics = {
            'Daily_Volatility': daily_returns.std(),
            'Annualized_Volatility': daily_returns.std() * np.sqrt(252),
            'Sharpe_Ratio': (daily_returns.mean() / daily_returns.std()) * np.sqrt(252),
            'Max_Drawdown': (df['Close'] / df['Close'].cummax() - 1).min(),
        }
        
        return metrics
    
    def get_support_resistance_levels(self, df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """
        Calculate support and resistance levels using pivot points.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            window (int): Window size for calculating levels
            
        Returns:
            Dict[str, float]: Support and resistance levels
        """
        pivot = (df['High'].rolling(window=window).max() + 
                df['Low'].rolling(window=window).min() + 
                df['Close'].rolling(window=window).mean()) / 3
        
        resistance1 = 2 * pivot - df['Low'].rolling(window=window).min()
        support1 = 2 * pivot - df['High'].rolling(window=window).max()
        
        levels = {
            'Pivot': pivot.iloc[-1],
            'Resistance1': resistance1.iloc[-1],
            'Support1': support1.iloc[-1]
        }
        
        return levels
