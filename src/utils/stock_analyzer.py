"""
Stock price analysis and correlation with sentiment.
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Dict, Tuple, Optional, Any, Union
from .base_analyzer import BaseAnalyzer

class StockAnalyzer(BaseAnalyzer):
    """Analyzer for stock price data and correlation with sentiment."""
    
    def __init__(self, price_data: pd.DataFrame, sentiment_data: pd.DataFrame,
                 price_date_col: str, sentiment_date_col: str,
                 close_price_col: str = 'close'):
        """
        Initialize the stock analyzer.
        
        Args:
            price_data: DataFrame containing stock prices
            sentiment_data: DataFrame containing sentiment scores
            price_date_col: Name of date column in price_data
            sentiment_date_col: Name of date column in sentiment_data
            close_price_col: Name of closing price column
        """
        self.sentiment_data = sentiment_data.copy()
        self.price_date_col = price_date_col
        self.sentiment_date_col = sentiment_date_col
        self.close_price_col = close_price_col
        super().__init__(price_data)
    
    def _validate_data(self) -> None:
        """Validate input data structure."""
        # Validate price data
        if self.close_price_col not in self.data.columns:
            raise ValueError(f"Missing price column: {self.close_price_col}")
        if self.price_date_col not in self.data.columns:
            raise ValueError(f"Missing date column in price data: {self.price_date_col}")
        
        # Validate sentiment data
        if self.sentiment_date_col not in self.sentiment_data.columns:
            raise ValueError(f"Missing date column in sentiment data: {self.sentiment_date_col}")
        if 'mean_polarity' not in self.sentiment_data.columns:
            raise ValueError("Missing mean_polarity column in sentiment data")
    
    def calculate_returns(self) -> pd.Series:
        """Calculate daily returns from price series."""
        if self.data.empty:
            return pd.Series(dtype=float)
        return self.data[self.close_price_col].pct_change()
    
    def align_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align sentiment and price data by date.
        
        Returns:
            Tuple of aligned (sentiment_df, returns_df)
        """
        # Calculate returns
        self.data['returns'] = self.calculate_returns()
        
        # Ensure datetime indices
        sentiment_df = self.sentiment_data.copy()
        returns_df = self.data.copy()
        
        sentiment_df[self.sentiment_date_col] = pd.to_datetime(sentiment_df[self.sentiment_date_col])
        returns_df[self.price_date_col] = pd.to_datetime(returns_df[self.price_date_col])
        
        # Set date indices
        sentiment_df = sentiment_df.set_index(self.sentiment_date_col)
        returns_df = returns_df.set_index(self.price_date_col)
        
        # Get common dates
        common_dates = sentiment_df.index.intersection(returns_df.index)
        
        return sentiment_df.loc[common_dates], returns_df.loc[common_dates]
    
    def calculate_correlation(self, sentiment_series: pd.Series,
                            returns_series: pd.Series) -> Dict[str, float]:
        """
        Calculate correlation metrics between sentiment and returns.
        
        Args:
            sentiment_series: Series of sentiment scores
            returns_series: Series of stock returns
            
        Returns:
            Dictionary of correlation metrics
        """
        if len(sentiment_series) != len(returns_series):
            raise ValueError("Series must have same length")
        
        if len(sentiment_series) < 2:
            return {
                'pearson_correlation': None,
                'pearson_p_value': None,
                'spearman_correlation': None,
                'spearman_p_value': None,
                'valid': False,
                'n_observations': len(sentiment_series)
            }
        
        try:
            pearson_corr, pearson_p = stats.pearsonr(sentiment_series, returns_series)
            spearman_corr, spearman_p = stats.spearmanr(sentiment_series, returns_series)
            
            return {
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'valid': True,
                'n_observations': len(sentiment_series)
            }
        except Exception as e:
            return {
                'pearson_correlation': None,
                'pearson_p_value': None,
                'spearman_correlation': None,
                'spearman_p_value': None,
                'valid': False,
                'n_observations': len(sentiment_series),
                'error': str(e)
            }
    
    def analyze_lagged_correlations(self, max_lag: int = 5) -> pd.DataFrame:
        """
        Analyze correlations with different lags between sentiment and returns.
        
        Args:
            max_lag: Maximum number of lags to analyze
            
        Returns:
            DataFrame containing correlation metrics for each lag
        """
        aligned_sentiment, aligned_returns = self.align_data()
        
        if aligned_sentiment.empty or aligned_returns.empty:
            return pd.DataFrame()
        
        results = []
        sentiment_series = aligned_sentiment['mean_polarity']
        returns_series = aligned_returns['returns']
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # Sentiment leads returns
                s1 = sentiment_series.shift(-lag)
                s2 = returns_series
                direction = 'sentiment_lead'
            elif lag > 0:
                # Returns lead sentiment
                s1 = sentiment_series
                s2 = returns_series.shift(lag)
                direction = 'returns_lead'
            else:
                # No lag
                s1 = sentiment_series
                s2 = returns_series
                direction = 'same_day'
            
            # Remove NaN values
            valid = ~(s1.isna() | s2.isna())
            if valid.sum() >= 2:
                corr_metrics = self.calculate_correlation(s1[valid], s2[valid])
                corr_metrics.update({
                    'lag': lag,
                    'direction': direction
                })
                results.append(corr_metrics)
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        cols = ['lag', 'direction', 'n_observations', 'pearson_correlation',
                'pearson_p_value', 'spearman_correlation', 'spearman_p_value', 'valid']
        return df[cols].sort_values('lag')
    
    def process(self) -> Dict[str, Any]:
        """
        Process the data and return correlation analysis results.
        
        Returns:
            Dictionary containing correlation metrics and aligned data
        """
        # Align data
        aligned_sentiment, aligned_returns = self.align_data()
        
        if aligned_sentiment.empty or aligned_returns.empty:
            return {
                'success': False,
                'error': 'No common dates between sentiment and returns data'
            }
        
        # Calculate correlations
        correlations = self.calculate_correlation(
            aligned_sentiment['mean_polarity'],
            aligned_returns['returns']
        )
        
        # Calculate lagged correlations
        lagged_correlations = self.analyze_lagged_correlations()
        
        return {
            'success': True,
            'correlations': correlations,
            'lagged_correlations': lagged_correlations,
            'aligned_sentiment': aligned_sentiment,
            'aligned_returns': aligned_returns
        }
