"""
Visualization utilities for sentiment and stock analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict

class VisualizationUtils:
    """Utility class for creating visualizations."""
    
    @staticmethod
    def plot_sentiment_distribution(sentiment_data: pd.DataFrame,
                                  polarity_col: str = 'mean_polarity',
                                  symbol_col: Optional[str] = None) -> None:
        """
        Plot distribution of sentiment scores.
        
        Args:
            sentiment_data: DataFrame containing sentiment scores
            polarity_col: Name of polarity score column
            symbol_col: Optional column name for stock symbols
        """
        plt.figure(figsize=(10, 6))
        
        if symbol_col and symbol_col in sentiment_data.columns:
            for symbol in sentiment_data[symbol_col].unique():
                symbol_data = sentiment_data[sentiment_data[symbol_col] == symbol]
                sns.kdeplot(data=symbol_data[polarity_col], label=symbol)
        else:
            sns.histplot(data=sentiment_data, x=polarity_col, kde=True)
        
        plt.title('Distribution of Sentiment Scores')
        plt.xlabel('Polarity Score')
        plt.ylabel('Density')
        if symbol_col:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_sentiment_trends(sentiment_data: pd.DataFrame,
                            date_col: str = 'date',
                            polarity_col: str = 'mean_polarity',
                            symbol_col: Optional[str] = None,
                            window: int = 7) -> None:
        """
        Plot sentiment trends over time.
        
        Args:
            sentiment_data: DataFrame containing sentiment scores
            date_col: Name of date column
            polarity_col: Name of polarity score column
            symbol_col: Optional column name for stock symbols
            window: Rolling window size for moving average
        """
        plt.figure(figsize=(12, 6))
        
        if symbol_col and symbol_col in sentiment_data.columns:
            for symbol in sentiment_data[symbol_col].unique():
                symbol_data = sentiment_data[sentiment_data[symbol_col] == symbol]
                symbol_data = symbol_data.sort_values(date_col)
                rolling_mean = symbol_data[polarity_col].rolling(window=window).mean()
                plt.plot(symbol_data[date_col], rolling_mean, label=f"{symbol} ({window}-day MA)")
        else:
            data = sentiment_data.sort_values(date_col)
            rolling_mean = data[polarity_col].rolling(window=window).mean()
            plt.plot(data[date_col], rolling_mean, label=f"{window}-day MA")
        
        plt.title('Sentiment Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Average Sentiment Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_heatmap(correlation_data: pd.DataFrame,
                               symbols: list) -> None:
        """
        Plot correlation heatmap between stocks.
        
        Args:
            correlation_data: DataFrame containing correlation values
            symbols: List of stock symbols
        """
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_data), k=1)
        sns.heatmap(correlation_data, mask=mask, annot=True, cmap='coolwarm',
                   vmin=-1, vmax=1, center=0, square=True,
                   xticklabels=symbols, yticklabels=symbols)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_sentiment_returns(sentiment_data: pd.DataFrame,
                             returns_data: pd.DataFrame,
                             date_col: str,
                             polarity_col: str = 'mean_polarity',
                             returns_col: str = 'returns',
                             symbol: Optional[str] = None) -> None:
        """
        Plot sentiment scores and returns on dual axes.
        
        Args:
            sentiment_data: DataFrame containing sentiment scores
            returns_data: DataFrame containing returns
            date_col: Name of date column
            polarity_col: Name of polarity score column
            returns_col: Name of returns column
            symbol: Optional stock symbol for title
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot sentiment
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sentiment Score', color=color)
        ax1.plot(sentiment_data[date_col], sentiment_data[polarity_col],
                color=color, label='Sentiment')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Plot returns on secondary y-axis
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Returns', color=color)
        ax2.plot(returns_data[date_col], returns_data[returns_col],
                color=color, label='Returns', alpha=0.6)
        ax2.tick_params(axis='y', labelcolor=color)
        
        title = 'Sentiment vs Returns'
        if symbol:
            title += f' for {symbol}'
        plt.title(title)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_lagged_correlations(lagged_corr_df: pd.DataFrame,
                               symbol: Optional[str] = None) -> None:
        """
        Plot lagged correlations.
        
        Args:
            lagged_corr_df: DataFrame containing lagged correlation results
            symbol: Optional stock symbol for title
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(lagged_corr_df['lag'],
                lagged_corr_df['pearson_correlation'],
                marker='o', label='Pearson')
        plt.plot(lagged_corr_df['lag'],
                lagged_corr_df['spearman_correlation'],
                marker='s', label='Spearman')
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        title = 'Lagged Correlations between Sentiment and Returns'
        if symbol:
            title += f' for {symbol}'
        plt.title(title)
        
        plt.xlabel('Lag (Days)')
        plt.ylabel('Correlation Coefficient')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
