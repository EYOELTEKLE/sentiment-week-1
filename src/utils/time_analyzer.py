"""
Module for time series analysis of news articles.
"""
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class TimeAnalyzer:
    """Class for analyzing temporal patterns in news articles."""
    
    def __init__(self, date_column: str = 'publication_date'):
        """
        Initialize TimeAnalyzer.
        
        Args:
            date_column (str): Name of the date column in the dataset
        """
        self.date_column = date_column

    def prepare_time_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare datetime data for analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with processed datetime column
        """
        df = df.copy()
        # Use format='mixed' to handle both timezone-aware and naive datetime strings
        df[self.date_column] = pd.to_datetime(df[self.date_column], format='ISO8601', errors='coerce')
        return df

    def get_publication_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Analyze publication patterns across different time periods.
        
        Args:
            df (pd.DataFrame): Input dataframe with datetime column
            
        Returns:
            Dict[str, pd.Series]: Dictionary containing various time-based patterns
        """
        df = self.prepare_time_data(df)
        
        return {
            'daily_counts': df.groupby(df[self.date_column].dt.date).size(),
            'hourly_counts': df.groupby(df[self.date_column].dt.hour).size(),
            'weekly_counts': df.groupby(df[self.date_column].dt.day_name()).size(),
            'monthly_counts': df.groupby(df[self.date_column].dt.month).size()
        }

    def analyze_temporal_density(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the temporal density of publications.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, Any]: Dictionary containing temporal density metrics
        """
        df = self.prepare_time_data(df)
        
        # Calculate time differences between consecutive publications
        time_diffs = df[self.date_column].sort_values().diff()
        
        return {
            'mean_time_between_publications': time_diffs.mean(),
            'median_time_between_publications': time_diffs.median(),
            'max_time_between_publications': time_diffs.max(),
            'min_time_between_publications': time_diffs.min()
        }

    def plot_time_series(self, df: pd.DataFrame, time_unit: str = 'D', 
                        figsize: tuple = (15, 6)) -> None:
        """
        Plot time series of publication frequency.
        
        Args:
            df (pd.DataFrame): Input dataframe
            time_unit (str): Time unit for resampling ('D' for daily, 'H' for hourly, etc.)
            figsize (tuple): Figure size
        """
        df = self.prepare_time_data(df)
        
        # Resample and plot
        ts = df.groupby(self.date_column).size().resample(time_unit).sum()
        
        plt.figure(figsize=figsize)
        ts.plot()
        plt.title(f'Publication Frequency (by {time_unit})')
        plt.xlabel('Date')
        plt.ylabel('Number of Publications')
        plt.grid(True)
        plt.tight_layout()

    def create_heatmap(self, df: pd.DataFrame, figsize: tuple = (12, 8)) -> None:
        """
        Create a heatmap of publication patterns by day and hour.
        
        Args:
            df (pd.DataFrame): Input dataframe
            figsize (tuple): Figure size
        """
        df = self.prepare_time_data(df)
        
        # Create hour-day matrix
        hour_day_matrix = pd.crosstab(
            df[self.date_column].dt.day_name(),
            df[self.date_column].dt.hour
        )
        
        plt.figure(figsize=figsize)
        sns.heatmap(hour_day_matrix, cmap='YlOrRd', annot=True, fmt='d')
        plt.title('Publication Patterns: Day of Week vs Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        plt.tight_layout()
