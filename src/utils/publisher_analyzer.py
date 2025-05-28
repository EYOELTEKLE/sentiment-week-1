"""
Module for analyzing publisher patterns and behaviors.
"""
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class PublisherAnalyzer:
    """Class for analyzing publisher-related patterns in news articles."""
    
    def __init__(self, publisher_column: str = 'publisher'):
        """
        Initialize PublisherAnalyzer.
        
        Args:
            publisher_column (str): Name of the publisher column in the dataset
        """
        self.publisher_column = publisher_column

    def extract_domain(self, email: str) -> str:
        """
        Extract domain from email address or return original string if not email.
        
        Args:
            email (str): Input string that might be an email address
            
        Returns:
            str: Domain or original string
        """
        try:
            return email.split('@')[1] if '@' in str(email) else str(email)
        except:
            return str(email)

    def get_publisher_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate basic statistics about publishers.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, Any]: Dictionary containing publisher statistics
        """
        publisher_counts = df[self.publisher_column].value_counts()
        
        return {
            'total_publishers': len(publisher_counts),
            'top_publishers': publisher_counts.head(10).to_dict(),
            'articles_per_publisher_mean': publisher_counts.mean(),
            'articles_per_publisher_median': publisher_counts.median(),
            'most_active_publisher': publisher_counts.index[0],
            'most_active_publisher_articles': publisher_counts.iloc[0]
        }

    def analyze_publisher_domains(self, df: pd.DataFrame) -> pd.Series:
        """
        Analyze publisher domains if publishers are email addresses.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.Series: Domain frequency counts
        """
        domains = df[self.publisher_column].apply(self.extract_domain)
        return domains.value_counts()

    def plot_top_publishers(self, df: pd.DataFrame, top_n: int = 10, 
                          figsize: tuple = (12, 6)) -> None:
        """
        Plot top publishers by article count.
        
        Args:
            df (pd.DataFrame): Input dataframe
            top_n (int): Number of top publishers to show
            figsize (tuple): Figure size
        """
        publisher_counts = df[self.publisher_column].value_counts()
        
        plt.figure(figsize=figsize)
        publisher_counts.head(top_n).plot(kind='bar')
        plt.title(f'Top {top_n} Publishers by Article Count')
        plt.xlabel('Publisher')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

    def analyze_publisher_patterns(self, df: pd.DataFrame, 
                                 date_column: str = 'publication_date') -> pd.DataFrame:
        """
        Analyze publishing patterns for top publishers.
        
        Args:
            df (pd.DataFrame): Input dataframe
            date_column (str): Name of the date column
            
        Returns:
            pd.DataFrame: Publishing patterns by hour for top publishers
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Get top 10 publishers
        top_publishers = df[self.publisher_column].value_counts().head(10).index
        
        # Filter for top publishers
        top_pub_df = df[df[self.publisher_column].isin(top_publishers)]
        
        # Create hour-publisher matrix
        return pd.crosstab(
            top_pub_df[self.publisher_column],
            top_pub_df[date_column].dt.hour
        )
