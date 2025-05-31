"""
Sentiment analysis for financial news headlines.
"""
from textblob import TextBlob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from .base_analyzer import BaseAnalyzer

class SentimentAnalyzer(BaseAnalyzer):
    """Analyzer for sentiment analysis of news headlines."""
    
    def __init__(self, data: pd.DataFrame, text_column: str, date_column: str, symbol_column: Optional[str] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            data: DataFrame containing news headlines
            text_column: Name of column containing text to analyze
            date_column: Name of column containing dates
            symbol_column: Optional name of column containing stock symbols
        """
        self.text_column = text_column
        self.date_column = date_column
        self.symbol_column = symbol_column
        super().__init__(data)
    
    def _validate_data(self) -> None:
        """Validate input data has required columns."""
        required_cols = {self.text_column, self.date_column}
        if self.symbol_column:
            required_cols.add(self.symbol_column)
            
        missing_cols = required_cols - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with polarity and subjectivity scores
        """
        analysis = TextBlob(str(text))
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
    
    def get_sentiment_label(self, polarity: float) -> str:
        """
        Convert polarity score to sentiment label.
        
        Args:
            polarity: Sentiment polarity score
            
        Returns:
            Sentiment label ('positive', 'negative', or 'neutral')
        """
        if polarity > 0:
            return 'positive'
        elif polarity < 0:
            return 'negative'
        return 'neutral'
    
    def process(self) -> pd.DataFrame:
        """
        Process all headlines and return sentiment scores.
        
        Returns:
            DataFrame with sentiment scores and labels for each headline
        """
        results = []
        for _, row in self.data.iterrows():
            if pd.isna(row[self.text_column]):
                sentiment = {'polarity': 0.0, 'subjectivity': 0.0}
                label = 'neutral'
            else:
                sentiment = self.analyze_sentiment(row[self.text_column])
                label = self.get_sentiment_label(sentiment['polarity'])
            
            result = {
                'date': row[self.date_column],
                'text': row[self.text_column],
                'polarity': sentiment['polarity'],
                'subjectivity': sentiment['subjectivity'],
                'label': label
            }
            
            if self.symbol_column:
                result['symbol'] = row[self.symbol_column]
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def aggregate_daily(self, min_headlines: int = 1) -> pd.DataFrame:
        """
        Aggregate sentiment scores by date and optionally by symbol.
        
        Args:
            min_headlines: Minimum number of headlines required per day
            
        Returns:
            DataFrame with daily aggregated sentiment metrics
        """
        processed = self.process()
        
        # Convert date to datetime
        processed['date'] = pd.to_datetime(processed['date'])
        
        # Group by date and optionally symbol
        group_cols = ['date']
        if self.symbol_column and self.symbol_column in processed.columns:
            group_cols.append('symbol')
        
        daily = processed.groupby(group_cols).agg({
            'polarity': ['mean', 'std', 'count'],
            'subjectivity': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily.columns = [
            'date',
            'symbol' if len(group_cols) > 1 else 'mean_polarity',
            'mean_polarity' if len(group_cols) > 1 else 'std_polarity',
            'std_polarity' if len(group_cols) > 1 else 'headline_count',
            'mean_subjectivity'
        ]
        
        # Filter by minimum headlines
        if min_headlines > 1:
            daily = daily[daily['headline_count'] >= min_headlines]
        
        return daily
    
    def get_coverage_stats(self) -> Dict[str, Any]:
        """Get statistics about sentiment data coverage."""
        processed = self.process()
        total_headlines = len(processed)
        unique_dates = processed['date'].nunique()
        
        stats = {
            'total_headlines': total_headlines,
            'unique_dates': unique_dates,
            'avg_headlines_per_day': total_headlines / unique_dates if unique_dates > 0 else 0
        }
        
        if self.symbol_column and self.symbol_column in processed.columns:
            stats.update({
                'unique_symbols': processed['symbol'].nunique(),
                'headlines_per_symbol': processed.groupby('symbol').size().to_dict()
            })
        
        return stats
