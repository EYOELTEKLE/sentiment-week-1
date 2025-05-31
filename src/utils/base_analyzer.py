"""
Base class for all analyzers.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional, Union

class BaseAnalyzer(ABC):
    """Base class for all analyzers."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with data.
        
        Args:
            data: Input DataFrame to analyze
        """
        self.data = data.copy()
        self._validate_data()
    
    @abstractmethod
    def _validate_data(self) -> None:
        """Validate input data structure."""
        pass
    
    @abstractmethod
    def process(self) -> pd.DataFrame:
        """Process the data and return results."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the analysis."""
        return {
            'rows': len(self.data),
            'columns': list(self.data.columns),
            'date_range': self._get_date_range()
        }
    
    def _get_date_range(self) -> Optional[str]:
        """Get the date range of the data if available."""
        date_cols = [col for col in self.data.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            start = self.data[date_col].min()
            end = self.data[date_col].max()
            return f"{start} to {end}"
        return None
