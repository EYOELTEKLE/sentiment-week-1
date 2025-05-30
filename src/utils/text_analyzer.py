"""
Module for text analysis functionality including preprocessing and feature extraction.
"""
from typing import List, Dict, Any, Union
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim import corpora, models
from pandarallel import pandarallel
from numba import jit
import swifter
import dask.dataframe as dd
from tqdm import tqdm
import random


# Initialize parallel processing
pandarallel.initialize(progress_bar=True)

class TextAnalyzer:
    """Class for analyzing text data from news articles."""
    
    def __init__(self):
        """Initialize the TextAnalyzer with necessary NLTK downloads."""
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")
            self.stop_words = set()

    @staticmethod
    def _calculate_stats(lengths: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """Calculate text statistics using numpy/pandas operations."""
        # Convert to numpy array if it's a pandas Series
        if isinstance(lengths, pd.Series):
            lengths = lengths.to_numpy()
            
    # Handle empty input gracefully
        if lengths.size == 0:
            return {
                'mean_length': 0.0,   # Or float('nan') if that's more appropriate
                'max_length': 0.0,    # Or float('nan'), or specific handling
                'min_length': 0.0,    # Or float('nan'), or specific handling
                'total_texts': 0
            }

        return {
            'mean_length': float(np.mean(lengths)),
            'max_length': float(np.max(lengths)),
            'min_length': float(np.min(lengths)),
            'total_texts': int(lengths.size) # or len(lengths)
        }

    def preprocess_text(self, text: Union[str, pd.Series]) -> Union[List[str], pd.Series]:
        """
        Preprocess text by tokenizing and removing stopwords.
        Optimized for both single strings and pandas Series.
        
        Args:
            text: Input text to preprocess (str or pd.Series)
            
        Returns:
            Preprocessed tokens (List[str] or pd.Series)
        """
        if isinstance(text, pd.Series):
            # Use pandas apply for processing series
            return text.apply(self._preprocess_single_text)
        return self._preprocess_single_text(text)

    def _preprocess_single_text(self, text: str) -> List[str]:
        """Process a single text string."""
        # Tokenize and convert to lower case
        tokens = word_tokenize(str(text).lower())
        # Remove stopwords and non-alphabetic tokens
        return [token for token in tokens if token.isalpha() and token not in self.stop_words]

    def get_text_statistics(self, texts: Union[List[str], pd.Series]) -> Dict[str, Any]:
        """
        Calculate basic statistics for texts.
        Optimized for both lists and pandas Series.
        
        Args:
            texts: List of text strings or pandas Series
            
        Returns:
            Dict[str, Any]: Dictionary containing text statistics
        """
        if isinstance(texts, pd.Series):
            # Convert to dask for out-of-memory processing if needed
            if len(texts) > 1_000_000:  # Threshold for using dask
                ddf = dd.from_pandas(pd.DataFrame({'text': texts}), npartitions=4)
                lengths = ddf['text'].str.len().compute()
                print("analyzed huge DS")
            else:
                lengths = texts.str.len().values
                print("analyzed min DS")
        else:
            lengths = np.array([len(str(text)) for text in texts])
        
        return self._calculate_stats(lengths)

    def extract_common_words(self, texts: Union[List[str], pd.Series], top_n: int = 10, batch_size: int = 10000) -> List[tuple]:
        """
        Extract most common words from texts.
        
        Args:
            texts (List[str]): List of text strings
            top_n (int): Number of top words to return
            
        Returns:
            List[tuple]: List of (word, frequency) tuples
        """
        if isinstance(texts, pd.Series):
            # Process in batches for large datasets
            counter = Counter()
            for i in tqdm(range(0, len(texts), batch_size), desc="Processing text batches"):
                batch = texts.iloc[i:i + batch_size]
                tokens = self.preprocess_text(batch)
                batch_tokens = [token for token_list in tokens for token in token_list]
                counter.update(batch_tokens)
        else:
            # Process list of texts
            all_tokens = []
            for text in tqdm(texts, desc="Processing texts"):
                all_tokens.extend(self.preprocess_text(text))
            counter = Counter(all_tokens)
        
        return counter.most_common(top_n)

    def generate_wordcloud(self, texts: List[str], **kwargs) -> WordCloud:
        """
        Generate WordCloud from texts.
        
        Args:
            texts (List[str]): List of text strings
            **kwargs: Additional arguments for WordCloud
            
        Returns:
            WordCloud: Generated WordCloud object
        """
        combined_text = ' '.join([' '.join(self.preprocess_text(text)) for text in texts])
        return WordCloud(width=800, height=400, background_color='white', **kwargs).generate(combined_text)

    def perform_topic_modeling(self, texts: List[str], num_topics: int = 5, sample_size: int = 10000) -> tuple:
        """
        Perform LDA topic modeling on texts.
        
        Args:
            texts (List[str]): List of text strings
            num_topics (int): Number of topics to extract
            
        Returns:
            tuple: (lda_model, corpus, dictionary)
        """
        # Sample for speed (optional)
        if len(texts) > sample_size:
            texts = random.sample(list(texts), sample_size)
        # Preprocess with progress bar
        processed_texts = [self.preprocess_text(text) for text in tqdm(texts, desc="Preprocessing for LDA")]
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        lda_model = models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            workers=4
        )
        return lda_model, corpus, dictionary
