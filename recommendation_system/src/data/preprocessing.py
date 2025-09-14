"""
Data preprocessing module for the recommendation system.
Handles loading, cleaning, and basic preprocessing of the datasets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Main data preprocessing class for the recommendation system."""
    
    def __init__(self, data_path: str = "../../"):
        """
        Initialize the data preprocessor.
        
        Args:
            data_path: Path to the directory containing CSV files
        """
        self.data_path = Path(data_path)
        self.books_df = None
        self.users_df = None
        self.ratings_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all datasets from CSV files.
        
        Returns:
            Tuple of (books_df, users_df, ratings_df)
        """
        try:
            logger.info("Loading datasets...")
            
            # Load books data
            books_path = self.data_path / "Books.csv"
            self.books_df = pd.read_csv(books_path, encoding='latin-1', low_memory=False)
            logger.info(f"Loaded books data: {self.books_df.shape}")
            
            # Load users data
            users_path = self.data_path / "Users.csv"
            self.users_df = pd.read_csv(users_path, encoding='latin-1', low_memory=False)
            logger.info(f"Loaded users data: {self.users_df.shape}")
            
            # Load ratings data
            ratings_path = self.data_path / "Ratings.csv"
            self.ratings_df = pd.read_csv(ratings_path, encoding='latin-1', low_memory=False)
            logger.info(f"Loaded ratings data: {self.ratings_df.shape}")
            
            return self.books_df, self.users_df, self.ratings_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get basic information about the datasets.
        
        Returns:
            Dictionary containing dataset information
        """
        if any(df is None for df in [self.books_df, self.users_df, self.ratings_df]):
            raise ValueError("Data not loaded. Call load_data() first.")
        
        info = {
            'books': {
                'shape': self.books_df.shape,
                'columns': list(self.books_df.columns),
                'missing_values': self.books_df.isnull().sum().to_dict(),
                'dtypes': self.books_df.dtypes.to_dict()
            },
            'users': {
                'shape': self.users_df.shape,
                'columns': list(self.users_df.columns),
                'missing_values': self.users_df.isnull().sum().to_dict(),
                'dtypes': self.users_df.dtypes.to_dict()
            },
            'ratings': {
                'shape': self.ratings_df.shape,
                'columns': list(self.ratings_df.columns),
                'missing_values': self.ratings_df.isnull().sum().to_dict(),
                'dtypes': self.ratings_df.dtypes.to_dict()
            }
        }
        
        return info
    
    def clean_books_data(self) -> pd.DataFrame:
        """
        Clean and preprocess books data.
        
        Returns:
            Cleaned books DataFrame
        """
        logger.info("Cleaning books data...")
        
        books_clean = self.books_df.copy()
        
        # Remove quotes from column names if present
        books_clean.columns = books_clean.columns.str.replace('"', '')
        
        # Handle missing values
        books_clean = books_clean.dropna(subset=['ISBN'])
        
        # Clean publication year
        if 'Year-Of-Publication' in books_clean.columns:
            books_clean['Year-Of-Publication'] = pd.to_numeric(
                books_clean['Year-Of-Publication'], errors='coerce'
            )
            # Remove outliers (e.g., future years or very old years)
            current_year = pd.Timestamp.now().year
            books_clean = books_clean[
                (books_clean['Year-Of-Publication'] >= 1900) & 
                (books_clean['Year-Of-Publication'] <= current_year)
            ]
        
        # Clean book titles and authors
        if 'Book-Title' in books_clean.columns:
            books_clean['Book-Title'] = books_clean['Book-Title'].str.strip()
            
        if 'Book-Author' in books_clean.columns:
            books_clean['Book-Author'] = books_clean['Book-Author'].str.strip()
        
        logger.info(f"Books data cleaned: {books_clean.shape}")
        return books_clean
    
    def clean_users_data(self) -> pd.DataFrame:
        """
        Clean and preprocess users data.
        
        Returns:
            Cleaned users DataFrame
        """
        logger.info("Cleaning users data...")
        
        users_clean = self.users_df.copy()
        
        # Remove quotes from column names if present
        users_clean.columns = users_clean.columns.str.replace('"', '')
        
        # Handle missing values in User-ID
        users_clean = users_clean.dropna(subset=['User-ID'])
        
        # Clean age data
        if 'Age' in users_clean.columns:
            users_clean['Age'] = pd.to_numeric(users_clean['Age'], errors='coerce')
            # Remove unrealistic ages
            users_clean = users_clean[
                (users_clean['Age'].isna()) | 
                ((users_clean['Age'] >= 5) & (users_clean['Age'] <= 100))
            ]
        
        # Clean location data
        if 'Location' in users_clean.columns:
            users_clean['Location'] = users_clean['Location'].str.strip()
        
        logger.info(f"Users data cleaned: {users_clean.shape}")
        return users_clean
    
    def clean_ratings_data(self, valid_users: set = None, valid_books: set = None) -> pd.DataFrame:
        """
        Clean and preprocess ratings data.
        
        Args:
            valid_users: Set of valid user IDs
            valid_books: Set of valid book ISBNs
            
        Returns:
            Cleaned ratings DataFrame
        """
        logger.info("Cleaning ratings data...")
        
        ratings_clean = self.ratings_df.copy()
        
        # Remove quotes from column names if present
        ratings_clean.columns = ratings_clean.columns.str.replace('"', '')
        
        # Remove rows with missing essential values
        ratings_clean = ratings_clean.dropna(subset=['User-ID', 'ISBN', 'Book-Rating'])
        
        # Filter valid ratings (0-10 scale)
        if 'Book-Rating' in ratings_clean.columns:
            ratings_clean = ratings_clean[
                (ratings_clean['Book-Rating'] >= 0) & 
                (ratings_clean['Book-Rating'] <= 10)
            ]
        
        # Filter by valid users and books if provided
        if valid_users is not None:
            ratings_clean = ratings_clean[ratings_clean['User-ID'].isin(valid_users)]
            
        if valid_books is not None:
            ratings_clean = ratings_clean[ratings_clean['ISBN'].isin(valid_books)]
        
        logger.info(f"Ratings data cleaned: {ratings_clean.shape}")
        return ratings_clean
    
    def filter_sparse_data(self, min_user_ratings: int = 5, min_book_ratings: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Filter out sparse users and books to reduce sparsity.
        
        Args:
            min_user_ratings: Minimum number of ratings per user
            min_book_ratings: Minimum number of ratings per book
            
        Returns:
            Tuple of filtered (books_df, users_df, ratings_df)
        """
        logger.info("Filtering sparse data...")
        
        # Start with clean data
        books_clean = self.clean_books_data()
        users_clean = self.clean_users_data()
        ratings_clean = self.clean_ratings_data()
        
        # Iteratively filter sparse data
        prev_ratings_count = len(ratings_clean)
        iteration = 0
        
        while True:
            iteration += 1
            logger.info(f"Filtering iteration {iteration}")
            
            # Count ratings per user and book
            user_counts = ratings_clean['User-ID'].value_counts()
            book_counts = ratings_clean['ISBN'].value_counts()
            
            # Get users and books with enough ratings
            valid_users = set(user_counts[user_counts >= min_user_ratings].index)
            valid_books = set(book_counts[book_counts >= min_book_ratings].index)
            
            # Filter ratings
            ratings_filtered = ratings_clean[
                ratings_clean['User-ID'].isin(valid_users) & 
                ratings_clean['ISBN'].isin(valid_books)
            ]
            
            # Check convergence
            if len(ratings_filtered) == prev_ratings_count:
                break
                
            ratings_clean = ratings_filtered
            prev_ratings_count = len(ratings_clean)
            
            if iteration > 10:  # Safety break
                logger.warning("Maximum filtering iterations reached")
                break
        
        # Filter books and users to match remaining ratings
        final_users = set(ratings_clean['User-ID'].unique())
        final_books = set(ratings_clean['ISBN'].unique())
        
        books_filtered = books_clean[books_clean['ISBN'].isin(final_books)]
        users_filtered = users_clean[users_clean['User-ID'].isin(final_users)]
        
        logger.info(f"Final filtered data - Books: {len(books_filtered)}, Users: {len(users_filtered)}, Ratings: {len(ratings_clean)}")
        
        return books_filtered, users_filtered, ratings_clean


def load_and_preprocess_data(data_path: str = "../../", 
                           min_user_ratings: int = 5, 
                           min_book_ratings: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenient function to load and preprocess all data.
    
    Args:
        data_path: Path to data directory
        min_user_ratings: Minimum ratings per user
        min_book_ratings: Minimum ratings per book
        
    Returns:
        Tuple of preprocessed (books_df, users_df, ratings_df)
    """
    preprocessor = DataPreprocessor(data_path)
    preprocessor.load_data()
    
    return preprocessor.filter_sparse_data(
        min_user_ratings=min_user_ratings,
        min_book_ratings=min_book_ratings
    )