"""
Feature engineering module for the recommendation system.
Creates user and item features for the Dual-Tower architecture.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering class for user and item features."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        self.location_encoder = LabelEncoder()
        self.author_encoder = LabelEncoder()
        self.publisher_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Feature mappings for consistency
        self.user_id_mapping = {}
        self.item_id_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def create_user_features(self, users_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive user features.
        
        Args:
            users_df: Users dataframe
            ratings_df: Ratings dataframe
            
        Returns:
            DataFrame with user features
        """
        logger.info("Creating user features...")
        
        users_features = users_df.copy()
        
        # 1. Basic demographic features
        users_features = self._process_age_features(users_features)
        users_features = self._process_location_features(users_features)
        
        # 2. Interaction-based features
        user_stats = self._compute_user_interaction_stats(ratings_df)
        users_features = users_features.merge(user_stats, on='User-ID', how='left')
        
        # 3. Reading behavior features
        reading_patterns = self._compute_reading_patterns(ratings_df)
        users_features = users_features.merge(reading_patterns, on='User-ID', how='left')
        
        # 4. Create user ID mapping
        unique_users = sorted(users_features['User-ID'].unique())
        self.user_id_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_id_mapping.items()}
        
        users_features['user_idx'] = users_features['User-ID'].map(self.user_id_mapping)
        
        # 5. Fill missing values
        users_features = self._fill_user_missing_values(users_features)
        
        logger.info(f"Created user features with shape: {users_features.shape}")
        return users_features
    
    def create_item_features(self, books_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive item (book) features.
        
        Args:
            books_df: Books dataframe
            ratings_df: Ratings dataframe
            
        Returns:
            DataFrame with item features
        """
        logger.info("Creating item features...")
        
        books_features = books_df.copy()
        
        # 1. Basic metadata features
        books_features = self._process_publication_year(books_features)
        books_features = self._process_author_features(books_features)
        books_features = self._process_publisher_features(books_features)
        
        # 2. Text-based features
        books_features = self._create_text_features(books_features)
        
        # 3. Popularity and rating-based features
        book_stats = self._compute_book_stats(ratings_df)
        books_features = books_features.merge(book_stats, on='ISBN', how='left')
        
        # 4. Content-based features
        books_features = self._create_content_features(books_features)
        
        # 5. Create item ID mapping
        unique_items = sorted(books_features['ISBN'].unique())
        self.item_id_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_id_mapping.items()}
        
        books_features['item_idx'] = books_features['ISBN'].map(self.item_id_mapping)
        
        # 6. Fill missing values
        books_features = self._fill_item_missing_values(books_features)
        
        logger.info(f"Created item features with shape: {books_features.shape}")
        return books_features
    
    def create_interaction_features(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction-based features.
        
        Args:
            ratings_df: Ratings dataframe
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        interactions = ratings_df.copy()
        
        # Map to indices
        interactions['user_idx'] = interactions['User-ID'].map(self.user_id_mapping)
        interactions['item_idx'] = interactions['ISBN'].map(self.item_id_mapping)
        
        # Remove unmapped interactions
        interactions = interactions.dropna(subset=['user_idx', 'item_idx'])
        
        # Normalize ratings to 0-1 scale
        interactions['rating_normalized'] = interactions['Book-Rating'] / 10.0
        
        # Create binary implicit feedback
        interactions['implicit_feedback'] = (interactions['Book-Rating'] > 0).astype(int)
        
        # Add temporal features if needed (using dummy timestamp for now)
        interactions['timestamp'] = pd.Timestamp.now()
        
        logger.info(f"Created interaction features with shape: {interactions.shape}")
        return interactions
    
    def _process_age_features(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Process age-related features."""
        users_df = users_df.copy()
        
        # Convert age to numeric
        users_df['Age'] = pd.to_numeric(users_df['Age'], errors='coerce')
        
        # Create age groups
        age_bins = [0, 18, 25, 35, 50, 65, 100]
        age_labels = ['child', 'young_adult', 'adult', 'middle_aged', 'senior', 'elderly']
        users_df['age_group'] = pd.cut(users_df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
        
        # Age normalization (will be done later with scaler)
        users_df['age_normalized'] = users_df['Age']
        
        return users_df
    
    def _process_location_features(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Process location-related features."""
        users_df = users_df.copy()
        
        if 'Location' in users_df.columns:
            # Clean location data
            users_df['Location'] = users_df['Location'].fillna('Unknown')
            users_df['Location'] = users_df['Location'].str.strip()
            
            # Extract country (assuming format: "city, state, country")
            def extract_country(location):
                if pd.isna(location) or location == 'Unknown':
                    return 'Unknown'
                parts = str(location).split(',')
                return parts[-1].strip() if len(parts) > 0 else 'Unknown'
            
            users_df['country'] = users_df['Location'].apply(extract_country)
            
            # Create location popularity feature
            location_counts = users_df['country'].value_counts()
            users_df['location_popularity'] = users_df['country'].map(location_counts)
        
        return users_df
    
    def _compute_user_interaction_stats(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Compute user interaction statistics."""
        user_stats = ratings_df.groupby('User-ID').agg({
            'Book-Rating': ['count', 'mean', 'std', 'min', 'max'],
            'ISBN': 'nunique'
        }).round(4)
        
        # Flatten column names
        user_stats.columns = [
            'total_ratings', 'avg_rating', 'rating_std', 'min_rating', 'max_rating', 'unique_books'
        ]
        
        # Rating frequency features
        user_stats['rating_frequency'] = user_stats['total_ratings']
        user_stats['rating_diversity'] = user_stats['rating_std'].fillna(0)
        
        # User activity level
        user_stats['activity_level'] = pd.cut(
            user_stats['total_ratings'], 
            bins=[0, 10, 50, 100, float('inf')], 
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        return user_stats.reset_index()
    
    def _compute_reading_patterns(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Compute reading behavior patterns."""
        # Rating distribution patterns
        rating_patterns = []
        
        for user_id in ratings_df['User-ID'].unique():
            user_ratings = ratings_df[ratings_df['User-ID'] == user_id]['Book-Rating']
            
            # Rating distribution
            rating_dist = user_ratings.value_counts(normalize=True).to_dict()
            
            patterns = {
                'User-ID': user_id,
                'rating_entropy': -sum(p * np.log2(p) for p in rating_dist.values() if p > 0),
                'high_rating_ratio': (user_ratings >= 7).mean(),
                'low_rating_ratio': (user_ratings <= 3).mean(),
                'rating_variance': user_ratings.var()
            }
            
            rating_patterns.append(patterns)
        
        return pd.DataFrame(rating_patterns)
    
    def _process_publication_year(self, books_df: pd.DataFrame) -> pd.DataFrame:
        """Process publication year features."""
        books_df = books_df.copy()
        
        if 'Year-Of-Publication' in books_df.columns:
            books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce')
            
            # Current year for age calculation
            current_year = pd.Timestamp.now().year
            books_df['book_age'] = current_year - books_df['Year-Of-Publication']
            
            # Publication era
            def get_era(year):
                if pd.isna(year):
                    return 'Unknown'
                elif year < 1950:
                    return 'Classic'
                elif year < 1980:
                    return 'Mid-Century'
                elif year < 2000:
                    return 'Modern'
                else:
                    return 'Contemporary'
            
            books_df['publication_era'] = books_df['Year-Of-Publication'].apply(get_era)
        
        return books_df
    
    def _process_author_features(self, books_df: pd.DataFrame) -> pd.DataFrame:
        """Process author-related features."""
        books_df = books_df.copy()
        
        if 'Book-Author' in books_df.columns:
            books_df['Book-Author'] = books_df['Book-Author'].fillna('Unknown')
            
            # Author popularity (number of books)
            author_counts = books_df['Book-Author'].value_counts()
            books_df['author_popularity'] = books_df['Book-Author'].map(author_counts)
            
            # Author productivity categories
            books_df['author_productivity'] = pd.cut(
                books_df['author_popularity'],
                bins=[0, 1, 5, 20, float('inf')],
                labels=['single_book', 'few_books', 'prolific', 'very_prolific']
            )
        
        return books_df
    
    def _process_publisher_features(self, books_df: pd.DataFrame) -> pd.DataFrame:
        """Process publisher-related features."""
        books_df = books_df.copy()
        
        if 'Publisher' in books_df.columns:
            books_df['Publisher'] = books_df['Publisher'].fillna('Unknown')
            
            # Publisher popularity
            publisher_counts = books_df['Publisher'].value_counts()
            books_df['publisher_popularity'] = books_df['Publisher'].map(publisher_counts)
            
            # Publisher size categories
            books_df['publisher_size'] = pd.cut(
                books_df['publisher_popularity'],
                bins=[0, 1, 10, 100, float('inf')],
                labels=['small', 'medium', 'large', 'major']
            )
        
        return books_df
    
    def _create_text_features(self, books_df: pd.DataFrame) -> pd.DataFrame:
        """Create text-based features from titles."""
        books_df = books_df.copy()
        
        if 'Book-Title' in books_df.columns:
            # Clean titles
            books_df['Book-Title'] = books_df['Book-Title'].fillna('')
            
            # Title length features
            books_df['title_length'] = books_df['Book-Title'].str.len()
            books_df['title_word_count'] = books_df['Book-Title'].str.split().str.len()
            
            # Title characteristics
            books_df['has_subtitle'] = books_df['Book-Title'].str.contains(':', na=False).astype(int)
            books_df['has_series_number'] = books_df['Book-Title'].str.contains(r'\d+', na=False).astype(int)
            books_df['title_complexity'] = books_df['title_word_count'] / (books_df['title_length'] + 1)
        
        return books_df
    
    def _compute_book_stats(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Compute book popularity and rating statistics."""
        book_stats = ratings_df.groupby('ISBN').agg({
            'Book-Rating': ['count', 'mean', 'std', 'min', 'max'],
            'User-ID': 'nunique'
        }).round(4)
        
        # Flatten column names
        book_stats.columns = [
            'total_ratings', 'avg_rating', 'rating_std', 'min_rating', 'max_rating', 'unique_users'
        ]
        
        # Popularity metrics
        book_stats['popularity_score'] = book_stats['total_ratings'] * book_stats['avg_rating']
        book_stats['rating_consistency'] = 1 / (1 + book_stats['rating_std'].fillna(0))
        
        # Popularity categories
        book_stats['popularity_category'] = pd.cut(
            book_stats['total_ratings'],
            bins=[0, 5, 20, 100, float('inf')],
            labels=['niche', 'moderate', 'popular', 'blockbuster']
        )
        
        return book_stats.reset_index()
    
    def _create_content_features(self, books_df: pd.DataFrame) -> pd.DataFrame:
        """Create content-based features."""
        books_df = books_df.copy()
        
        # Create combined text for content analysis
        text_columns = []
        if 'Book-Title' in books_df.columns:
            text_columns.append('Book-Title')
        if 'Book-Author' in books_df.columns:
            text_columns.append('Book-Author')
        
        if text_columns:
            books_df['combined_text'] = books_df[text_columns].fillna('').apply(
                lambda x: ' '.join(x.astype(str)), axis=1
            )
        else:
            books_df['combined_text'] = ''
        
        return books_df
    
    def _fill_user_missing_values(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in user features."""
        users_df = users_df.copy()
        
        # Fill age with median
        if 'Age' in users_df.columns:
            median_age = users_df['Age'].median()
            users_df['Age'] = users_df['Age'].fillna(median_age)
            users_df['age_normalized'] = users_df['age_normalized'].fillna(median_age)
        
        # Fill location features
        if 'country' in users_df.columns:
            users_df['country'] = users_df['country'].fillna('Unknown')
            users_df['location_popularity'] = users_df['location_popularity'].fillna(1)
        
        # Fill interaction stats with 0
        numeric_columns = users_df.select_dtypes(include=[np.number]).columns
        users_df[numeric_columns] = users_df[numeric_columns].fillna(0)
        
        return users_df
    
    def _fill_item_missing_values(self, books_df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in item features."""
        books_df = books_df.copy()
        
        # Fill publication year with median
        if 'Year-Of-Publication' in books_df.columns:
            median_year = books_df['Year-Of-Publication'].median()
            books_df['Year-Of-Publication'] = books_df['Year-Of-Publication'].fillna(median_year)
            books_df['book_age'] = books_df['book_age'].fillna(
                pd.Timestamp.now().year - median_year
            )
        
        # Fill text features
        text_features = ['title_length', 'title_word_count', 'title_complexity']
        for feature in text_features:
            if feature in books_df.columns:
                books_df[feature] = books_df[feature].fillna(0)
        
        # Fill popularity features with 1 (minimum count)
        popularity_features = ['author_popularity', 'publisher_popularity']
        for feature in popularity_features:
            if feature in books_df.columns:
                books_df[feature] = books_df[feature].fillna(1)
        
        # Fill numeric columns with 0
        numeric_columns = books_df.select_dtypes(include=[np.number]).columns
        books_df[numeric_columns] = books_df[numeric_columns].fillna(0)
        
        return books_df
    
    def prepare_final_features(self, users_features: pd.DataFrame, 
                             books_features: pd.DataFrame,
                             interactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare final feature matrices for model training.
        
        Args:
            users_features: User features DataFrame
            books_features: Book features DataFrame
            interactions: Interactions DataFrame
            
        Returns:
            Tuple of (user_features_final, item_features_final, interactions_final)
        """
        logger.info("Preparing final feature matrices...")
        
        # Select and scale user features
        user_numeric_features = [
            'user_idx', 'age_normalized', 'location_popularity', 
            'total_ratings', 'avg_rating', 'rating_std', 'unique_books',
            'rating_frequency', 'rating_diversity', 'rating_entropy',
            'high_rating_ratio', 'low_rating_ratio', 'rating_variance'
        ]
        
        # Only keep features that exist
        user_features_final = users_features[[col for col in user_numeric_features if col in users_features.columns]].copy()
        
        # Select and scale item features
        item_numeric_features = [
            'item_idx', 'Year-Of-Publication', 'book_age', 'author_popularity',
            'publisher_popularity', 'title_length', 'title_word_count',
            'title_complexity', 'total_ratings', 'avg_rating', 'rating_std',
            'unique_users', 'popularity_score', 'rating_consistency'
        ]
        
        # Only keep features that exist
        item_features_final = books_features[[col for col in item_numeric_features if col in books_features.columns]].copy()
        
        # Prepare interactions
        interaction_features = [
            'user_idx', 'item_idx', 'Book-Rating', 'rating_normalized', 'implicit_feedback'
        ]
        
        interactions_final = interactions[[col for col in interaction_features if col in interactions.columns]].copy()
        
        # Scale numeric features (except indices)
        user_scale_features = [col for col in user_features_final.columns if col != 'user_idx']
        if user_scale_features:
            user_features_final[user_scale_features] = self.user_scaler.fit_transform(
                user_features_final[user_scale_features]
            )
        
        item_scale_features = [col for col in item_features_final.columns if col != 'item_idx']
        if item_scale_features:
            item_features_final[item_scale_features] = self.item_scaler.fit_transform(
                item_features_final[item_scale_features]
            )
        
        logger.info(f"Final user features shape: {user_features_final.shape}")
        logger.info(f"Final item features shape: {item_features_final.shape}")
        logger.info(f"Final interactions shape: {interactions_final.shape}")
        
        return user_features_final, item_features_final, interactions_final


def create_features_pipeline(books_df: pd.DataFrame, 
                           users_df: pd.DataFrame, 
                           ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureEngineer]:
    """
    Complete feature engineering pipeline.
    
    Args:
        books_df: Books DataFrame
        users_df: Users DataFrame
        ratings_df: Ratings DataFrame
        
    Returns:
        Tuple of (user_features, item_features, interactions, feature_engineer)
    """
    feature_engineer = FeatureEngineer()
    
    # Create features
    user_features = feature_engineer.create_user_features(users_df, ratings_df)
    item_features = feature_engineer.create_item_features(books_df, ratings_df)
    interactions = feature_engineer.create_interaction_features(ratings_df)
    
    # Prepare final features
    user_features_final, item_features_final, interactions_final = feature_engineer.prepare_final_features(
        user_features, item_features, interactions
    )
    
    return user_features_final, item_features_final, interactions_final, feature_engineer