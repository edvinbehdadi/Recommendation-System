"""
DLRM Inference Engine for Book Recommendations
Loads trained DLRM model and provides recommendation functionality
"""

import torch
import numpy as np
import pandas as pd
import pickle
import mlflow
from mlflow import MlflowClient
import tempfile
import os
from typing import List, Dict, Tuple, Optional, Any
from functools import partial
import warnings
warnings.filterwarnings('ignore')

from torchrec import EmbeddingBagCollection
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.datasets.utils import Batch

class DLRMBookRecommender:
    """DLRM-based book recommender for inference"""
    
    def __init__(self, model_path: str = None, run_id: str = None):
        """
        Initialize DLRM book recommender
        
        Args:
            model_path: Path to saved model state dict
            run_id: MLflow run ID to load model from
        """
        self.device = torch.device("cpu")
        self.model = None
        self.preprocessing_info = None
        
        # Load preprocessing info
        self._load_preprocessing_info()
        
        # Load model
        if model_path and os.path.exists(model_path):
            self._load_model_from_path(model_path)
        elif run_id:
            self._load_model_from_mlflow(run_id)
        else:
            print("⚠️ No model loaded. Please provide model_path or run_id")
    
    def _load_preprocessing_info(self):
        """Load preprocessing information"""
        if os.path.exists('book_dlrm_preprocessing.pkl'):
            with open('book_dlrm_preprocessing.pkl', 'rb') as f:
                self.preprocessing_info = pickle.load(f)
            
            self.dense_cols = self.preprocessing_info['dense_cols']
            self.cat_cols = self.preprocessing_info['cat_cols']
            self.emb_counts = self.preprocessing_info['emb_counts']
            self.user_encoder = self.preprocessing_info['user_encoder']
            self.book_encoder = self.preprocessing_info['book_encoder']
            self.publisher_encoder = self.preprocessing_info['publisher_encoder']
            self.location_encoder = self.preprocessing_info['location_encoder']
            self.scaler = self.preprocessing_info['scaler']
            
            print("✅ Preprocessing info loaded")
        else:
            raise FileNotFoundError("book_dlrm_preprocessing.pkl not found. Run preprocessing first.")
    
    def _load_model_from_path(self, model_path: str):
        """Load model from saved state dict"""
        try:
            # Create model architecture
            eb_configs = [
                EmbeddingBagConfig(
                    name=f"t_{feature_name}",
                    embedding_dim=64,  # Default embedding dim
                    num_embeddings=self.emb_counts[feature_idx],
                    feature_names=[feature_name],
                )
                for feature_idx, feature_name in enumerate(self.cat_cols)
            ]

            dlrm_model = DLRM(
                embedding_bag_collection=EmbeddingBagCollection(
                    tables=eb_configs, device=self.device
                ),
                dense_in_features=len(self.dense_cols),
                dense_arch_layer_sizes=[256, 128, 64],
                over_arch_layer_sizes=[512, 256, 128, 1],
                dense_device=self.device,
            )

            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Remove 'model.' prefix if present
            if any(key.startswith('model.') for key in state_dict.keys()):
                state_dict = {k[6:]: v for k, v in state_dict.items()}
            
            dlrm_model.load_state_dict(state_dict)
            self.model = dlrm_model
            self.model.eval()
            
            print(f"✅ Model loaded from {model_path}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def _load_model_from_mlflow(self, run_id: str):
        """Load model from MLflow"""
        try:
            client = MlflowClient()
            run = client.get_run(run_id)
            
            # Get model parameters from MLflow
            params = run.data.params
            cat_cols = eval(params.get('cat_cols'))
            emb_counts = eval(params.get('emb_counts'))
            dense_cols = eval(params.get('dense_cols'))
            embedding_dim = int(params.get('embedding_dim', 64))
            dense_arch_layer_sizes = eval(params.get('dense_arch_layer_sizes'))
            over_arch_layer_sizes = eval(params.get('over_arch_layer_sizes'))
            
            # Download model from MLflow
            temp_dir = tempfile.mkdtemp()
            
            # Try different artifact paths
            for artifact_path in ['model_state_dict_final', 'model_state_dict_2', 'model_state_dict_1', 'model_state_dict_0']:
                try:
                    client.download_artifacts(run_id, f"{artifact_path}/state_dict.pth", temp_dir)
                    state_dict = mlflow.pytorch.load_state_dict(f"{temp_dir}/{artifact_path}")
                    break
                except:
                    continue
            else:
                raise Exception("No model artifacts found")
            
            # Create model
            eb_configs = [
                EmbeddingBagConfig(
                    name=f"t_{feature_name}",
                    embedding_dim=embedding_dim,
                    num_embeddings=emb_counts[feature_idx],
                    feature_names=[feature_name],
                )
                for feature_idx, feature_name in enumerate(cat_cols)
            ]

            dlrm_model = DLRM(
                embedding_bag_collection=EmbeddingBagCollection(
                    tables=eb_configs, device=self.device
                ),
                dense_in_features=len(dense_cols),
                dense_arch_layer_sizes=dense_arch_layer_sizes,
                over_arch_layer_sizes=over_arch_layer_sizes,
                dense_device=self.device,
            )

            # Remove prefix and load state dict
            if any(key.startswith('model.') for key in state_dict.keys()):
                state_dict = {k[6:]: v for k, v in state_dict.items()}
            
            dlrm_model.load_state_dict(state_dict)
            self.model = dlrm_model
            self.model.eval()
            
            print(f"✅ Model loaded from MLflow run: {run_id}")
            
        except Exception as e:
            print(f"❌ Error loading model from MLflow: {e}")
    
    def _prepare_user_features(self, user_id: int, user_data: Optional[Dict] = None) -> Tuple[torch.Tensor, KeyedJaggedTensor]:
        """Prepare user features for inference"""
        
        if user_data is None:
            # Create default user features
            user_data = {
                'User-ID': user_id,
                'Age': 30,  # Default age
                'Location': 'usa',  # Default location
            }
        
        # Encode categorical features
        try:
            user_id_encoded = self.user_encoder.transform([str(user_id)])[0]
        except:
            # Handle unknown user
            user_id_encoded = 0
        
        try:
            location = str(user_data.get('Location', 'usa')).split(',')[-1].strip().lower()
            country_encoded = self.location_encoder.transform([location])[0]
        except:
            country_encoded = 0
        
        # Age group
        age = user_data.get('Age', 30)
        if age < 18:
            age_group = 0
        elif age < 25:
            age_group = 1
        elif age < 35:
            age_group = 2
        elif age < 50:
            age_group = 3
        elif age < 65:
            age_group = 4
        else:
            age_group = 5
        
        # Get user statistics (if available)
        user_activity = user_data.get('user_activity', 10)  # Default
        user_avg_rating = user_data.get('user_avg_rating', 6.0)  # Default
        age_normalized = user_data.get('Age', 30)
        
        # Normalize dense features
        dense_features = np.array([[age_normalized, 2000, user_activity, 10, user_avg_rating, 6.0]])  # Default values
        dense_features = self.scaler.transform(dense_features)
        dense_features = torch.tensor(dense_features, dtype=torch.float32)
        
        return dense_features, user_id_encoded, country_encoded, age_group
    
    def _prepare_book_features(self, book_isbn: str, book_data: Optional[Dict] = None) -> Tuple[int, int, int, int]:
        """Prepare book features for inference"""
        
        if book_data is None:
            book_data = {}
        
        # Encode book ID
        try:
            book_id_encoded = self.book_encoder.transform([str(book_isbn)])[0]
        except:
            book_id_encoded = 0
        
        # Encode publisher
        try:
            publisher = str(book_data.get('Publisher', 'Unknown'))
            publisher_encoded = self.publisher_encoder.transform([publisher])[0]
        except:
            publisher_encoded = 0
        
        # Publication decade
        year = book_data.get('Year-Of-Publication', 2000)
        decade = ((int(year) // 10) * 10)
        try:
            decade_encoded = preprocessing_info.get('decade_encoder', LabelEncoder()).transform([str(decade)])[0]
        except:
            decade_encoded = 6  # Default to 2000s
        
        # Rating level (default to medium)
        rating_level = 1
        
        return book_id_encoded, publisher_encoded, decade_encoded, rating_level
    
    def predict_rating(self, user_id: int, book_isbn: str, 
                      user_data: Optional[Dict] = None, 
                      book_data: Optional[Dict] = None) -> float:
        """
        Predict rating probability for user-book pair
        
        Args:
            user_id: User ID
            book_isbn: Book ISBN
            user_data: Additional user data (optional)
            book_data: Additional book data (optional)
            
        Returns:
            Prediction probability (0-1)
        """
        if self.model is None:
            print("❌ Model not loaded")
            return 0.0
        
        try:
            # Prepare features
            dense_features, user_id_encoded, country_encoded, age_group = self._prepare_user_features(user_id, user_data)
            book_id_encoded, publisher_encoded, decade_encoded, rating_level = self._prepare_book_features(book_isbn, book_data)
            
            # Create sparse features
            kjt_values = [user_id_encoded, book_id_encoded, publisher_encoded, country_encoded, age_group, decade_encoded, rating_level]
            kjt_lengths = [1] * len(kjt_values)
            
            sparse_features = KeyedJaggedTensor.from_lengths_sync(
                self.cat_cols,
                torch.tensor(kjt_values),
                torch.tensor(kjt_lengths, dtype=torch.int32),
            )
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(dense_features=dense_features, sparse_features=sparse_features)
                prediction = torch.sigmoid(logits).item()
            
            return prediction
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0.0
    
    def get_user_recommendations(self, user_id: int, 
                               candidate_books: List[str],
                               k: int = 10,
                               user_data: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """
        Get top-k book recommendations for a user
        
        Args:
            user_id: User ID
            candidate_books: List of candidate book ISBNs
            k: Number of recommendations
            user_data: Additional user data
            
        Returns:
            List of (book_isbn, prediction_score) tuples
        """
        if self.model is None:
            print("❌ Model not loaded")
            return []
        
        recommendations = []
        
        print(f"Generating recommendations for user {user_id} from {len(candidate_books)} candidates...")
        
        for book_isbn in candidate_books:
            score = self.predict_rating(user_id, book_isbn, user_data)
            recommendations.append((book_isbn, score))
        
        # Sort by score and return top-k
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:k]
    
    def batch_recommend(self, user_ids: List[int], 
                       candidate_books: List[str],
                       k: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Generate recommendations for multiple users
        
        Args:
            user_ids: List of user IDs
            candidate_books: List of candidate book ISBNs
            k: Number of recommendations per user
            
        Returns:
            Dictionary mapping user_id to recommendations
        """
        results = {}
        
        for user_id in user_ids:
            results[user_id] = self.get_user_recommendations(user_id, candidate_books, k)
        
        return results
    
    def get_similar_books(self, target_book_isbn: str,
                         candidate_books: List[str],
                         sample_users: List[int],
                         k: int = 10) -> List[Tuple[str, float]]:
        """
        Find books similar to target book by comparing user preferences
        
        Args:
            target_book_isbn: Target book ISBN
            candidate_books: List of candidate book ISBNs
            sample_users: Sample users to test similarity with
            k: Number of similar books
            
        Returns:
            List of (book_isbn, similarity_score) tuples
        """
        target_scores = []
        candidate_scores = {book: [] for book in candidate_books}
        
        # Get predictions for target book and candidates across sample users
        for user_id in sample_users:
            target_score = self.predict_rating(user_id, target_book_isbn)
            target_scores.append(target_score)
            
            for book_isbn in candidate_books:
                if book_isbn != target_book_isbn:
                    score = self.predict_rating(user_id, book_isbn)
                    candidate_scores[book_isbn].append(score)
        
        # Calculate similarity based on correlation of user preferences
        similarities = []
        target_scores = np.array(target_scores)
        
        for book_isbn, scores in candidate_scores.items():
            if len(scores) > 0:
                scores_array = np.array(scores)
                # Calculate correlation as similarity measure
                correlation = np.corrcoef(target_scores, scores_array)[0, 1]
                if not np.isnan(correlation):
                    similarities.append((book_isbn, correlation))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]


def load_dlrm_recommender(model_source: str = "latest") -> DLRMBookRecommender:
    """
    Load DLRM recommender from various sources
    
    Args:
        model_source: "latest" for latest MLflow run, "file" for local file, or specific run_id
        
    Returns:
        DLRMBookRecommender instance
    """
    recommender = DLRMBookRecommender()
    
    if model_source == "latest":
        # Try to get latest MLflow run
        try:
            experiment = mlflow.get_experiment_by_name('dlrm-book-recommendation-book_recommender')
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], 
                                        order_by=["start_time desc"], max_results=1)
                if len(runs) > 0:
                    latest_run_id = runs.iloc[0].run_id
                    recommender = DLRMBookRecommender(run_id=latest_run_id)
                    return recommender
        except:
            pass
    
    elif model_source == "file":
        # Try to load from local file
        for filename in ['dlrm_book_model_final.pth', 'dlrm_book_model_epoch_2.pth', 'dlrm_book_model_epoch_1.pth']:
            if os.path.exists(filename):
                recommender = DLRMBookRecommender(model_path=filename)
                return recommender
    
    else:
        # Treat as run_id
        recommender = DLRMBookRecommender(run_id=model_source)
        return recommender
    
    print("⚠️ Could not load any trained model")
    return recommender


def demo_dlrm_recommendations():
    """Demo function to show DLRM recommendations"""
    
    print("🚀 DLRM Book Recommendation Demo")
    print("=" * 50)
    
    # Load book data for demo
    books_df = pd.read_csv('Books.csv', encoding='latin-1', low_memory=False)
    users_df = pd.read_csv('Users.csv', encoding='latin-1', low_memory=False)
    ratings_df = pd.read_csv('Ratings.csv', encoding='latin-1', low_memory=False)
    
    books_df.columns = books_df.columns.str.replace('"', '')
    users_df.columns = users_df.columns.str.replace('"', '')
    ratings_df.columns = ratings_df.columns.str.replace('"', '')
    
    # Load recommender
    recommender = load_dlrm_recommender("file")
    
    if recommender.model is None:
        print("❌ No trained model found. Please run training first.")
        return
    
    # Get sample user and books
    sample_user_id = ratings_df['User-ID'].iloc[0]
    sample_books = books_df['ISBN'].head(20).tolist()
    
    print(f"\n📚 Getting recommendations for User {sample_user_id}")
    print(f"Testing with {len(sample_books)} candidate books...")
    
    # Get recommendations
    recommendations = recommender.get_user_recommendations(
        user_id=sample_user_id,
        candidate_books=sample_books,
        k=10
    )
    
    print(f"\n🎯 Top 10 DLRM Recommendations:")
    print("-" * 50)
    
    for i, (book_isbn, score) in enumerate(recommendations, 1):
        # Get book info
        book_info = books_df[books_df['ISBN'] == book_isbn]
        if len(book_info) > 0:
            book = book_info.iloc[0]
            title = book['Book-Title']
            author = book['Book-Author']
            print(f"{i:2d}. {title} by {author}")
            print(f"    ISBN: {book_isbn}, Score: {score:.4f}")
        else:
            print(f"{i:2d}. ISBN: {book_isbn}, Score: {score:.4f}")
        print()
    
    # Show user's actual ratings for comparison
    user_ratings = ratings_df[ratings_df['User-ID'] == sample_user_id]
    if len(user_ratings) > 0:
        print(f"\n📖 User {sample_user_id}'s Actual Reading History:")
        print("-" * 50)
        
        for _, rating in user_ratings.head(5).iterrows():
            book_info = books_df[books_df['ISBN'] == rating['ISBN']]
            if len(book_info) > 0:
                book = book_info.iloc[0]
                print(f"• {book['Book-Title']} by {book['Book-Author']} - Rating: {rating['Book-Rating']}/10")
    
    # Test book similarity
    if len(recommendations) > 0:
        target_book = recommendations[0][0]
        print(f"\n🔍 Finding books similar to: {target_book}")
        
        similar_books = recommender.get_similar_books(
            target_book_isbn=target_book,
            candidate_books=sample_books,
            sample_users=ratings_df['User-ID'].head(10).tolist(),
            k=5
        )
        
        print(f"\n📚 Similar Books:")
        print("-" * 30)
        for i, (book_isbn, similarity) in enumerate(similar_books, 1):
            book_info = books_df[books_df['ISBN'] == book_isbn]
            if len(book_info) > 0:
                book = book_info.iloc[0]
                print(f"{i}. {book['Book-Title']} (similarity: {similarity:.3f})")

if __name__ == "__main__":
    demo_dlrm_recommendations()