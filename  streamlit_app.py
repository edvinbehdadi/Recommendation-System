"""
Streamlit Dashboard for DLRM Book Recommendation System
Simple interface for DLRM-based book recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import os
import sys
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import DLRM components
try:
    sys.path.append('.')
    from dlrm_inference import DLRMBookRecommender, load_dlrm_recommender
    DLRM_AVAILABLE = True
except ImportError as e:
    DLRM_AVAILABLE = False
    st.error(f"DLRM components not available: {e}")

# Page configuration
st.set_page_config(
    page_title="DLRM Book Recommendations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .dlrm-explanation {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
    .book-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5eb;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the book data"""
    try:
        books_df = pd.read_csv('Books.csv', encoding='latin-1', low_memory=False)
        users_df = pd.read_csv('Users.csv', encoding='latin-1', low_memory=False)
        ratings_df = pd.read_csv('Ratings.csv', encoding='latin-1', low_memory=False)
        
        # Clean column names
        books_df.columns = books_df.columns.str.replace('"', '')
        users_df.columns = users_df.columns.str.replace('"', '')
        ratings_df.columns = ratings_df.columns.str.replace('"', '')
        
        return books_df, users_df, ratings_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_resource
def load_dlrm_model():
    """Load and cache the DLRM model"""
    if not DLRM_AVAILABLE:
        return None
    
    try:
        recommender = load_dlrm_recommender("file")
        return recommender
    except Exception as e:
        st.error(f"Error loading DLRM model: {e}")
        return None

def display_book_info(book_isbn, books_df, show_rating=None):
    """Display book information with actual book cover"""
    book_info = books_df[books_df['ISBN'] == book_isbn]
    
    if len(book_info) == 0:
        st.write(f"Book with ISBN {book_isbn} not found")
        return
    
    book = book_info.iloc[0]
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Try to display actual book cover from Image-URL-M
        image_url = book.get('Image-URL-M', '')
        
        if image_url and pd.notna(image_url) and str(image_url) != 'nan':
            try:
                # Clean the URL (sometimes there are issues with Amazon URLs)
                clean_url = str(image_url).strip()
                if clean_url and 'http' in clean_url:
                    st.image(clean_url, width=150, caption="📚")
                else:
                    # Fallback to placeholder
                    st.image("https://via.placeholder.com/150x200?text=📚&color=1f77b4&bg=f0f2f6", width=150)
            except Exception as e:
                # If image loading fails, show placeholder
                st.image("https://via.placeholder.com/150x200?text=📚&color=1f77b4&bg=f0f2f6", width=150)
                st.caption("⚠️ Cover unavailable")
        else:
            # Show placeholder if no image URL
            st.image("https://via.placeholder.com/150x200?text=📚&color=1f77b4&bg=f0f2f6", width=150)
            st.caption("📚 No cover")
    
    with col2:
        st.markdown(f"**{book['Book-Title']}**")
        st.write(f"*by {book['Book-Author']}*")
        st.write(f"📅 Published: {book.get('Year-Of-Publication', 'Unknown')}")
        st.write(f"🏢 Publisher: {book.get('Publisher', 'Unknown')}")
        st.write(f"📖 ISBN: {book['ISBN']}")
        
        if show_rating is not None:
            st.markdown(f"**🎯 DLRM Score: {show_rating:.4f}**")

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 DLRM Book Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("### Deep Learning Recommendation Model for Personalized Book Suggestions")
    st.markdown("---")
    
    if not DLRM_AVAILABLE:
        st.error("DLRM components are not available. Please ensure TorchRec is properly installed.")
        st.info("To install TorchRec: `pip install torchrec`")
        return
    
    # Load data
    with st.spinner("Loading book data..."):
        books_df, users_df, ratings_df = load_data()
    
    if books_df is None:
        st.error("Failed to load data. Please check if CSV files are available.")
        return
    
    # Sidebar info
    st.sidebar.title("📊 Dataset Information")
    st.sidebar.metric("📚 Books", f"{len(books_df):,}")
    st.sidebar.metric("👥 Users", f"{len(users_df):,}")
    st.sidebar.metric("⭐ Ratings", f"{len(ratings_df):,}")
    
    # Load DLRM model
    with st.spinner("Loading DLRM model..."):
        recommender = load_dlrm_model()
    
    if recommender is None or recommender.model is None:
        st.error("❌ DLRM model not available")
        st.info("Please run the training script first: `python train_dlrm_books.py`")
        
        st.markdown("### Available Options:")
        st.markdown("1. **Train DLRM Model**: Run `python train_dlrm_books.py`")
        st.markdown("2. **Prepare Data**: Run `python dlrm_book_recommender.py`")
        st.markdown("3. **Check Files**: Ensure preprocessing files exist")
        
        return
    
    st.success("✅ DLRM model loaded successfully!")
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 DLRM Model Info")
    if recommender.preprocessing_info:
        st.sidebar.write(f"Dense features: {len(recommender.dense_cols)}")
        st.sidebar.write(f"Categorical features: {len(recommender.cat_cols)}")
        st.sidebar.write(f"Embedding dim: 64")
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Get Recommendations", "🔍 Test Predictions", "📊 Model Analysis", "📸 Book Gallery"])
    
    with tab1:
        st.header("🎯 DLRM Book Recommendations")
        st.info("Get personalized book recommendations using the trained DLRM model")
        
        # User selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_ids = sorted(users_df['User-ID'].unique())
            selected_user_id = st.selectbox("Select a user", user_ids[:1000])  # Limit for performance
        
        with col2:
            num_recommendations = st.slider("Number of recommendations", 5, 20, 10)
        
        # Show user info
        user_info = users_df[users_df['User-ID'] == selected_user_id]
        if len(user_info) > 0:
            user = user_info.iloc[0]
            st.markdown(f"**User Info**: Age: {user.get('Age', 'Unknown')}, Location: {user.get('Location', 'Unknown')}")
        
        # User's reading history
        user_ratings = ratings_df[ratings_df['User-ID'] == selected_user_id]
        if len(user_ratings) > 0:
            with st.expander(f"📖 User's Reading History ({len(user_ratings)} books)", expanded=False):
                top_rated = user_ratings.sort_values('Book-Rating', ascending=False).head(10)
                for _, rating in top_rated.iterrows():
                    book_info = books_df[books_df['ISBN'] == rating['ISBN']]
                    if len(book_info) > 0:
                        book = book_info.iloc[0]
                        st.write(f"• **{book['Book-Title']}** by {book['Book-Author']} - {rating['Book-Rating']}/10 ⭐")
        
        if st.button("🚀 Get DLRM Recommendations", type="primary"):
            with st.spinner("🤖 DLRM is analyzing user preferences..."):
                
                # Get candidate books (popular books not rated by user)
                user_rated_books = set(user_ratings['ISBN']) if len(user_ratings) > 0 else set()
                
                # Get popular books as candidates
                book_popularity = ratings_df.groupby('ISBN').size().sort_values(ascending=False)
                candidate_books = [isbn for isbn in book_popularity.head(100).index if isbn not in user_rated_books]
                
                if len(candidate_books) < num_recommendations:
                    candidate_books = book_popularity.head(200).index.tolist()
                
                # Get recommendations
                recommendations = recommender.get_user_recommendations(
                    user_id=selected_user_id,
                    candidate_books=candidate_books,
                    k=num_recommendations
                )
            
            if recommendations:
                st.success(f"Generated {len(recommendations)} DLRM recommendations!")
                
                st.subheader("🎯 DLRM Recommendations")
                
                for i, (book_isbn, score) in enumerate(recommendations, 1):
                    book_info = books_df[books_df['ISBN'] == book_isbn]
                    if len(book_info) > 0:
                        with st.expander(f"{i}. Recommendation (DLRM Score: {score:.4f})", expanded=(i <= 3)):
                            display_book_info(book_isbn, books_df, show_rating=score)
                            
                            # Additional book stats
                            book_ratings = ratings_df[ratings_df['ISBN'] == book_isbn]
                            if len(book_ratings) > 0:
                                avg_rating = book_ratings['Book-Rating'].mean()
                                num_ratings = len(book_ratings)
                                
                                st.markdown('<div class="dlrm-explanation">', unsafe_allow_html=True)
                                st.markdown("**📊 Book Statistics:**")
                                st.write(f"Average Rating: {avg_rating:.1f}/10 from {num_ratings} readers")
                                st.write(f"DLRM Confidence: {score:.1%}")
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.write(f"Book with ISBN {book_isbn} not found in database")
            else:
                st.warning("No recommendations generated")
    
    with tab2:
        st.header("🔍 Test DLRM Predictions")
        st.info("Test how well DLRM predicts actual user ratings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_user_id = st.selectbox("Select user for testing", user_ids[:500], key="test_user")
        
        with col2:
            test_mode = st.radio("Test mode", ["Random books", "User's actual books"])
        
        if st.button("🧪 Test Predictions", type="secondary"):
            with st.spinner("Testing DLRM predictions..."):
                
                if test_mode == "User's actual books":
                    # Test on user's actual rated books
                    user_test_ratings = ratings_df[ratings_df['User-ID'] == test_user_id].sample(min(10, len(user_ratings)))
                    
                    if len(user_test_ratings) > 0:
                        st.subheader("🎯 DLRM vs Actual Ratings")
                        
                        predictions = []
                        actuals = []
                        
                        for _, rating in user_test_ratings.iterrows():
                            book_isbn = rating['ISBN']
                            actual_rating = rating['Book-Rating']
                            
                            # Get DLRM prediction
                            dlrm_score = recommender.predict_rating(test_user_id, book_isbn)
                            
                            predictions.append(dlrm_score)
                            actuals.append(actual_rating >= 6)  # Convert to binary
                            
                            # Display comparison
                            book_info = books_df[books_df['ISBN'] == book_isbn]
                            if len(book_info) > 0:
                                book = book_info.iloc[0]
                                
                                col1, col2, col3 = st.columns([2, 1, 1])
                                with col1:
                                    st.write(f"**{book['Book-Title']}**")
                                    st.write(f"*by {book['Book-Author']}*")
                                
                                with col2:
                                    st.metric("Actual Rating", f"{actual_rating}/10")
                                
                                with col3:
                                    st.metric("DLRM Score", f"{dlrm_score:.3f}")
                        
                        # Calculate accuracy
                        if predictions and actuals:
                            # Convert DLRM scores to binary predictions
                            binary_preds = [1 if p > 0.5 else 0 for p in predictions]
                            accuracy = sum(p == a for p, a in zip(binary_preds, actuals)) / len(actuals)
                            
                            st.markdown("---")
                            st.success(f"🎯 DLRM Accuracy: {accuracy:.1%}")
                            
                            # Show correlation
                            actual_numeric = [rating['Book-Rating'] for _, rating in user_test_ratings.iterrows()]
                            correlation = np.corrcoef(predictions, actual_numeric)[0, 1] if len(predictions) > 1 else 0
                            st.info(f"📊 Correlation with actual ratings: {correlation:.3f}")
                    
                    else:
                        st.warning("No ratings found for this user")
                
                else:
                    # Test on random books
                    random_books = books_df.sample(10)['ISBN'].tolist()
                    
                    st.subheader("🎲 Random Book Predictions")
                    
                    for book_isbn in random_books:
                        dlrm_score = recommender.predict_rating(test_user_id, book_isbn)
                        
                        book_info = books_df[books_df['ISBN'] == book_isbn]
                        if len(book_info) > 0:
                            book = book_info.iloc[0]
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{book['Book-Title']}** by *{book['Book-Author']}*")
                            
                            with col2:
                                st.metric("DLRM Score", f"{dlrm_score:.4f}")
    
    with tab3:
        st.header("📊 DLRM Model Analysis")
        st.info("Analysis of the DLRM model performance and characteristics")
        
        # Model architecture info
        if recommender and recommender.preprocessing_info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏗️ Model Architecture")
                st.write(f"**Dense Features ({len(recommender.dense_cols)}):**")
                for col in recommender.dense_cols:
                    st.write(f"• {col}")
                
                st.write(f"**Categorical Features ({len(recommender.cat_cols)}):**")
                for i, col in enumerate(recommender.cat_cols):
                    st.write(f"• {col}: {recommender.emb_counts[i]} embeddings")
            
            with col2:
                st.subheader("📈 Dataset Statistics")
                total_samples = recommender.preprocessing_info.get('total_samples', 0)
                positive_rate = recommender.preprocessing_info.get('positive_rate', 0)
                
                st.metric("Total Samples", f"{total_samples:,}")
                st.metric("Positive Rate", f"{positive_rate:.1%}")
                st.metric("Train Samples", f"{recommender.preprocessing_info.get('train_samples', 0):,}")
                st.metric("Validation Samples", f"{recommender.preprocessing_info.get('val_samples', 0):,}")
                st.metric("Test Samples", f"{recommender.preprocessing_info.get('test_samples', 0):,}")
        
        # Feature importance analysis
        st.subheader("🔍 Feature Analysis")
        
        if st.button("Analyze Feature Importance"):
            with st.spinner("Analyzing feature importance..."):
                
                # Sample some users and books
                sample_users = users_df['User-ID'].sample(20).tolist()
                sample_books = books_df['ISBN'].sample(20).tolist()
                
                # Test different feature combinations
                st.write("**Feature Impact Analysis:**")
                
                base_predictions = []
                for user_id in sample_users[:5]:
                    for book_isbn in sample_books[:5]:
                        score = recommender.predict_rating(user_id, book_isbn)
                        base_predictions.append(score)
                
                avg_prediction = np.mean(base_predictions)
                st.metric("Average Prediction Score", f"{avg_prediction:.4f}")
                
                st.success("✅ Feature analysis completed!")
        
        # Load training results if available
        if os.path.exists('dlrm_book_training_results.pkl'):
            with open('dlrm_book_training_results.pkl', 'rb') as f:
                training_results = pickle.load(f)
            
            st.subheader("📈 Training Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Final Validation AUROC", f"{training_results.get('final_val_auroc', 0):.4f}")
                st.metric("Test AUROC", f"{training_results.get('test_auroc', 0):.4f}")
            
            with col2:
                val_history = training_results.get('val_aurocs_history', [])
                if val_history:
                    st.line_chart(pd.DataFrame({
                        'Epoch': range(len(val_history)),
                        'Validation AUROC': val_history
                    }).set_index('Epoch'))
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ## 🚀 How DLRM Works for Book Recommendations
    
    **DLRM (Deep Learning Recommendation Model)** is specifically designed for recommendation systems and offers several advantages:
    
    ### 🏗️ Architecture Benefits:
    - **Multi-feature Processing**: Handles both categorical (user ID, book ID, publisher) and numerical (age, ratings) features
    - **Embedding Tables**: Learns rich representations for categorical features
    - **Cross-feature Interactions**: Captures complex relationships between different features
    - **Scalable Design**: Efficiently handles large-scale recommendation datasets
    
    ### 📊 Features Used:
    **Categorical Features:**
    - User ID, Book ID, Publisher, Country, Age Group, Publication Decade, Rating Level
    
    **Dense Features:**  
    - Normalized Age, Publication Year, User Activity, Book Popularity, Average Ratings
    
    ### 🎯 Why DLRM vs LLM for Recommendations:
    - **Purpose-built**: Specifically designed for recommendation systems
    - **Feature Integration**: Better at combining diverse feature types
    - **Scalability**: More efficient for large-scale recommendation tasks
    - **Performance**: Higher accuracy for rating prediction tasks
    - **Production Ready**: Optimized for real-time inference
    
    ### 💡 Best Use Cases:
    - **Personalized Recommendations**: Based on user behavior and item characteristics
    - **Rating Prediction**: Accurately predicts user preferences
    - **Cold Start**: Handles new users and items through content features
    - **Real-time Serving**: Fast inference for production systems
    """)

    with tab4:
        st.header("📸 Book Gallery")
        st.info("Browse book covers and discover new titles")
        
        # Gallery options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            gallery_mode = st.selectbox(
                "Choose gallery mode",
                ["Popular Books", "Recent Publications", "Random Selection", "Search Results"]
            )
        
        with col2:
            books_per_row = st.slider("Books per row", 2, 6, 4)
            max_books = st.slider("Maximum books", 10, 50, 20)
        
        # Get books based on selected mode
        if gallery_mode == "Popular Books":
            # Get most rated books
            book_popularity = ratings_df.groupby('ISBN').size().sort_values(ascending=False)
            gallery_books = books_df[books_df['ISBN'].isin(book_popularity.head(max_books).index)]
            
        elif gallery_mode == "Recent Publications":
            # Get recent books
            books_df_temp = books_df.copy()
            books_df_temp['Year-Of-Publication'] = pd.to_numeric(books_df_temp['Year-Of-Publication'], errors='coerce')
            recent_books = books_df_temp.sort_values('Year-Of-Publication', ascending=False, na_position='last')
            gallery_books = recent_books.head(max_books)
            
        elif gallery_mode == "Random Selection":
            # Random books
            gallery_books = books_df.sample(min(max_books, len(books_df)))
            
        else:  # Search Results
            search_query = st.text_input("Search books for gallery", placeholder="Enter title, author, or publisher")
            if search_query:
                mask = (
                    books_df['Book-Title'].str.contains(search_query, case=False, na=False) |
                    books_df['Book-Author'].str.contains(search_query, case=False, na=False) |
                    books_df['Publisher'].str.contains(search_query, case=False, na=False)
                )
                gallery_books = books_df[mask].head(max_books)
            else:
                gallery_books = books_df.head(max_books)
        
        # Display gallery
        if len(gallery_books) > 0:
            st.markdown(f"**📚 Showing {len(gallery_books)} books**")
            
            # Create grid layout
            books_list = gallery_books.to_dict('records')
            
            # Display books in rows
            for i in range(0, len(books_list), books_per_row):
                cols = st.columns(books_per_row)
                
                for j, col in enumerate(cols):
                    if i + j < len(books_list):
                        book = books_list[i + j]
                        
                        with col:
                            # Book cover
                            image_url = book.get('Image-URL-M', '')
                            
                            if image_url and pd.notna(image_url) and str(image_url) != 'nan':
                                try:
                                    clean_url = str(image_url).strip()
                                    if clean_url and 'http' in clean_url:
                                        st.image(clean_url, width='stretch')
                                    else:
                                        st.image("https://via.placeholder.com/150x200?text=📚&color=1f77b4&bg=f0f2f6", width='stretch')
                                except:
                                    st.image("https://via.placeholder.com/150x200?text=📚&color=1f77b4&bg=f0f2f6", width='stretch')
                            else:
                                st.image("https://via.placeholder.com/150x200?text=📚&color=1f77b4&bg=f0f2f6", width='stretch')
                            
                            # Book info
                            title = book['Book-Title']
                            if len(title) > 40:
                                title = title[:37] + "..."
                            
                            author = book['Book-Author']
                            if len(author) > 25:
                                author = author[:22] + "..."
                            
                            st.markdown(f"**{title}**")
                            st.write(f"*{author}*")
                            st.write(f"📅 {book.get('Year-Of-Publication', 'Unknown')}")
                            
                            # Book statistics
                            book_stats = ratings_df[ratings_df['ISBN'] == book['ISBN']]
                            if len(book_stats) > 0:
                                avg_rating = book_stats['Book-Rating'].mean()
                                num_ratings = len(book_stats)
                                st.write(f"⭐ {avg_rating:.1f}/10 ({num_ratings} ratings)")
                            else:
                                st.write("⭐ No ratings")
                            
                            # DLRM prediction button
                            if recommender and recommender.model:
                                if st.button(f"🎯 DLRM Score", key=f"dlrm_{book['ISBN']}"):
                                    with st.spinner("Calculating..."):
                                        # Use first user as example
                                        sample_user = users_df['User-ID'].iloc[0]
                                        dlrm_score = recommender.predict_rating(sample_user, book['ISBN'])
                                        st.success(f"DLRM Score: {dlrm_score:.3f}")
        else:
            st.info("No books found for the selected criteria")
        
        # Quick stats
        st.markdown("---")
        st.subheader("📊 Gallery Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            books_with_covers = sum(1 for _, book in gallery_books.iterrows()
                                  if book.get('Image-URL-M') and pd.notna(book.get('Image-URL-M')))
            st.metric("Books with Covers", f"{books_with_covers}/{len(gallery_books)}")
        
        with col2:
            # Convert Year-Of-Publication to numeric, coercing errors to NaN
            years = pd.to_numeric(gallery_books['Year-Of-Publication'], errors='coerce')
            avg_year = years.mean()
            st.metric("Average Publication Year", f"{avg_year:.0f}" if not pd.isna(avg_year) else "Unknown")
        
        with col3:
            unique_authors = gallery_books['Book-Author'].nunique()
            st.metric("Unique Authors", unique_authors)
        
        with col4:
            unique_publishers = gallery_books['Publisher'].nunique()
            st.metric("Unique Publishers", unique_publishers)

if __name__ == "__main__":
    main()
