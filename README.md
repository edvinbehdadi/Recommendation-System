# DLRM Book Recommendation System

A Deep Learning Recommendation Model (DLRM) for book recommendations, implementing a dual-tower architecture to provide personalized book recommendations based on user preferences and book characteristics.

## Project Overview

This project implements a book recommendation system using Meta's Deep Learning Recommendation Model (DLRM) architecture. The system processes user-book interaction data to predict whether a user will like a book (rating ≥ 6.0) based on various user and book features.

### Key Features

- **Dual-Tower Architecture**: Separate processing paths for user and item features
- **Categorical & Dense Features**: Processes both categorical (e.g., user ID, book ID) and dense features (e.g., user age, book popularity)
- **Embedding Layers**: Learns representations of categorical features
- **Binary Classification**: Predicts whether a user will like a book (rating ≥ 6.0)
- **Production-Ready**: Includes data preprocessing, model training, and inference components

## Loss Function

The model uses **Binary Cross Entropy (BCE) Loss** for training, which is appropriate for the binary classification task of predicting whether a user will like a book. BCE is defined as:

```
BCE(y, p) = -[y * log(p) + (1 - y) * log(1 - p)]
```

Where:
- `y` is the true label (1 for ratings ≥ 6.0, 0 otherwise)
- `p` is the predicted probability

BCE loss is particularly well-suited for this recommendation task because:

1. **Binary Target**: Our task is formulated as binary classification (like/dislike)
2. **Probability Output**: The model outputs a probability that can be directly used for ranking recommendations
3. **Imbalanced Data Handling**: BCE works well with imbalanced datasets, which is common in recommendation systems
4. **Gradient Properties**: Provides stable gradients for efficient training

The loss is implemented in the DLRMTrain wrapper using PyTorch's BCEWithLogitsLoss, which combines a sigmoid activation with the BCE loss for numerical stability.

## Deployment and Running Instructions

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- TorchRec
- Streaming Dataset library
- MLflow (for experiment tracking)
- Required datasets: Books.csv, Users.csv, Ratings.csv

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your environment:
   ```bash
   # Set environment variables for distributed training (if needed)
   export RANK=0
   export LOCAL_RANK=0
   export WORLD_SIZE=1
   export MASTER_ADDR=localhost
   export MASTER_PORT=29500
   ```

### Data Preparation

1. Place the dataset files in the project root:
   - Books.csv
   - Users.csv
   - Ratings.csv

2. Run the data preprocessing script:
   ```bash
   python dlrm_book_recommender.py
   ```

   This will:
   - Clean and filter the data
   - Create features for DLRM
   - Split data into train/validation/test sets
   - Save data in MDS format for training
   - Save preprocessing information for inference

### Model Training

1. Run the training script:
   ```bash
   python -m notebooks.02_feature_engineering_and_model_training
   ```

   Or open and run the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/02_feature_engineering_and_model_training.ipynb
   ```

2. Monitor training progress:
   - The script will log metrics to MLflow
   - Model checkpoints will be saved after each epoch
   - The final model will be saved as `dlrm_book_model_final.pth`

### Inference and Recommendations

1. To generate recommendations, use the inference script:
   ```bash
   python dlrm_inference.py
   ```

   This will:
   - Load the trained model
   - Generate recommendations for sample users
   - Display the top recommended books

2. For custom recommendations, you can use the DLRMBookRecommender class:
   ```python
   from dlrm_inference import DLRMBookRecommender
   
   # Initialize recommender with trained model
   recommender = DLRMBookRecommender(model_path="dlrm_book_model_final.pth")
   
   # Get recommendations for a user
   recommendations = recommender.get_user_recommendations(
       user_id=123,
       candidate_books=candidate_books,
       k=10
   )
   ```

## Model Efficiency Metrics

We track several metrics to measure the efficiency and effectiveness of our recommendation model:

### Accuracy Metrics

1. **AUROC (Area Under the Receiver Operating Characteristic curve)**
   - Primary evaluation metric for binary classification performance
   - Measures the model's ability to distinguish between positive and negative examples
   - Range: 0.5 (random) to 1.0 (perfect)
   - Why: AUROC is threshold-independent and works well for imbalanced datasets, which is common in recommendation systems

2. **Binary Cross Entropy Loss**
   - Measures the difference between predicted probabilities and actual labels
   - Used to monitor training convergence
   - Why: Directly optimizes the model's ability to predict user preferences

### Efficiency Metrics

3. **Inference Time**
   - Measures the time taken to generate recommendations
   - Critical for real-time recommendation systems
   - Why: Ensures the model can serve recommendations with low latency

4. **Memory Usage**
   - Tracks the memory footprint of the model during training and inference
   - Why: Important for deployment in resource-constrained environments

5. **Throughput**
   - Measures the number of recommendations that can be generated per second
   - Why: Ensures the system can scale to handle many users

### Business Metrics

6. **Coverage**
   - Percentage of items that the system is able to recommend
   - Why: Ensures diverse recommendations and prevents popularity bias

7. **Diversity**
   - Variety of items recommended to users
   - Why: Prevents recommendation fatigue and improves user experience

8. **Novelty**
   - Ability to recommend new or less popular items
   - Why: Helps users discover new content and improves engagement

These metrics were chosen to provide a comprehensive view of the model's performance, balancing accuracy, efficiency, and business value. By tracking these metrics, we can identify areas for improvement and ensure the model meets both technical and business requirements.

## 📱 Streamlit Dashboard

The system includes an interactive web dashboard for easy model testing and analysis:

```bash
streamlit run streamlit_dlrm_app.py
```

**Dashboard Features:**
- **🎯 Personalized Recommendations**: Get book recommendations for any user with real-time scoring
- **🧪 Model Testing**: Test prediction accuracy against actual user ratings with visual comparison
- **📊 Performance Analysis**: Monitor model metrics, feature distributions, and recommendation quality
- **🔍 Interactive Exploration**: Search users, filter books, and analyze prediction patterns
- **📈 Real-time Visualization**: Charts and graphs showing model performance and data insights

The dashboard provides a user-friendly interface to interact with the DLRM model without coding, making it easy to evaluate recommendations and understand model behavior.

## Project Structure

- `dlrm_book_recommender.py`: Data preprocessing and feature engineering
- `dlrm_inference.py`: Model inference and recommendation generation
- `streamlit_dlrm_app.py`: Interactive web dashboard for model testing and analysis
- `train_dlrm_books.py`: Model training script with DLRM implementation
- `notebooks/01_data_exploration_and_preprocessing.ipynb`: Data exploration and analysis
- `notebooks/02_feature_engineering_and_model_training.ipynb`: Model training and evaluation
- `book_dlrm_preprocessing.pkl`: Saved preprocessing information
- `dlrm_book_model_final.pth`: Trained model weights
- `requirements.txt`: Project dependencies

## License

[MIT License](LICENSE)