"""
Model training and evaluation utilities for tennis upset prediction.

This module contains functions for training various ML models and
performing hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class TennisUpsetNN(nn.Module):
    """Neural network for tennis upset prediction."""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 100) -> Tuple:
    """
    Prepare data for training and testing.
    
    Args:
        df: DataFrame with features and target
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from features import get_feature_columns
    
    features = get_feature_columns()
    target = 'upset'
    
    X = df[features]
    y = df[target].astype(int)
    
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, 
                            X_test: pd.DataFrame, y_test: pd.Series,
                            use_engineered_features: bool = True) -> Dict[str, Any]:
    """
    Train logistic regression model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        use_engineered_features: Whether to use all features or just basic ones
        
    Returns:
        Dictionary with model and results
    """
    if not use_engineered_features:
        # Use only basic features
        basic_features = ['fav_age', 'under_age', 'tourney_level_enc',
                         'fav_rank', 'under_rank', 'round_enc',
                         'surface_Clay', 'surface_Grass', 'surface_Hard', 'surface_Carpet']
        X_train = X_train[basic_features]
        X_test = X_test[basic_features]
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear'))
    ])
    
    # Hyperparameter grid
    param_grid = {
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [0.01, 0.1, 1, 10]
    }
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Predictions
    y_pred = grid_search.predict(X_test)
    y_proba = grid_search.predict_proba(X_test)[:, 1]
    
    return {
        'model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Train Random Forest model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Dictionary with model and results
    """
    # Create pipeline
    pipeline = Pipeline([
        ('clf', RandomForestClassifier(random_state=100, class_weight='balanced'))
    ])
    
    # Hyperparameter grid
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__max_features': ['sqrt', 'log2']
    }
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Predictions
    y_pred = grid_search.predict(X_test)
    y_proba = grid_search.predict_proba(X_test)[:, 1]
    
    return {
        'model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'feature_importance': grid_search.best_estimator_.named_steps['clf'].feature_importances_
    }


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Train XGBoost model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Dictionary with model and results
    """
    # Calculate class weights
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    # Create pipeline
    pipeline = Pipeline([
        ('clf', XGBClassifier(
            eval_metric='logloss',
            random_state=100,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False
        ))
    ])
    
    # Hyperparameter grid
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 6],
        'clf__learning_rate': [0.01, 0.1],
        'clf__subsample': [0.8, 1.0]
    }
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Predictions
    y_pred = grid_search.predict(X_test)
    y_proba = grid_search.predict_proba(X_test)[:, 1]
    
    return {
        'model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'feature_importance': grid_search.best_estimator_.named_steps['clf'].feature_importances_
    }


def train_neural_network(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
    """
    Train neural network model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary with model and results
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    # DataLoader
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Model
    model = TennisUpsetNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluation
    with torch.no_grad():
        preds = model(X_test_t)
        y_pred = (preds > 0.5).int().numpy().flatten()
        y_proba = preds.numpy().flatten()
    
    return {
        'model': model,
        'scaler': scaler,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }


def train_all_models(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Train all models and return results.
    
    Args:
        df: DataFrame with features and target
        
    Returns:
        Dictionary with results for all models
    """
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    results = {}
    
    print("Training Logistic Regression (raw features)...")
    results['logistic_raw'] = train_logistic_regression(
        X_train, y_train, X_test, y_test, use_engineered_features=False
    )
    
    print("Training Logistic Regression (engineered features)...")
    results['logistic_engineered'] = train_logistic_regression(
        X_train, y_train, X_test, y_test, use_engineered_features=True
    )
    
    print("Training Random Forest...")
    results['random_forest'] = train_random_forest(X_train, y_train, X_test, y_test)
    
    print("Training XGBoost...")
    results['xgboost'] = train_xgboost(X_train, y_train, X_test, y_test)
    
    print("Training Neural Network...")
    results['neural_network'] = train_neural_network(X_train, y_train, X_test, y_test)
    
    return results


if __name__ == "__main__":
    # Example usage
    from data_loader import load_tennis_data, restructure_data
    from features import prepare_features
    
    print("Loading and processing data...")
    data = load_tennis_data(range(2020, 2022))  # Small sample for testing
    restructured = restructure_data(data)
    features = prepare_features(restructured)
    
    print("Training models...")
    results = train_all_models(features)
    
    print("Model training complete!")
    for model_name, result in results.items():
        print(f"{model_name}: Accuracy = {result['accuracy']:.3f}, AUC = {result['auc']:.3f}")
