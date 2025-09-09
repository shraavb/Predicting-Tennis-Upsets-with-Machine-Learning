"""
Model evaluation utilities for tennis upset prediction.

This module contains functions for evaluating model performance,
creating visualizations, and generating reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = "Model") -> None:
    """
    Plot confusion matrix for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for the title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Upset', 'Upset'],
                yticklabels=['No Upset', 'Upset'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, 
                   model_name: str = "Model") -> None:
    """
    Plot ROC curve for model evaluation.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model for the title
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray,
                               model_name: str = "Model") -> None:
    """
    Plot precision-recall curve for model evaluation.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model for the title
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {ap:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names: List[str], importances: np.ndarray,
                           model_name: str = "Model", top_n: int = 10) -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        model_name: Name of the model for the title
        top_n: Number of top features to show
    """
    # Get top N features
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_importances)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances - {model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def compare_models(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare performance of different models.
    
    Args:
        results: Dictionary with model results
        
    Returns:
        DataFrame with model comparison
    """
    comparison_data = []
    
    for model_name, result in results.items():
        # Extract metrics from classification report
        report = result['classification_report']
        lines = report.split('\n')
        
        # Parse precision, recall, f1-score for upset class (class 1)
        upset_metrics = {}
        for line in lines:
            if '1' in line and 'avg' not in line and 'support' not in line:
                parts = line.split()
                if len(parts) >= 4:
                    upset_metrics['precision'] = float(parts[1])
                    upset_metrics['recall'] = float(parts[2])
                    upset_metrics['f1_score'] = float(parts[3])
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'AUC': result['auc'],
            'Precision (Upset)': upset_metrics.get('precision', 0),
            'Recall (Upset)': upset_metrics.get('recall', 0),
            'F1-Score (Upset)': upset_metrics.get('f1_score', 0)
        })
    
    return pd.DataFrame(comparison_data)


def plot_model_comparison(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Create visualizations comparing all models.
    
    Args:
        results: Dictionary with model results
    """
    comparison_df = compare_models(results)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    axes[0, 0].bar(comparison_df['Model'], comparison_df['Accuracy'])
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # AUC comparison
    axes[0, 1].bar(comparison_df['Model'], comparison_df['AUC'])
    axes[0, 1].set_title('Model AUC Comparison')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Precision comparison
    axes[1, 0].bar(comparison_df['Model'], comparison_df['Precision (Upset)'])
    axes[1, 0].set_title('Model Precision (Upset) Comparison')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Recall comparison
    axes[1, 1].bar(comparison_df['Model'], comparison_df['Recall (Upset)'])
    axes[1, 1].set_title('Model Recall (Upset) Comparison')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison table
    print("Model Performance Comparison:")
    print(comparison_df.round(3))


def generate_evaluation_report(results: Dict[str, Dict[str, Any]], 
                             output_file: str = "evaluation_report.txt") -> None:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        results: Dictionary with model results
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        f.write("TENNIS UPSET PREDICTION - MODEL EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Model comparison
        comparison_df = compare_models(results)
        f.write("MODEL PERFORMANCE COMPARISON:\n")
        f.write("-" * 30 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # Individual model details
        for model_name, result in results.items():
            f.write(f"DETAILED RESULTS - {model_name.upper()}:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best Parameters: {result.get('best_params', 'N/A')}\n")
            f.write(f"Accuracy: {result['accuracy']:.3f}\n")
            f.write(f"AUC: {result['auc']:.3f}\n")
            f.write("\nClassification Report:\n")
            f.write(result['classification_report'])
            f.write("\n" + "=" * 50 + "\n\n")
    
    print(f"Evaluation report saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    from model import train_all_models
    from data_loader import load_tennis_data, restructure_data
    from features import prepare_features
    
    print("Loading and processing data...")
    data = load_tennis_data(range(2020, 2022))  # Small sample for testing
    restructured = restructure_data(data)
    features = prepare_features(restructured)
    
    print("Training models...")
    results = train_all_models(features)
    
    print("Generating evaluation report...")
    plot_model_comparison(results)
    generate_evaluation_report(results)
