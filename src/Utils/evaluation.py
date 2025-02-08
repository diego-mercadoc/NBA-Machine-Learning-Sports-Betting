from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional

def evaluate_binary_classification(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate binary classification model with comprehensive metrics.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_pred_proba: Predicted probabilities (shape: n_samples, 2)
        
    Returns:
        Dictionary containing various evaluation metrics
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    # Calculate additional betting-specific metrics
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    
    # Calculate metrics per class
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'class_metrics': {
            'precision': class_precision.tolist(),
            'recall': class_recall.tolist(),
            'f1': class_f1.tolist()
        },
        'support': {
            'total_samples': total_predictions,
            'correct_predictions': correct_predictions
        }
    }

def evaluate_multiclass_classification(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate multiclass classification model with comprehensive metrics.
    
    Args:
        y_true: True labels (0, 1, or 2 for Under/Over/Push)
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (shape: n_samples, n_classes)
        
    Returns:
        Dictionary containing various evaluation metrics
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate additional betting-specific metrics
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    
    # Calculate metrics per class
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'class_metrics': {
            'precision': class_precision.tolist(),
            'recall': class_recall.tolist(),
            'f1': class_f1.tolist()
        },
        'support': {
            'total_samples': total_predictions,
            'correct_predictions': correct_predictions
        }
    }

def print_evaluation_report(metrics: Dict[str, Any], model_type: str = "binary"):
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics
        model_type: Type of model ("binary" or "multiclass")
    """
    print("\n" + "="*50)
    print(f"Model Evaluation Report ({model_type.title()} Classification)")
    print("="*50)
    
    print(f"\nOverall Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted Precision: {metrics['precision']:.4f}")
    print(f"Weighted Recall: {metrics['recall']:.4f}")
    print(f"Weighted F1: {metrics['f1']:.4f}")
    if 'auc' in metrics:
        print(f"ROC AUC: {metrics['auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    print(f"\nPer-Class Metrics:")
    classes = ['Away Win', 'Home Win'] if model_type == "binary" else ['Under', 'Over', 'Push']
    for i, class_name in enumerate(classes):
        print(f"\n{class_name}:")
        print(f"Precision: {metrics['class_metrics']['precision'][i]:.4f}")
        print(f"Recall: {metrics['class_metrics']['recall'][i]:.4f}")
        print(f"F1: {metrics['class_metrics']['f1'][i]:.4f}")
    
    print(f"\nSupport:")
    print(f"Total Samples: {metrics['support']['total_samples']}")
    print(f"Correct Predictions: {metrics['support']['correct_predictions']}")
    print("="*50 + "\n") 