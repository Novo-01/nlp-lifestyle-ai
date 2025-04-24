import numpy as np
from typing import Union, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    分類問題の評価指標を計算する
    
    Args:
        y_true (np.ndarray): 真の値
        y_pred (np.ndarray): 予測値
        
    Returns:
        Dict[str, float]: 評価指標の辞書
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    return metrics

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    回帰問題の評価指標を計算する
    
    Args:
        y_true (np.ndarray): 真の値
        y_pred (np.ndarray): 予測値
        
    Returns:
        Dict[str, float]: 評価指標の辞書
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    return metrics 