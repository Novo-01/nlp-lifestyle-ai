import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Union

def plot_feature_importance(
    feature_importance: Dict[str, float],
    top_n: int = 10
) -> None:
    """
    特徴量の重要度をプロットする
    
    Args:
        feature_importance (Dict[str, float]): 特徴量の重要度の辞書
        top_n (int): 表示する上位の特徴量の数
    """
    # 重要度でソート
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    features, importance = zip(*sorted_features)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(importance), y=list(features))
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None
) -> None:
    """
    混同行列をプロットする
    
    Args:
        y_true (np.ndarray): 真の値
        y_pred (np.ndarray): 予測値
        labels (List[str], optional): クラスラベル
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray
) -> None:
    """
    学習曲線をプロットする
    
    Args:
        train_sizes (np.ndarray): 訓練データサイズ
        train_scores (np.ndarray): 訓練スコア
        val_scores (np.ndarray): 検証スコア
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
    plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
    plt.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.1
    )
    plt.fill_between(
        train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.1
    )
    plt.title('Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 