from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Any, Dict, Union

class BaseModel(ABC):
    """
    モデルの基本クラス
    """
    def __init__(self, **kwargs):
        self.model = None
        self.params = kwargs
        
    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """
        モデルの学習を行う
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): 特徴量
            y (Union[pd.Series, np.ndarray]): ターゲット変数
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        予測を行う
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): 特徴量
            
        Returns:
            np.ndarray: 予測値
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        モデルのパラメータを取得する
        
        Returns:
            Dict[str, Any]: パラメータの辞書
        """
        return self.params
    
    def set_params(self, **params) -> None:
        """
        モデルのパラメータを設定する
        
        Args:
            **params: パラメータのキーワード引数
        """
        self.params.update(params) 