import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any

class DataPreprocessor:
    """
    データ前処理を行う基本クラス
    """
    def __init__(self):
        self.feature_columns = None
        self.target_column = None
        
    def fit(self, df: pd.DataFrame, feature_columns: List[str], target_column: str) -> None:
        """
        前処理の設定を行う
        
        Args:
            df (pd.DataFrame): 入力データフレーム
            feature_columns (List[str]): 特徴量の列名リスト
            target_column (str): ターゲット変数の列名
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データの前処理を実行する
        
        Args:
            df (pd.DataFrame): 入力データフレーム
            
        Returns:
            pd.DataFrame: 前処理済みのデータフレーム
        """
        if self.feature_columns is None or self.target_column is None:
            raise ValueError("fitメソッドを先に実行してください")
            
        # 基本的な前処理
        df = df.copy()
        
        # 欠損値の処理
        df = self._handle_missing_values(df)
        
        # カテゴリカル変数の処理
        df = self._handle_categorical_variables(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        欠損値の処理を行う
        
        Args:
            df (pd.DataFrame): 入力データフレーム
            
        Returns:
            pd.DataFrame: 欠損値処理済みのデータフレーム
        """
        # 数値型の列の欠損値を中央値で補完
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # カテゴリカル変数の欠損値を最頻値で補完
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        
        return df
    
    def _handle_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        カテゴリカル変数の処理を行う
        
        Args:
            df (pd.DataFrame): 入力データフレーム
            
        Returns:
            pd.DataFrame: カテゴリカル変数処理済みのデータフレーム
        """
        # カテゴリカル変数をダミー変数に変換
        categorical_cols = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        return df 