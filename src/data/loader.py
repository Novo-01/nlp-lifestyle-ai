import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
    """
    データを読み込む関数
    
    Args:
        file_path (str): データファイルのパス
        
    Returns:
        pd.DataFrame: 読み込んだデータフレーム
    """
    return pd.read_csv(file_path)

def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    データを訓練データとテストデータに分割する
    
    Args:
        df (pd.DataFrame): 入力データフレーム
        target_column (str): ターゲット変数の列名
        test_size (float): テストデータの割合
        random_state (int): 乱数シード
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            訓練データの特徴量、テストデータの特徴量、
            訓練データのターゲット、テストデータのターゲット
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test 