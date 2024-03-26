from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from typing import Tuple

def fill_dataframe(df: pd.DataFrame, target_col: str, scale: bool=True) -> Tuple[pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    df[target_col] = df[target_col].ffill().bfill()
    if scale:
        df[target_col] = scaler.fit_transform(df[[target_col]])
    return df, scaler

def fill_missing_date(df: pd.DataFrame, date_range: pd.DataFrame, merge_col: str = 'date'):
    return df.merge(date_range, on=merge_col, how='left')



