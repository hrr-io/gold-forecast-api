import numpy as np
import pandas as pd

def merge_df(DFs, TARGET="XAUUSD", COLUMNS=[]):
    df = pd.concat(DFs, axis=1).sort_index()
    for col in df.columns:
        if col != TARGET:
            df[col] = df[col].ffill()
    df = df.dropna(subset=[TARGET]).dropna()
    # full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    # df = df.reindex(full_index).ffill().dropna()
    df = df.reindex(columns=COLUMNS)
    df.index.name = 'DATE'
    return df

def log_return_difference(DF, COLUMNS, METHOD='log_return'):
    df = DF.copy()
    for col, new_col in COLUMNS.items():
        if METHOD == 'log_return':
            df[new_col] = np.log(df[col]).diff()
        else:
            df[new_col] = df[col].diff()
    return df

def winsorize(SERIES, P=0.01):
    lo, hi = SERIES.quantile([P, 1 - P])
    return SERIES.clip(lo, hi)

def rolling_winsorize(SERIES, WINDOW=252, P=0.01):
    lo = SERIES.rolling(WINDOW, min_periods=WINDOW).quantile(P)
    hi = SERIES.rolling(WINDOW, min_periods=WINDOW).quantile(1 - P)
    return SERIES.clip(lo, hi)

def rolling_zscore(SERIES, WINDOW=252):
    mean = SERIES.rolling(WINDOW, min_periods=WINDOW).mean()
    std  = SERIES.rolling(WINDOW, min_periods=WINDOW).std()
    return (SERIES - mean) / std

def ewma_zscore(SERIES, SPAN=126, EPS=1e-8):
    mean = SERIES.ewm(span=SPAN, adjust=False).mean()
    var  = SERIES.ewm(span=SPAN, adjust=False).var()
    std  = np.sqrt(var).clip(lower=EPS)
    return (SERIES - mean) / std

def robust_zscore(SERIES, WINDOW=126, EPS=1e-8):
    median = SERIES.rolling(WINDOW, min_periods=WINDOW//2).median()
    mad = (
        SERIES.sub(median)
              .abs()
              .rolling(WINDOW, min_periods=WINDOW//2)
              .median()
    )
    mad = (1.4826 * mad).clip(lower=EPS)
    return (SERIES - median) / mad

def ewma_robust_zscore(SERIES, SPAN=126, EPS=1e-8):
    mean = SERIES.ewm(span=SPAN, adjust=False).mean()
    mad = (
        SERIES.sub(mean)
              .abs()
              .ewm(span=SPAN, adjust=False)
              .mean()
    )
    mad = (1.4826 * mad).clip(lower=EPS)
    return (SERIES - mean) / mad

def add_lag_features(DF, COLS, LAGS):
    df = DF.copy()
    for col in COLS:
        for lag in LAGS:
            df[f'{col}_LAG_{lag}'] = df[col].shift(lag)
    return df

def add_rolling_stats(DF, COLS, WINDOWS):
    df = DF.copy()
    for col in COLS:
        for window in WINDOWS:
            df[f'{col}_ROLL_MEAN_{window}'] = (df[col].rolling(window).mean())
            df[f'{col}_ROLL_STD_{window}'] = (df[col].rolling(window).std())
    return df

def shift_target_col(DF, COL, HORIZONS):
    df = DF.copy()
    for h in HORIZONS:
        df[f'TRGT_{h}D'] = (df[COL].rolling(h).sum().shift(-h))
    return df