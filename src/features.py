import pandas as pd
import numpy as np

# Tính log returns để mô hình học được sự biến động của giá
def create_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    return df

# Tạo lag feature cho log returns để mô hình học momentum
def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['returns_lag_1'] = df['log_returns'].shift(1)
    df['returns_lag_3'] = df['log_returns'].shift(3)
    df['returns_lag_7'] = df['log_returns'].shift(7)
    return df

# Tạo các feature rolling mean và rolling std để mô hình học được xu hướng và sự biến động trong ngắn hạn
def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['rolling_mean_7'] = df['log_returns'].rolling(window=7).mean()
    df['rolling_std_7'] = df['log_returns'].rolling(window=7).std()
    
    df['rolling_mean_14'] = df['log_returns'].rolling(window=14).mean()
    df['rolling_std_14'] = df['log_returns'].rolling(window=14).std()
    return df

# Tạo các feature momentum dựa trên giá đóng cửa
def create_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df= df.copy()
    df['momentum_3'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_7'] = df['close'] / df['close'].shift(7) - 1
    df['momentum_14'] = df['close'] / df['close'].shift(14) - 1
    return df

# Tạo target biến đổi thành bài toán phân loại: 1 nếu giá tăng vào ngày tiếp theo, 0 nếu giảm hoặc không đổi
def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['target'] = (df['log_returns'].shift(-1) > 0).astype(int)
    return df

# Hàm tổng hợp để tạo tất cả các feature
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = create_returns(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_momentum_features(df)
    df = create_targets(df)
    
    # Xóa các hàng có giá trị NaN do việc tạo feature
    df = df.dropna()
    return df
