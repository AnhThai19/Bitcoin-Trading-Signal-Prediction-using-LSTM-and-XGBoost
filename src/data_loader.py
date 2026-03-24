import yfinance as yf
import pandas as pd
import os

def download_btc_data(
    ticker: str = "BTC-USD",
    start_date: str = "2017-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker} between {start_date} and {end_date}.")
    
    # Sử dụng giá trị index ở level 0 (Close, Open, High, Low, Volume) làm tên cột
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Chuyển tên cột thành chữ thường 
    df.columns = [col.lower() for col in df.columns]
    
    # Đảm bảo index là datetime
    df.index = pd.to_datetime(df.index)
    
    # Sắp xếp index theo thứ tự tăng dần
    df = df.sort_index()
    
    # Xóa các missing values
    df = df.dropna()
    
    return df

def save_to_csv(df: pd.DataFrame, path: str ="../data/btc_data.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    
def load_from_csv(path: str = "../data/btc_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df
