import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def collect_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv('stock_data.csv')
    return data

def preprocess_data(data):
    data['Date'] = pd.to_datetime(data.index)
    data.set_index('Date', inplace=True)
    
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data = data.dropna()
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    preprocessed_data = pd.DataFrame(data_scaled, columns=data.columns)
    preprocessed_data.to_csv('preprocessed_data.csv', index=False)
    return preprocessed_data

if __name__ == "__main__":
    ticker = '^GSPC'
    start_date = '2010-01-01'
    end_date = '2023-01-01'
    
    data = collect_data(ticker, start_date, end_date)
    preprocess_data(data)
