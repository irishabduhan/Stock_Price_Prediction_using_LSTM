import pandas as pd
import numpy as np
import talib

def preprocess_stock_data(
    file_path: str,
    window_size: int = 10,
    normalize: bool = False,
    dropna: bool = True
):
    """
    Preprocess stock data for time series modeling (e.g., LSTM).
    
    Parameters:
    - file_path (str): Path to CSV file containing stock data with 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'.
    - window_size (int): Sequence window size for model input.
    - normalize (bool): If True, applies StandardScaler normalization on features.
    - dropna (bool): If True, drops rows with NaNs after technical indicators.
    
    Returns:
    - X (np.ndarray): Feature sequences
    - y (np.ndarray): Target labels (0 or 1)
    - df (pd.DataFrame): Final processed DataFrame
    """

    df = pd.read_csv(file_path, parse_dates=["Datetime"])
    df.set_index("Datetime", inplace=True)
    df = df.sort_index()
    
    # Drop incomplete rows
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

    # Log return
    # df['LogOpen'] = np.log(df['Open']) - np.log(df['Open'].shift(1))
    # df['LogHigh'] = np.log(df['High']) - np.log(df['High'].shift(1))
    # df['LogLow'] = np.log(df['Low']) - np.log(df['Low'].shift(1))
    df['LogClose'] = np.log(df['Close']) - np.log(df['Close'].shift(1))
    # df['LogVolume'] = np.log(df['Volume']) - np.log(df['Volume'].shift(1))

    # EWMA of Log Return
    # df['EWMA_LogOpen'] = df['LogOpen'].ewm(span=10, adjust=False).mean()
    # df['EWMA_LogHigh'] = df['LogHigh'].ewm(span=10, adjust=False).mean()
    # df['EWMA_LogLow'] = df['LogLow'].ewm(span=10, adjust=False).mean()
    df['EWMA_LogClose'] = df['LogClose'].ewm(span=10, adjust=False).mean()
    # df['EWMA_LogVolume'] = df['LogVolume'].ewm(span=10, adjust=False).mean()


    # Technical Indicators
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'])
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20)

    # Target: Binary label (1 if price goes up next time step)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop rows with NaNs
    if dropna:
        df = df.dropna().reset_index(drop=True)

    # Feature columns
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'LogClose', 'EWMA_LogClose',
                # 'LogOpen', 'LogHigh', 'LogLow', 'LogClose', 'LogVolume', 
                # 'EWMA_LogOpen', 'EWMA_LogHigh', 'EWMA_LogLow', 'EWMA_LogClose', 'EWMA_LogVolume', 
                'RSI', 'MACD', 'MACD_signal', 'ADX', 'CCI', 'BB_upper', 'BB_middle', 'BB_lower']

    # Normalize features
    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])

    # Create sequences
    X, y = create_sequences(df[features].values, df['Target'].values, window_size)

    return X, y, df

def create_sequences(data, target, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)
