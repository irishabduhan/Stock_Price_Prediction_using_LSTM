import pandas as pd
import numpy as np
import talib

# ------------------------
# Step 1: Load the CSV
# ------------------------
df = pd.read_csv("Dataset\\RELIANCE_15min.csv", parse_dates=["Datetime"])
df.set_index("Datetime", inplace=True)
df = df.sort_index()  # Ensure time order

# # Ensure datetime is parsed
# df['Datetime'] = pd.to_datetime(df['Datetime'])
# df = df.sort_values('Datetime').reset_index(drop=True)

# Drop any rows with missing values in essential columns
df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

# ------------------------
# Step 2: Log Return
# ------------------------
df['LogReturn'] = np.log(df['Close']) - np.log(df['Close'].shift(1))

# ------------------------
# Step 3: EWMA of Log Return
# ------------------------
df['EWMA_LogReturn'] = df['LogReturn'].ewm(span=10, adjust=False).mean()

# ------------------------
# Step 4: Technical Indicators from TA-Lib
# ------------------------
# Example indicators (you can add more from TA-Lib)
df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'])
df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20)

# Add more indicators as needed (up to 175 if following paper exactly)

# ------------------------
# Step 5: Binary Target Label (Price Up in Next Step)
# ------------------------
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# ------------------------
# Step 6: Drop rows with NaNs from indicator calculations
# ------------------------
df = df.dropna().reset_index(drop=True)

# # ------------------------
# # Step 7: Optional — Normalize Features (MinMax or Standard)
# # ------------------------
# from sklearn.preprocessing import StandardScaler

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'LogReturn', 'EWMA_LogReturn',
            'RSI', 'MACD', 'MACD_signal', 'ADX', 'CCI', 'BB_upper', 'BB_middle', 'BB_lower']

# scaler = StandardScaler()
# df[features] = scaler.fit_transform(df[features])

# ------------------------
# Step 8: Sequence Creation for LSTM (Optional Function)
# ------------------------
def create_sequences(data, target, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(target[i+window_size])
    return np.array(X), np.array(y)

X, y = create_sequences(df[features].values, df['Target'].values)

# ------------------------
# Final Output
# ------------------------
print(f"Final dataset shape: X = {X.shape}, y = {y.shape}")

