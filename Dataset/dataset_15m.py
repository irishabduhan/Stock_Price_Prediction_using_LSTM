import yfinance as yf
import pandas as pd

# Download historical stock data
# Fetch 15-minute interval data for the last 60 days
df = yf.download(
    tickers="RELIANCE.NS",
    period="60d",           # Max allowed for 15m data
    interval="15m",         # Intraday interval
    progress=True,
    auto_adjust=True
)

# Reset index to get 'Date' as a column
df = df.reset_index()

# Ensure column names are clean (no multi-index)
df.columns.name = None  # Removes 'Price' level
df.columns = [col if not isinstance(col, tuple) else col[0] for col in df.columns]  # Flatten if multi-level

# Round OHLC columns to 4 decimal places
for col in ['Open', 'High', 'Low', 'Close']:
    if col in df.columns:
        df[col] = df[col].round(5)

# Save cleaned DataFrame to CSV
df.to_csv("RELIANCE_15min.csv", index=False)

# Print preview
print(df.head())
