import yfinance as yf
import pandas as pd

# Download historical stock data
df = yf.download("RELIANCE.NS", start="2001-01-01", end="2024-12-31")

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
df.to_csv("RELIANCE_1D.csv", index=False)

# Print preview
print(df.head())
