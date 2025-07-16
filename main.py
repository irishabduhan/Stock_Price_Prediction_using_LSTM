from preprocessing_pkg import preprocess_stock_data

file_path = "Dataset/RELIANCE_15min.csv"
X, y, df = preprocess_stock_data(file_path, window_size=10, normalize=True)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(df.head())


# 1. Save DataFrame `df` to CSV
df.to_csv("processed_data.csv", index=False)
print("✅ DataFrame saved to processed_data.csv")
