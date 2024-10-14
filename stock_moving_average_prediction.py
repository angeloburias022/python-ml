import yfinance as yf
import pandas as pd

# Fetch AAPL stock data
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

# Calculate the 50-day and 200-day moving averages
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['MA_200'] = data['Close'].rolling(window=200).mean()

# Create a target column: 1 if the next day's close is higher, 0 otherwise
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Drop rows with missing data
data = data.dropna()

# Add a descriptive column
def interpret_values(row):
    if row['Target'] == 1:
        target_desc = "Expected price increase tomorrow."
    else:
        target_desc = "Expected price decrease or no change tomorrow."
        
    if row['MA_50'] > row['MA_200']:
        trend_desc = "Short-term trend is bullish."
    else:
        trend_desc = "Short-term trend is bearish."
        
    return f"{target_desc} {trend_desc}"

data['Description'] = data.apply(interpret_values, axis=1)

# Display the first few rows
print(data[['Close', 'MA_50', 'MA_200', 'Target', 'Description']].head())
