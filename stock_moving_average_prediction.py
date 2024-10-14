import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to calculate the Relative Strength Index (RSI)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to fetch stock data
def fetch_data(ticker):
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    return data

# Function for backtesting the trading strategy
def backtest_strategy(data, predictions):
    initial_balance = 10000  # Starting balance
    balance = initial_balance
    position = 0  # No shares held initially
    trades = []  # To record trades

    # Loop through each data point to simulate trading
    for i in range(len(data)):
        # Buy signal
        if predictions[i] == 1 and position == 0:
            position = balance / data['Close'].iloc[i]  # Buy shares
            balance = 0  # Spend all balance
            trades.append(('BUY', data.index[i], data['Close'].iloc[i]))

        # Sell signal
        elif predictions[i] == 0 and position > 0:
            balance = position * data['Close'].iloc[i]  # Sell shares
            trades.append(('SELL', data.index[i], data['Close'].iloc[i]))
            position = 0  # No shares held

    # Final balance calculation if still holding shares
    if position > 0:
        balance = position * data['Close'].iloc[-1]
        trades.append(('SELL', data.index[-1], data['Close'].iloc[-1]))

    return balance, trades

# Function to highlight rows based on prediction
def highlight_rows(row):
    if row['Predicted'] == 1:
        return ['background-color: green'] * len(row)  # Green for buy
    elif row['Predicted'] == 0:
        return ['background-color: yellow'] * len(row)  # Yellow for sell
    else:
        return [''] * len(row)  # No color

# Main execution
if __name__ == "__main__":
    # Fetch stock data
    ticker = 'AAPL'
    data = fetch_data(ticker)

    # Feature engineering
    data['MA_50'] = data['Close'].rolling(window=50).mean()  # 50-day moving average
    data['MA_200'] = data['Close'].rolling(window=200).mean()  # 200-day moving average
    data['RSI'] = calculate_rsi(data)  # Calculate RSI
    data.dropna(inplace=True)  # Remove rows with NaN values

    # Create target variable for predictions
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)  # Price increase

    # Prepare features and target variable
    features = ['Close', 'MA_50', 'MA_200', 'RSI']
    X = data[features]
    y = data['Target']

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions for the entire dataset
    predictions = model.predict(X)

    # Backtest the strategy
    final_balance, trades = backtest_strategy(data, predictions)

    # Add predictions to the DataFrame
    data['Predicted'] = predictions

    # Create descriptions for predictions
    data['Description'] = np.where(data['Predicted'] == 1,
                                   'The model predicts a price increase tomorrow.',
                                   'The model predicts no price increase or a decrease tomorrow.')

    # Select last 20 rows for highlighting
    last_20_rows = data.tail(20)

    # Apply highlighting for the last 20 rows
    styled_data = last_20_rows.style.apply(highlight_rows, axis=1)

    # Save results to CSV
    csv_filename = f"stock_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    last_20_rows.to_csv(csv_filename, index=True)
    logging.info(f"Results saved to {csv_filename}")

    # Save styled DataFrame to an Excel file
    excel_filename = f"styled_stock_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    styled_data.to_excel(excel_filename, index=True, engine='openpyxl')
    logging.info(f"Styled results saved to {excel_filename}")

    # Output final balance and trades executed
    print(f"Final balance: ${final_balance:.2f}")
    print("Trades executed:")
    for trade in trades:
        print(trade)
