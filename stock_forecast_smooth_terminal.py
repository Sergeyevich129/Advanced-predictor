
import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

alpha_api_key = os.getenv("alpha_api_key")
finnhub_api_key = os.getenv("finnhub_api_key")
symbol = input("Enter stock symbol (e.g. AAPL): ").upper()

# Fetch price data
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=compact&apikey={alpha_api_key}"
response = requests.get(url)
data = response.json()

if "Time Series (Daily)" not in data:
    print("❌ Failed to fetch stock data.")
    exit()

df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
df = df.rename(columns={"4. close": "Close"})
df.index = pd.to_datetime(df.index)
df = df.sort_index()
df = df.astype(float)

# Feature engineering
df["SMA_10"] = df["Close"].rolling(window=10).mean()
df["Momentum"] = df["Close"] - df["Close"].shift(10)
df["Volatility"] = df["Close"].pct_change().rolling(window=10).std()
df["Future_10d"] = df["Close"].shift(-10)
df.dropna(inplace=True)

if len(df) < 40:
    print("⚠️ Not enough data to generate forecast.")
    exit()

# Model training
X = df[["SMA_10", "Momentum", "Volatility"]]
y = df["Future_10d"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Forecast next 10 days
last_X = X[-10:].copy()
predictions = []

for _ in range(10):
    pred = model.predict([last_X.iloc[-1]])[0]
    predictions.append(pred)
    next_row = last_X.iloc[-1].copy()
    next_row["SMA_10"] = (last_X["SMA_10"].mean() + pred) / 2
    next_row["Momentum"] = pred - last_X.iloc[-1]["SMA_10"]
    next_row["Volatility"] = last_X["Volatility"].mean()
    last_X = pd.concat([last_X, pd.DataFrame([next_row])], ignore_index=True)

# Print results
current_price = df["Close"].iloc[-1]
forecast_price = predictions[-1]
change = forecast_price - current_price
percent = (change / current_price) * 100

print(f"📍 Current Price for {symbol}: ${current_price:.2f}")
print(f"🔮 Predicted Price in 10 Days: ${forecast_price:.2f}")
print(f"📊 Change: ${change:.2f} ({percent:.2f}%)")

# Get company info
info_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={finnhub_api_key}"
info_response = requests.get(info_url)

if info_response.status_code == 200:
    profile = info_response.json()
    print("\n🏢 Company Information")
    print(f"Name: {profile.get('name', 'N/A')}")
    print(f"Industry: {profile.get('finnhubIndustry', 'N/A')}")
    print(f"Exchange: {profile.get('exchange', 'N/A')}")
    print(f"Website: {profile.get('weburl', 'N/A')}")
else:
    print("⚠️ Could not fetch company profile.")

# Export CSV
dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=10)
forecast_df = pd.DataFrame({"Date": dates, "Predicted Price": predictions})
forecast_df.to_csv(f"{symbol}_forecast_terminal.csv", index=False)
print(f"📁 Forecast saved to {symbol}_forecast_terminal.csv")
