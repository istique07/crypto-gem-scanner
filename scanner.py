
import requests
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

def get_coin_ohlcv(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': 7, 'interval': 'hourly'}
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['price'].astype(float)
    df['volume'] = [v[1] for v in data['total_volumes']]
    df.set_index('timestamp', inplace=True)
    return df

def apply_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    return df.dropna()

def build_ml_dataset(df):
    df['target'] = df['close'].shift(-3) > df['close']
    return df.dropna()

def train_buy_model(df):
    X = df[['rsi', 'volume']]
    y = df['target'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def train_price_forecast(df):
    df['target_price'] = df['close'].shift(-3)
    df = df.dropna()
    X = df[['rsi', 'volume']]
    y = df['target_price']
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

def run_multi_coin_ml_scan(limit=5):
    coins = ['bitcoin', 'ethereum', 'solana', 'cardano', 'polkadot'][:limit]
    results = []
    for coin in coins:
        df = get_coin_ohlcv(coin)
        df = apply_indicators(df)
        df_ml = build_ml_dataset(df)
        model_buy = train_buy_model(df_ml)
        model_price = train_price_forecast(df_ml)
        latest = df.iloc[-1:]
        signal = model_buy.predict(latest[['rsi', 'volume']])[0]
        forecast = model_price.predict(latest[['rsi', 'volume']])[0]
        results.append({
            'Coin': coin,
            'Current Price': latest['close'].values[0],
            'BUY Signal': 'BUY' if signal else 'HOLD',
            'AI Forecast': round(forecast, 2)
        })
    return pd.DataFrame(results)
