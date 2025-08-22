import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
import google.generativeai as genai
import openai
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# Load config
with open('config.json', 'r') as f:
    config = json.load(f)


# Extract parameters
SYMBOL = config['data']['symbol']
INTERVAL = '1m'
START_DATE = config['data']['start_date']
DATA_LIMIT = config['data']['data_limit']
TIME_STEP = config['model']['time_step']
LSTM_UNITS = config['model']['lstm_units']
EPOCHS = config['model']['epochs']
BATCH_SIZE = config['model']['batch_size']
QUANTITY_BASE = config['trading']['quantity']
PRICE_THRESHOLD = config['trading']['price_threshold']
SENTIMENT_BUY = config['trading']['sentiment_buy_threshold']
SENTIMENT_SELL = config['trading']['sentiment_sell_threshold']
TRADE_INTERVAL = config['trading']['trade_interval']
INITIAL_CAPITAL = config['trading']['initial_capital']
FEE_RATE = config['trading']['fee_rate']
METRICS_INTERVAL = config['trading']['metrics_interval']
RISK_PER_TRADE = config['trading']['risk_per_trade']
STOP_LOSS = config['trading']['stop_loss']
MAX_TRADES_PER_DAY = config['trading']['max_trades_per_day']
NEWS_SOURCE = (config['sentiment']['news_source']
              .replace('YOUR_NEWS_API_KEY', os.getenv('NEWS_API_KEY', '')))
NEWS_LIMIT = int(config['sentiment']['news_limit'])
GEMINI_MODEL = config['sentiment']['gemini_model']
OPENAI_MODEL = config['sentiment']['openai_model']
LOG_FILE = config['general']['log_file']
DAILY_METRICS_FILE = config['general']['daily_metrics_file']
TESTNET = config['general']['testnet']

# Backward compatible aliases to prevent NameError if old names linger
SENTIMENT_BUY_THRESHOLD = SENTIMENT_BUY
SENTIMENT_SELL_THRESHOLD = SENTIMENT_SELL


# API setup
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')
trade_client = Client(
    os.getenv('BINANCE_TESTNET_API_KEY'),
    os.getenv('BINANCE_TESTNET_API_SECRET'),
    testnet=TESTNET
)


# Data generation and utilities
def get_historical_data():
    try:
        dates = pd.date_range(start=START_DATE, periods=DATA_LIMIT, freq='D')
        base_prices = np.linspace(46000.0, 113000.0, DATA_LIMIT)
        volatility = np.random.normal(0, 0.05, DATA_LIMIT)
        prices = base_prices * (1 + volatility)
        prices = np.clip(prices, 40000.0, 120000.0)
        data = pd.DataFrame({'timestamp': dates, 'close': prices})
        data['open'] = data['close'].shift(1).fillna(data['close'])
        data['high'] = data[['open', 'close']].max(axis=1)
        data['low'] = data[['open', 'close']].min(axis=1)
        data['volume'] = np.random.randint(1000, 10000, DATA_LIMIT)
        print(
            f"AI-generated historical data: {len(data)} days "
            f"from {START_DATE} to {dates[-1].date()}"
        )
        return data
    except Exception as e:
        print(f"Error generating historical data: {e}")
        return pd.DataFrame()


def get_current_price():
    try:
        base_price = 113000.0
        volatility = np.random.normal(0, 0.05)
        ai_price = base_price * (1 + volatility)
        ai_price = max(100000.0, min(120000.0, ai_price))
        print(
            f"AI-generated current price: ${ai_price:.2f} at "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return ai_price
    except Exception as e:
        print(f"Error generating AI price: {e}")
        return 113000.0


def get_global_factors():
    try:
        vix_data = yf.download(
            '^VIX', period='1d', progress=False, auto_adjust=False
        )
        vix = float(vix_data['Close'].iloc[-1]) if not vix_data.empty else 20.0
        return {'vix': vix}
    except Exception as e:
        print(f"Error fetching VIX: {e}, using default value 20.0")
        return {'vix': 20.0}


def get_x_sentiment(query="bitcoin price sentiment"):
    # Placeholder score
    return 0.6


def prepare_scaled_data():
    data = get_historical_data()
    if data.empty:
        print("No historical data available, using fallback.")
        return None, None
    global_factors = get_global_factors()
    data['vix'] = global_factors['vix']
    x_sentiment = get_x_sentiment()
    data['x_sentiment'] = x_sentiment
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['close', 'vix', 'x_sentiment']])
    return scaled_data, scaler


def create_dataset(dataset, time_step):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), :])
        Y.append(dataset[i + time_step, 0])  # Close column
    return np.array(X), np.array(Y)


# Train models
scaled_data, scaler = prepare_scaled_data()
if scaled_data is None or scaler is None:
    print("Failed to prepare data for training. Exiting.")
    raise SystemExit(1)

X_train, y_train = create_dataset(scaled_data, TIME_STEP)
X_train = np.reshape(X_train, (X_train.shape[0], TIME_STEP, 3))

model = Sequential()
model.add(Input(shape=(TIME_STEP, 3)))
model.add(GRU(LSTM_UNITS, return_sequences=True))
model.add(GRU(LSTM_UNITS))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)


def predict_next_price():
    scaled, sc = prepare_scaled_data()
    if scaled is None or sc is None or len(scaled) < TIME_STEP:
        return 0.0
    last_window = scaled[-TIME_STEP:]
    gru_pred = model.predict(
        np.reshape(last_window, (1, TIME_STEP, 3)), verbose=0
    )[0][0]
    rf_pred = rf_model.predict(last_window.reshape(1, -1))[0]
    pred_scaled = (gru_pred + rf_pred) / 2.0
    # Correct inverse for MinMaxScaler: X = (X_scaled - min_) / scale_
    price_scale = sc.scale_[0]
    price_min = sc.min_[0]
    inverse_pred = (pred_scaled - price_min) / price_scale
    return float(inverse_pred)


# Sentiment
def get_gemini_sentiment(news_texts):
    try:
        gmodel = genai.GenerativeModel(GEMINI_MODEL)
        prompt = (
            f"Analyze these for {SYMBOL} sentiment as a single float between 0 "
            f"and 1, where 1 is very bullish: {'; '.join(news_texts)}"
        )
        response = gmodel.generate_content(prompt)
        return float(response.text.strip())
    except Exception:
        return 0.5


def get_openai_sentiment(news_texts):
    try:
        prompt = (
            f"Analyze these news for {SYMBOL} sentiment as a single float "
            f"between 0 and 1, where 1 is very bullish: {'; '.join(news_texts)}"
        )
        # Legacy ChatCompletion for compatibility
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return float(resp.choices[0].message['content'].strip())
    except Exception:
        return 0.5


def get_recent_news():
    try:
        r = requests.get(NEWS_SOURCE, timeout=10)
        r.raise_for_status()
        data = r.json().get('data', [])[:NEWS_LIMIT]
        return [item.get('title', '') for item in data] or ["No news."]
    except Exception as e:
        print(f"Error fetching news: {e}")
        return ["No news."]


# Metrics
def calculate_metrics(trade_log, current_price, initial_capital):
    if not trade_log:
        return {
            'daily_earnings': 0.0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'num_trades': 0,
            'avg_trade_profit': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'portfolio_value': initial_capital
        }
    df = pd.DataFrame(trade_log)
    df['time'] = pd.to_datetime(df['time'])
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    daily_trades = df[df['time'] >= today]
    # Daily earnings
    daily_earnings = 0.0
    buy_trades = daily_trades[daily_trades['action'] == 'buy'].iloc[::-1] \
        .to_dict('records')
    sell_trades = daily_trades[daily_trades['action'] == 'sell'] \
        .to_dict('records')
    for sell in sell_trades:
        for buy in buy_trades:
            if buy['time'] < sell['time']:
                profit = (sell['price'] - buy['price']) * buy['quantity'] * \
                         (1 - 2 * FEE_RATE)
                daily_earnings += profit
                buy_trades.remove(buy)
                break
    # Total P&L
    total_pnl = 0.0
    all_buy_trades = df[df['action'] == 'buy'].iloc[::-1].to_dict('records')
    all_sell_trades = df[df['action'] == 'sell'].to_dict('records')
    for sell in all_sell_trades:
        for buy in all_buy_trades:
            if buy['time'] < sell['time']:
                profit = (sell['price'] - buy['price']) * buy['quantity'] * \
                         (1 - 2 * FEE_RATE)
                total_pnl += profit
                all_buy_trades.remove(buy)
                break
    # Win rate and avg profit
    profits = []
    all_buy_trades = df[df['action'] == 'buy'].iloc[::-1].to_dict('records')
    for sell in all_sell_trades:
        for buy in all_buy_trades:
            if buy['time'] < sell['time']:
                profit = (sell['price'] - buy['price']) * buy['quantity']
                profits.append(profit)
                all_buy_trades.remove(buy)
                break
    win_rate = len([p for p in profits if p > 0]) / len(profits) if profits \
        else 0.0
    num_trades = int(len(df[df['action'].isin(['buy', 'sell'])]))
    avg_trade_profit = float(np.mean(profits)) if profits else 0.0
    # Simple portfolio and drawdown
    portfolio_values = [INITIAL_CAPITAL]
    current_cash = INITIAL_CAPITAL
    trades_open = []
    for _, trade in df.iterrows():
        if trade['action'] == 'buy':
            trades_open.append({
                'buy_price': trade['price'],
                'quantity': trade['quantity']
            })
            current_cash -= trade['price'] * trade['quantity'] * (1 + FEE_RATE)
        elif trade['action'] == 'sell' and trades_open:
            trades_open.pop(0)
            current_cash += trade['price'] * trade['quantity'] * (1 - FEE_RATE)
        portfolio_values.append(
            current_cash + sum(t['quantity'] * current_price for t in trades_open)
        )
    max_drawdown = max(0, max(portfolio_values) - min(portfolio_values))
    portfolio_value = current_cash + sum(
        t['quantity'] * current_price for t in trades_open
    )
    # Sharpe placeholder to avoid divide by zero with small logs
    daily_returns = np.diff(portfolio_values) if len(portfolio_values) > 1 \
        else np.array([0.0])
    sharpe_ratio = float(np.mean(daily_returns) / np.std(daily_returns)) \
        if np.std(daily_returns) != 0 else 0.0
    return {
        'daily_earnings': float(daily_earnings),
        'total_pnl': float(total_pnl),
        'win_rate': float(win_rate),
        'num_trades': int(num_trades),
        'avg_trade_profit': float(avg_trade_profit),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'portfolio_value': float(portfolio_value)
    }


def adjust_parameters(daily_earnings):
    global PRICE_THRESHOLD, SENTIMENT_BUY, SENTIMENT_SELL
    MIN_EARNINGS = 1000.0
    if daily_earnings < MIN_EARNINGS:
        PRICE_THRESHOLD = max(0.0001, PRICE_THRESHOLD * 0.9)
        SENTIMENT_BUY = max(0.3, SENTIMENT_BUY - 0.05)
        SENTIMENT_SELL = min(0.3, SENTIMENT_SELL + 0.05)
        print(
            f"Auto-correction applied: Price Threshold={PRICE_THRESHOLD}, "
            f"Buy Threshold={SENTIMENT_BUY}, Sell Threshold={SENTIMENT_SELL}"
        )


# Trading loop setup
trade_log = []
last_metrics_time = time.time()
current_capital = INITIAL_CAPITAL
daily_trade_count = 0
last_day = datetime.now().date()
open_positions = []
last_daily_earnings = 0.0

# RL environment placeholder
env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
model_rl = PPO("MlpPolicy", env, verbose=0)
observation = env.reset()

# Initialize empty log files
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        'time', 'price', 'predicted', 'sentiment', 'action', 'quantity'
    ]).to_csv(LOG_FILE, index=False)
    print(f"Initialized empty {LOG_FILE}")
if not os.path.exists(DAILY_METRICS_FILE):
    pd.DataFrame(columns=[
        'date', 'daily_earnings', 'total_pnl', 'win_rate', 'num_trades',
        'avg_trade_profit', 'sharpe_ratio', 'max_drawdown', 'portfolio_value'
    ]).to_csv(DAILY_METRICS_FILE, index=False)
    print(f"Initialized empty {DAILY_METRICS_FILE}")

print("Starting trading loop with capital:", current_capital)
last_save_time = time.time()
SAVE_INTERVAL = 3600

while True:
    current_price = get_current_price()
    if current_price == 0.0:
        print("Price generation failed. Retrying...")
        time.sleep(60)
        continue

    current_day = datetime.now().date()
    if current_day != last_day:
        daily_trade_count = 0
        last_day = current_day
        adjust_parameters(last_daily_earnings)

    if daily_trade_count >= MAX_TRADES_PER_DAY:
        print("Max trades reached. Waiting...")
        time.sleep(TRADE_INTERVAL)
        continue

    # Dynamic stop-loss
    vix = get_global_factors()['vix']
    dynamic_stop_loss = max(0.01, STOP_LOSS * (vix / 20 if vix > 0 else 1))

    # Stop-loss check
    for pos in open_positions[:]:
        if current_price < pos['buy_price'] * (1 - dynamic_stop_loss):
            try:
                trade_client.order_market_sell(symbol=SYMBOL,
                                               quantity=pos['quantity'])
            except Exception as e:
                print(f"Binance sell error on stop-loss: {e}")
            trade_log.append({
                'time': time.ctime(),
                'price': current_price,
                'predicted': 0,
                'sentiment': 0,
                'action': 'sell',
                'quantity': pos['quantity']
            })
            current_capital += current_price * pos['quantity'] * (1 - FEE_RATE)
            open_positions.remove(pos)
            daily_trade_count += 1
            print(f"Stop-loss: Sold {pos['quantity']} at {current_price}")

    predicted_price = predict_next_price()
    news = get_recent_news()
    gemini_score = get_gemini_sentiment(news)
    openai_score = get_openai_sentiment(news)
    sentiment_score = (gemini_score + openai_score) / 2.0

    # RL action placeholder
    action_vec, _states = model_rl.predict(observation, deterministic=True)
    observation, rewards, dones, infos = env.step(action_vec)
    _ = last_daily_earnings  # Placeholder to show feedback is available

    # Risk-based quantity
    quantity = max(QUANTITY_BASE, (current_capital * RISK_PER_TRADE) /
                   current_price)

    # Trading logic
    action = "hold"
    if (predicted_price > current_price * (1 + PRICE_THRESHOLD) and
            sentiment_score > SENTIMENT_BUY):
        try:
            trade_client.order_market_buy(symbol=SYMBOL, quantity=quantity)
        except Exception as e:
            print(f"Binance buy error: {e}")
        action = "buy"
        current_capital -= current_price * quantity * (1 + FEE_RATE)
        open_positions.append({'buy_price': current_price, 'quantity': quantity})
        daily_trade_count += 1
        print(
            f"Buy {quantity} at {current_price}, Pred: {predicted_price}, "
            f"Sentiment: {sentiment_score}"
        )
    elif (predicted_price < current_price * (1 - PRICE_THRESHOLD) and
          sentiment_score < SENTIMENT_SELL):
        try:
            trade_client.order_market_sell(symbol=SYMBOL, quantity=quantity)
        except Exception as e:
            print(f"Binance sell error: {e}")
        action = "sell"
        current_capital += current_price * quantity * (1 - FEE_RATE)
        daily_trade_count += 1
        print(
            f"Sell {quantity} at {current_price}, Pred: {predicted_price}, "
            f"Sentiment: {sentiment_score}"
        )

    # Log trade
    trade_log.append({
        'time': time.ctime(),
        'price': current_price,
        'predicted': predicted_price,
        'sentiment': sentiment_score,
        'action': action,
        'quantity': quantity
    })
    pd.DataFrame(trade_log).to_csv(LOG_FILE, index=False)
    print(f"Trade saved: {action} {quantity} at {current_price}")

    # Periodic save and metrics
    if time.time() - last_save_time >= SAVE_INTERVAL:
        pd.DataFrame(trade_log).to_csv(LOG_FILE, index=False)
        print(f"Periodic trade log saved to {LOG_FILE}")
        if time.time() - last_metrics_time >= METRICS_INTERVAL:
            metrics = calculate_metrics(trade_log, current_price, INITIAL_CAPITAL)
            last_daily_earnings = metrics['daily_earnings']
            daily_metrics = [{'date': datetime.now().date(), **metrics}]
            pd.DataFrame(daily_metrics).to_csv(
                DAILY_METRICS_FILE,
                mode='a',
                header=not os.path.exists(DAILY_METRICS_FILE),
                index=False
            )
            print(
                f"Periodic metrics saved for {datetime.now().date()} with "
                f"earnings: {metrics['daily_earnings']}"
            )
            last_metrics_time = time.time()
        last_save_time = time.time()

    if time.time() - last_metrics_time >= METRICS_INTERVAL:
        metrics = calculate_metrics(trade_log, current_price, INITIAL_CAPITAL)
        last_daily_earnings = metrics['daily_earnings']
        print(f"\nOMNITRADES Metrics ({datetime.now().date()}):")
        for k, v in metrics.items():
            if k in ['daily_earnings', 'total_pnl', 'avg_trade_profit',
                     'max_drawdown', 'portfolio_value']:
                print(f"{k.replace('_', ' ').title()}: ${v:.2f}")
            elif k == 'sharpe_ratio':
                print(f"{k.replace('_', ' ').title()}: {v:.2f}")
            elif k == 'win_rate':
                print(f"{k.replace('_', ' ').title()}: {int(v*100)}%")
            else:
                print(f"{k.replace('_', ' ').title()}: {v}")
        daily_metrics = [{'date': datetime.now().date(), **metrics}]
        pd.DataFrame(daily_metrics).to_csv(
            DAILY_METRICS_FILE,
            mode='a',
            header=not os.path.exists(DAILY_METRICS_FILE),
            index=False
        )
        print(
            f"Metrics saved for {datetime.now().date()} with "
            f"earnings: {metrics['daily_earnings']}"
        )
        last_metrics_time = time.time()

    time.sleep(TRADE_INTERVAL)
