import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import time
from datetime import datetime, timedelta
from GoogleNews import GoogleNews
from textblob import TextBlob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA

def get_stock_news(stock_symbol, num_articles=5):
    googlenews = GoogleNews(lang='en', region='IN')
    googlenews.search(stock_symbol)
    result = googlenews.result()
    df = pd.DataFrame(result)
    if not df.empty:
        df = df[['title', 'desc', 'date', 'link']].head(num_articles)
    return df

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0.05:
        return 'Positive'
    elif sentiment < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def get_news_sentiment(stock_symbol):
    news_df = get_stock_news(stock_symbol)
    if news_df.empty:
        return None, None
    
    news_df['sentiment'] = news_df['title'].apply(analyze_sentiment)
    sentiment_counts = news_df['sentiment'].value_counts()
    overall_sentiment = sentiment_counts.index[0] if not sentiment_counts.empty else 'Neutral'
    
    return news_df, overall_sentiment

def search_indian_stocks(query):
    # This is a simplified version. In a real-world scenario, you'd want to use a more comprehensive API or database
    indian_stocks = pd.read_csv('https://archives.nseindia.com/content/equities/EQUITY_L.csv')
    return indian_stocks[indian_stocks['SYMBOL'].str.contains(query, case=False) | 
                         indian_stocks['NAME OF COMPANY'].str.contains(query, case=False)]

# Function to fetch real stock data

def get_realtime_data(symbol):
    if not symbol:
        return None
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Check if we got valid data
        if 'symbol' not in info or 'longName' not in info:
            return None

        return {
            'symbol': info.get('symbol', 'N/A'),
            'companyName': info.get('longName', 'N/A'),
            'lastPrice': info.get('currentPrice', 'N/A'),
            'change': info.get('currentPrice', 0) - info.get('previousClose', 0),
            'pChange': ((info.get('currentPrice', 0) - info.get('previousClose', 0)) / info.get('previousClose', 1)) * 100 if info.get('previousClose', 0) != 0 else 0,
            'open': info.get('open', 'N/A'),
            'dayHigh': info.get('dayHigh', 'N/A'),
            'dayLow': info.get('dayLow', 'N/A'),
            'previousClose': info.get('previousClose', 'N/A'),
            'bid': info.get('bid', 'N/A'),
            'ask': info.get('ask', 'N/A'),
            'dayRange': f"{info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}",
            'weekRange': f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}",
            'volume': info.get('volume', 'N/A'),
            'avgVolume': info.get('averageVolume', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'peRatio': info.get('trailingPE', 'N/A'),
            'eps': info.get('trailingEps', 'N/A'),
            'earningsDate': info.get('earningsTimestamp', 'N/A'),
            'forwardPE': info.get('forwardPE', 'N/A'),
            'dividendYield': info.get('dividendYield', 'N/A'),
            'exDividendDate': info.get('exDividendDate', 'N/A'),
            'targetEstimate': info.get('targetMeanPrice', 'N/A'),
        }
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def get_market_data(start_date, end_date, symbol='^NSEI'):  # Nifty 50 index
    market = yf.Ticker(symbol)
    market_data = market.history(start=start_date, end=end_date)
    return market_data['Close']


def get_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data
####################################real tme data ends here
# Technical indicators
def calculate_technical_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = calculate_rsi(data['Close'], window=14)
    data['MACD'], data['Signal'], data['MACD_Hist'] = calculate_macd(data['Close'])
    data['BB_upper'], data['BB_lower'] = calculate_bollinger_bands(data['Close'])
    data['Stochastic'] = calculate_stochastic(data)
    data['ADX'] = calculate_adx(data)
    data['OBV'] = calculate_obv(data)
    data['VWAP'] = calculate_vwap(data)
    data['Parabolic_SAR'] = calculate_parabolic_sar(data)
    data['Chaikin_MF'] = calculate_chaikin_money_flow(data)
    return data

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, window=20):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def calculate_stochastic(data, window=14):
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    return 100 * (data['Close'] - low_min) / (high_max - low_min)

def calculate_adx(data, window=14):
    plus_dm = data['High'].diff()
    minus_dm = data['Low'].diff()
    tr = pd.concat([data['High'] - data['Low'], 
                    abs(data['High'] - data['Close'].shift()), 
                    abs(data['Low'] - data['Close'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/window, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()
    return adx

def calculate_obv(data):
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

def calculate_vwap(data):
    return (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()

def calculate_parabolic_sar(data, step=0.02, max_step=0.2):
    sar = data['Close'].copy()
    trend = 1
    ep = data['Low'].min()
    af = step
    for i in range(1, len(data)):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if trend == 1:
            if data['Low'][i] < sar[i]:
                trend = 0
                sar[i] = ep
                af = step
                ep = data['Low'][i]
            else:
                if data['High'][i] > ep:
                    ep = data['High'][i]
                    af = min(af + step, max_step)
        else:
            if data['High'][i] > sar[i]:
                trend = 1
                sar[i] = ep
                af = step
                ep = data['High'][i]
            else:
                if data['Low'][i] < ep:
                    ep = data['Low'][i]
                    af = min(af + step, max_step)
    return sar

def calculate_chaikin_money_flow(data, window=20):
    mfv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    mfv = mfv.fillna(0)  # Handle division by zero cases
    mfv_volume = mfv * data['Volume']
    cmf = mfv_volume.rolling(window=window).sum() / data['Volume'].rolling(window=window).sum()
    return cmf

# Performance metrics
def calculate_performance_metrics(data):
    daily_returns = data['Close'].pct_change()
    market_data = get_market_data(data.index[0], data.index[-1])
    
    metrics = {
        'Daily Return': daily_returns.mean(),
        'Volatility': daily_returns.std(),
        'Sharpe Ratio': (daily_returns.mean() / daily_returns.std()) * np.sqrt(252),
        'Max Drawdown': calculate_max_drawdown(data['Close']),
        'Beta': calculate_beta(data['Close'], market_data)
    }
    return metrics

def calculate_max_drawdown(prices):
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices/peak - 1.0)
    return drawdown.min()

def get_market_data(start_date, end_date, symbol='^GSPC'):
    market = yf.Ticker(symbol)
    market_data = market.history(start=start_date, end=end_date)
    return market_data['Close']

def calculate_beta(stock_returns, market_returns):
    common_index = stock_returns.index.intersection(market_returns.index)
    stock_returns = stock_returns.loc[common_index]
    market_returns = market_returns.loc[common_index]
    
    stock_returns = stock_returns.pct_change().dropna()
    market_returns = market_returns.pct_change().dropna()
    
    min_length = min(len(stock_returns), len(market_returns))
    stock_returns = stock_returns[-min_length:]
    market_returns = market_returns[-min_length:]
    
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    
    return covariance / market_variance if market_variance != 0 else np.nan

 #########advanced analysis
def calculate_fibonacci_levels(data):
    max_price = data['High'].max()
    min_price = data['Low'].min()
    diff = max_price - min_price
    levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    fib_levels = [(max_price - level * diff) for level in levels]
    return fib_levels

def calculate_pivot_points(data):
    pivot = (data['High'].iloc[-1] + data['Low'].iloc[-1] + data['Close'].iloc[-1]) / 3
    s1 = 2 * pivot - data['High'].iloc[-1]
    s2 = pivot - (data['High'].iloc[-1] - data['Low'].iloc[-1])
    r1 = 2 * pivot - data['Low'].iloc[-1]
    r2 = pivot + (data['High'].iloc[-1] - data['Low'].iloc[-1])
    return {'Pivot': pivot, 'S1': s1, 'S2': s2, 'R1': r1, 'R2': r2}

def perform_monte_carlo_simulation(data, num_simulations=1000, time_horizon=30):
    returns = data['Close'].pct_change()
    last_price = data['Close'].iloc[-1]
    
    simulation_df = pd.DataFrame()
    
    for i in range(num_simulations):
        prices = [last_price]
        for _ in range(time_horizon):
            prices.append(prices[-1] * (1 + np.random.normal(returns.mean(), returns.std())))
        simulation_df[i] = prices
    
    return simulation_df

def lstm_prediction(data, future_days=30):
    if len(data) < 60:
        return None  # Not enough data for LSTM prediction
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    
    if not x_train or not y_train:
        return None  # Not enough data to create training sets
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=0)
    
    inputs = data['Close'].values[-60:]
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    
    X_test = []
    for i in range(min(future_days, len(inputs) - 59)):
        X_test.append(inputs[i:i+60, 0])
    
    if not X_test:
        return None  # Not enough data for prediction
    
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    # Pad the prediction if necessary
    if len(predicted_prices) < future_days:
        padding = np.full((future_days - len(predicted_prices), 1), np.nan)
        predicted_prices = np.vstack((predicted_prices, padding))
    
    return predicted_prices.flatten()

def arima_prediction(data, future_days=30):
    model = ARIMA(data['Close'], order=(1, 1, 1))
    results = model.fit()
    forecast = results.forecast(steps=future_days)
    return forecast

def plot_advanced_analysis(data, stock_symbol):
    # Calculate advanced indicators
    fib_levels = calculate_fibonacci_levels(data)
    pivot_points = calculate_pivot_points(data)
    monte_carlo = perform_monte_carlo_simulation(data)
    lstm_pred = lstm_prediction(data)
    arima_pred = arima_prediction(data)

    # Create subplot figure
    fig = make_subplots(rows=3, cols=2, subplot_titles=('Price with Fibonacci Levels', 'Pivot Points',
                                                        'Monte Carlo Simulation', 'LSTM Prediction',
                                                        'ARIMA Prediction', 'Comparison of Predictions'))

    # Price with Fibonacci Levels
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)
    for level, value in zip(['0', '23.6%', '38.2%', '50%', '61.8%', '78.6%', '100%'], fib_levels):
        fig.add_trace(go.Scatter(x=data.index, y=[value]*len(data), name=f'Fib {level}',
                                 line=dict(dash='dash')), row=1, col=1)

    # Pivot Points
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'), row=1, col=2)
    for level, value in pivot_points.items():
        fig.add_trace(go.Scatter(x=[data.index[-1]], y=[value], name=level,
                                 mode='markers', marker=dict(size=10)), row=1, col=2)

    # Monte Carlo Simulation
    for i in range(monte_carlo.shape[1]):
        fig.add_trace(go.Scatter(y=monte_carlo[i], name=f'Sim {i}', opacity=0.1), row=2, col=1)

    # LSTM Prediction
    fig.add_trace(go.Scatter(y=data['Close'].values[-30:], name='Actual', line=dict(color='blue')), row=2, col=2)
    fig.add_trace(go.Scatter(y=lstm_pred, name='LSTM Prediction', line=dict(color='red')), row=2, col=2)

    # ARIMA Prediction
    fig.add_trace(go.Scatter(y=data['Close'].values[-30:], name='Actual', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(y=arima_pred, name='ARIMA Prediction', line=dict(color='green')), row=3, col=1)

    # Comparison of Predictions
    fig.add_trace(go.Scatter(y=data['Close'].values[-30:], name='Actual', line=dict(color='blue')), row=3, col=2)
    fig.add_trace(go.Scatter(y=lstm_pred, name='LSTM', line=dict(color='red')), row=3, col=2)
    fig.add_trace(go.Scatter(y=arima_pred, name='ARIMA', line=dict(color='green')), row=3, col=2)

    fig.update_layout(height=1500, title_text=f"Advanced Analysis for {stock_symbol}")
    return fig

# Usage in Streamlit

 ########advanced anaysis ends here

def set_custom_theme():
    st.set_page_config(
        page_title="StockInsight Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
    .main > div {
        padding-left: 2rem;
        padding-right: 2rem;
    }
   
    .stButton>button {
        background-color: #3498db;
        color: #ffffff;
    }
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #ffffff;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
    }
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 500;
    }
    .positive-delta {
        color: #38a169;
    }
    .negative-delta {
        color: #e53e3e;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
def add_header():
    st.markdown("""
    <div style="background-color:#2c3e50;padding:1rem;border-radius:0.5rem;margin-bottom:1rem;">
    <h1 style="color:white;text-align:center;font-size:2rem;margin:0;">📈 StockInsight Pro</h1>
    <h3 style="color:#3498db;text-align:center;font-size:1rem;margin:0.5rem 0 0 0;">Advanced Stock Market Analysis Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)

# Footer
def add_footer():
    st.markdown("""
    <div style="background-color:#2c3e50;padding:0.5rem;border-radius:0.5rem;text-align:center;font-size:0.8rem;margin-top:1rem;">
    <p style="color:white;margin:0;">© 2024 StockInsight Pro. All rights reserved.</p>
    <p style="color:#3498db;margin:0;">Disclaimer: This app is for educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

# Main layout
set_custom_theme()
add_header()

def display_metric(label, value, delta=None):
    delta_html = ""
    if delta:
        delta_class = "positive-delta" if float(delta.split()[0]) >= 0 else "negative-delta"
        delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://www.nseindia.com/assets/images/NSE_Logo.svg", width=150)
    st.title("Stock Analyzer")
    
    # Replace selectbox with search bar
    search_query = st.text_input("Search for a stock (e.g., RELIANCE, TCS)")
    if search_query:
        search_results = search_indian_stocks(search_query)
        if not search_results.empty:
            selected_stock = st.selectbox("Select a stock", 
                                          options=search_results['SYMBOL'].tolist(),
                                          format_func=lambda x: f"{x} - {search_results[search_results['SYMBOL'] == x]['NAME OF COMPANY'].values[0]}")
            stock_symbol = f"{selected_stock}.NS"  # Append .NS for NSE stocks
        else:
            st.warning("No matching stocks found. Please try a different search term.")
            stock_symbol = None
    else:
        stock_symbol = None
        
    current_date = datetime.now()

    # Calculate the date one month before the current date
    one_year_ago = current_date - timedelta(days=365)

    # Set the start and end date inputs with default values
    start_date = st.date_input("Start Date", one_year_ago)
    end_date = st.date_input("End Date", current_date)
    
    indicators = st.multiselect(
        "Choose indicators to plot:",
        ['SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'Bollinger Bands', 'Stochastic', 'ADX', 'OBV', 'VWAP', 'Parabolic SAR', 'Chaikin Money Flow'],
        default=['SMA_20', 'SMA_50', 'EMA_20']
    )
    if st.button("Perform Advanced Analysis"):
        display_advanced_analysis(stock_symbol, start_date, end_date)

# Main content
if stock_symbol:
    
    @st.cache_data(ttl=2)
    def get_realtime_data_cached(symbol):
        return get_realtime_data(symbol)
    realtime_data = get_realtime_data_cached(stock_symbol)
    try:
        # Fetch real-time data once
        def update_realtime_data():
            realtime_data = get_realtime_data_cached(stock_symbol)
        
        if realtime_data is None:
            st.error(f"Unable to fetch real-time data for {stock_symbol}. Please check the stock symbol and try again.")
        else:
            st.subheader(f"{realtime_data['companyName']} ({realtime_data['symbol']}) - Real-time Data")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                display_metric("Current Price", f"₹{realtime_data['lastPrice']}" if realtime_data['lastPrice'] != 'N/A' else 'N/A', 
                               f"{realtime_data['change']:.2f} ({realtime_data['pChange']:.2f}%)" if realtime_data['change'] != 'N/A' else 'N/A')
            with col2:
                display_metric("Previous Close", f"₹{realtime_data['previousClose']}" if realtime_data['previousClose'] != 'N/A' else 'N/A')
            with col3:
                display_metric("Open", f"₹{realtime_data['open']}" if realtime_data['open'] != 'N/A' else 'N/A')
            with col4:
                display_metric("Volume", f"{realtime_data['volume']:,}" if realtime_data['volume'] != 'N/A' else 'N/A')
            
            st.subheader("Additional Stock Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Bid:** ₹{realtime_data['bid']}")
                st.write(f"**Ask:** ₹{realtime_data['ask']}")
                st.write(f"**Day's Range:** ₹{realtime_data['dayRange']}")
                st.write(f"**52 Week Range:** ₹{realtime_data['weekRange']}")
                st.write(f"**Avg. Volume:** {realtime_data['avgVolume']:,}")
            with col2:
                st.write(f"**Beta (5Y Monthly):** {realtime_data['beta']}")
                st.write(f"**Market Cap:** ₹{realtime_data['marketCap']:,}")
                st.write(f"**PE Ratio (TTM):** {realtime_data['peRatio']}")
                st.write(f"**EPS (TTM):** ₹{realtime_data['eps']}")
                st.write(f"**Forward PE:** {realtime_data['forwardPE']}")
            with col3:
                st.write(f"**Earnings Date:** {realtime_data['earningsDate']}")
                st.write(f"**Dividend & Yield:** {realtime_data['dividendYield']}")
                st.write(f"**Ex-Dividend Date:** {realtime_data['exDividendDate']}")
                st.write(f"**1y Target Est:** ₹{realtime_data['targetEstimate']}")
        
        # Fetch and process historical data
        data = get_stock_data(stock_symbol, start_date, end_date)
        if data.empty:
            st.error(f"No historical data available for {stock_symbol} between {start_date} and {end_date}. Please try a different date range or stock symbol.")
        else:
            data = calculate_technical_indicators(data)
            
            # Plotting
            st.subheader(f"{stock_symbol} Historical Analysis")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                subplot_titles=('Stock Price with Indicators', 'Volume'),
                                row_heights=[0.7, 0.3])
            
            # Add traces for candlestick and indicators
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                         low=data['Low'], close=data['Close'], name='Candlestick'), row=1, col=1)
            
            # Add selected indicators
            for indicator in indicators:
                if indicator in data.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=data[indicator], mode='lines', name=indicator), row=1, col=1)
            
            # Add volume
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=2, col=1)
            
            # Update layout
            fig.update_layout(
                height=800,
                template="plotly_white",
                title_text=f"Stock Analysis for {stock_symbol}",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    st.subheader(f"{stock_symbol} News Sentiment Analysis")
    news_df, overall_sentiment = get_news_sentiment(stock_symbol)
    if news_df is not None:
        st.write(f"**Overall Market Sentiment:** {overall_sentiment}")
        
        # Display sentiment distribution
        sentiment_counts = news_df['sentiment'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hole=.3)])
        fig.update_layout(title_text="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        #Advanced analysis
        def display_advanced_analysis(stock_symbol, start_date, end_date):
            data = yf.download(stock_symbol, start=start_date, end=end_date)
        if not data.empty:
            fig = plot_advanced_analysis(data, stock_symbol)
        st.plotly_chart(fig, use_container_width=True)
        

        # Display news articles
        st.subheader("Recent News Articles")
        for _, row in news_df.iterrows():
            st.markdown(f"**{row['title']}** - {row['date']}")
            st.write(row['desc'])
            st.markdown(f"Sentiment: {row['sentiment']} | [Read more]({row['link']})")
            st.markdown("---")
        else:
            st.warning(f"No recent news found for {stock_symbol}")

    else:
        st.info("Please search and select a stock to begin analysis.")

else:
    st.info("Please search and select a stock to begin analysis.")

# Add footer
add_footer()
