import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import time
from datetime import datetime, timedelta


def search_indian_stocks(query):
    # This is a simplified version. In a real-world scenario, you'd want to use a more comprehensive API or database
    indian_stocks = pd.read_csv('https://archives.nseindia.com/content/equities/EQUITY_L.csv')
    return indian_stocks[indian_stocks['SYMBOL'].str.contains(query, case=False) | 
                         indian_stocks['NAME OF COMPANY'].str.contains(query, case=False)]

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

# Function to fetch real stock data
def get_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data

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
    .stApp {
        background-color: #003844;
        color: #1e1e1e;
    }
    .stSidebar {
        background-color: #2c3e50;
        color: #ecf0f1;
    }
    .stButton>button {
        background-color: #3498db;
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        background-color: #ecf0f1;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #718096;
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
    
    start_date = st.date_input("Start Date", pd.to_datetime('2023-01-01'))
    end_date = st.date_input("End Date", pd.to_datetime('2023-12-31'))
    indicators = st.multiselect(
        "Choose indicators to plot:",
        ['SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'Bollinger Bands', 'Stochastic', 'ADX', 'OBV', 'VWAP', 'Parabolic SAR', 'Chaikin Money Flow'],
        default=['SMA_20', 'SMA_50', 'EMA_20']
    )


# Main content
if stock_symbol:
    try:
        # Create placeholders for real-time data
        realtime_data_placeholder = st.empty()
        
        # Function to update real-time data
        def update_realtime_data():
            realtime_data = get_realtime_data(stock_symbol)
            
            if realtime_data is None:
                realtime_data_placeholder.error(f"Unable to fetch real-time data for {stock_symbol}. Please check the stock symbol and try again.")
                return
            
            with realtime_data_placeholder.container():
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
                
                # ... (keep the rest of the real-time data display code, adding similar checks)

        # Initial update
        update_realtime_data()

        # Continuously update real-time data
        while True:
            time.sleep(5)  # Update every 5 seconds
            update_realtime_data()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

else:
    st.info("Please search and select a stock to begin analysis.")


try:
    realtime_data = get_realtime_data(stock_symbol)
    
    st.subheader(f"{realtime_data['companyName']} ({realtime_data['symbol']}) - Real-time Data")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_metric("Current Price", f"₹{realtime_data['lastPrice']:.2f}", f"{realtime_data['change']:.2f} ({realtime_data['pChange']:.2f}%)")
    with col2:
        display_metric("Previous Close", f"₹{realtime_data['previousClose']:.2f}")
    with col3:
        display_metric("Open", f"₹{realtime_data['open']:.2f}")
    with col4:
        display_metric("Volume", f"{realtime_data['volume']:,}")
    
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
        st.stop()
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

# Add footer
add_footer()