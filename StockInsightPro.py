import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
import ta
from GoogleNews import GoogleNews
from textblob import TextBlob

# Helper functions
def search_indian_stocks(query):
    try:
        indian_stocks = pd.read_csv('https://archives.nseindia.com/content/equities/EQUITY_L.csv')
        return indian_stocks[indian_stocks['SYMBOL'].str.contains(query, case=False) | 
                             indian_stocks['NAME OF COMPANY'].str.contains(query, case=False)]
    except Exception as e:
        st.error(f"Error fetching Indian stocks: {str(e)}")
        return pd.DataFrame()

def get_realtime_data(symbol):
    if not symbol:
        return None
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
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
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_stock_data(symbol, start_date, end_date):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching historical data for {symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_technical_indicators(data):
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['MACD'] = ta.trend.macd(data['Close'])
    data['Signal'] = ta.trend.macd_signal(data['Close'])
    bollinger = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['BB_upper'] = bollinger.bollinger_hband()
    data['BB_middle'] = bollinger.bollinger_mavg()
    data['BB_lower'] = bollinger.bollinger_lband()
    data['Stochastic'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=14)
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=14)
    data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
    data['VWAP'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'], data['Close'], data['Volume'])
    data['Parabolic_SAR'] = ta.trend.psar_down(data['High'], data['Low'], data['Close'])
    data['Chaikin_MF'] = ta.volume.chaikin_money_flow(data['High'], data['Low'], data['Close'], data['Volume'], window=20)
    return data

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
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
    
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
    model = ARIMA(data['Close'], order=(5, 1, 0))
    results = model.fit()
    forecast = results.forecast(steps=future_days)
    return forecast

def random_forest_prediction(data, future_days=30):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    X = data['Close'].values.reshape(-1, 1)
    y = data['Close'].shift(-future_days).dropna().values
    model.fit(X[:-future_days], y)
    predictions = model.predict(X[-future_days:])
    return predictions

def plot_stock_analysis(data, stock_symbol, indicators):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('Stock Price with Indicators', 'Volume'),
                        row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'], name='Candlestick'), row=1, col=1)
    
    for indicator in indicators:
        if indicator in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[indicator], mode='lines', name=indicator), row=1, col=1)
    
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=2, col=1)
    
    fig.update_layout(
        height=800,
        template="plotly_white",
        title_text=f"Stock Analysis for {stock_symbol}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False
    )
    return fig

def plot_advanced_analysis(data, stock_symbol, indicators):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('Stock Price with Indicators and Predictions', 'Volume'),
                        row_heights=[0.7, 0.3])
    
    # Add Candlestick chart for historical data
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'], name='Candlestick'), row=1, col=1)
    
    # Add technical indicators
    for indicator in indicators:
        if indicator in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[indicator], mode='lines', name=indicator), row=1, col=1)
    
    # Add Volume chart
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=2, col=1)
    
    # Get predictions
    lstm_pred = lstm_prediction(data)
    arima_pred = arima_prediction(data)
    rf_pred = random_forest_prediction(data)
    
    # Add predictions to the first subplot
    if lstm_pred is not None:
        fig.add_trace(go.Scatter(x=pd.date_range(start=data.index[-1], periods=31)[1:], y=lstm_pred, name='LSTM Prediction', line=dict(color='red')), row=1, col=1)
    
    if arima_pred is not None:
        fig.add_trace(go.Scatter(x=pd.date_range(start=data.index[-1], periods=31)[1:], y=arima_pred, name='ARIMA Prediction', line=dict(color='green')), row=1, col=1)
    
    if rf_pred is not None:
        fig.add_trace(go.Scatter(x=pd.date_range(start=data.index[-1], periods=31)[1:], y=rf_pred, name='Random Forest Prediction', line=dict(color='blue')), row=1, col=1)
    
    fig.update_layout(
        height=1000,
        template="plotly_white",
        title_text=f"Advanced Stock Analysis for {stock_symbol}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False
    )
    return fig

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

# Streamlit app
def set_custom_theme():
    st.set_page_config(
        page_title="MarketMind",
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
    .metric-card {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #000000;
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

def add_header():
    st.markdown("""
    <div style="background-color:#2c3e50;padding:1rem;border-radius:0.5rem;margin-bottom:1rem;">
    <h1 style="color:white;text-align:center;font-size:2rem;margin:0;">📈 Market Mind</h1>
    <h3 style="color:#3498db;text-align:center;font-size:1rem;margin:0.5rem 0 0 0;">Advanced Stock Market Analysis Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)

def add_footer():
    st.markdown("""
    <div style="background-color:#2c3e50;padding:0.5rem;border-radius:0.5rem;text-align:center;font-size:0.8rem;margin-top:1rem;">
    <p style="color:white;margin:0;">© 2024 StockInsight Pro. All rights reserved.</p>
    <p style="color:#3498db;margin:0;">Disclaimer: This app is for educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

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

def plot_portfolio_composition(portfolio_data):
    fig = go.Figure()
    for symbol, data in portfolio_data.items():
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name=symbol))
    
    fig.update_layout(
        title="Portfolio Composition",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Symbols",
        template="plotly_white"
    )
    return fig

def plot_portfolio_performance(total_return, annualized_return, volatility, total_return_value, total_initial_value, total_current_value):
    fig = go.Figure(data=[
        go.Bar(name='Total Return', x=['Total Return'], y=[total_return.mean()]),
        go.Bar(name='Annualized Return', x=['Annualized Return'], y=[annualized_return.mean()]),
        go.Bar(name='Volatility', x=['Volatility'], y=[volatility.mean()]),
        go.Bar(name='Total Return Value', x=['Total Return Value'], y=[total_return_value]),
        go.Bar(name='Total Initial Value', x=['Total Initial Value'], y=[total_initial_value]),
        go.Bar(name='Total Current Value', x=['Total Current Value'], y=[total_current_value])
    ])
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Metrics",
        yaxis_title="Values",
        barmode='group',
        template="plotly_white"
    )
    return fig

def get_portfolio_data(portfolio, start_date, end_date):
    portfolio_data = {}
    for symbol in portfolio:
        data = get_stock_data(symbol, start_date, end_date)
        if not data.empty:
            portfolio_data[symbol] = data
    return portfolio_data

def calculate_portfolio_performance(portfolio_data, portfolio_info):
    portfolio_returns = {}
    portfolio_values = {}
    
    for symbol, data in portfolio_data.items():
        data['Return'] = data['Close'].pct_change()
        portfolio_returns[symbol] = data['Return']
        
        avg_price = portfolio_info[symbol]['avg_price']
        num_shares = portfolio_info[symbol]['num_shares']
        initial_value = avg_price * num_shares
        current_value = data['Close'][-1] * num_shares
        portfolio_values[symbol] = {'initial_value': initial_value, 'current_value': current_value}
    
    portfolio_returns = pd.DataFrame(portfolio_returns)
    portfolio_returns = portfolio_returns.dropna()
    
    total_return = (1 + portfolio_returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    
    # Calculate portfolio value metrics
    total_initial_value = sum(info['initial_value'] for info in portfolio_values.values())
    total_current_value = sum(info['current_value'] for info in portfolio_values.values())
    total_return_value = (total_current_value - total_initial_value) / total_initial_value
    
    return total_return, annualized_return, volatility, total_return_value, total_initial_value, total_current_value

# Main layout
set_custom_theme()
add_header()

# Sidebar
with st.sidebar:
    st.title("Stock Analyzer")
    
    search_query = st.text_input("Search for a stock (e.g., RELIANCE, TCS)")
    stock_symbol = None
    if search_query:
        search_results = search_indian_stocks(search_query)
        if not search_results.empty:
            selected_stock = st.selectbox("Select a stock", 
                                          options=search_results['SYMBOL'].tolist(),
                                          format_func=lambda x: f"{x} - {search_results[search_results['SYMBOL'] == x]['NAME OF COMPANY'].values[0]}")
            stock_symbol = f"{selected_stock}.NS"  # Append .NS for NSE stocks
        else:
            st.warning("No matching stocks found. Please try a different search term.")
    
    current_date = datetime.now()
    one_year_ago = current_date - timedelta(days=365)
    start_date = st.date_input("Start Date", one_year_ago)
    end_date = st.date_input("End Date", current_date)
    
    sma_window = st.sidebar.slider("SMA Window", min_value=5, max_value=50, value=20)
    
    indicators = st.multiselect(
        "Choose indicators to plot:",
        ['SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'Bollinger Bands', 'Stochastic', 'ADX', 'OBV', 'VWAP', 'Parabolic SAR', 'Chaikin Money Flow'],
        default=['SMA_20', 'SMA_50', 'EMA_20']
    )
    advanced_analysis = st.button("Run Advanced Analysis")

    # Portfolio Management
    st.subheader("Portfolio Management")
    portfolio = st.text_area("Add stocks to your portfolio (comma-separated symbols)", value="")
    portfolio = [symbol.strip() for symbol in portfolio.split(",") if symbol.strip()]
    
    portfolio_info = {}
    for symbol in portfolio:
        col1, col2 = st.columns(2)
        with col1:
            avg_price = st.number_input(f"Average Price for {symbol}", min_value=0.0, value=0.0, key=f"{symbol}_avg_price")
        with col2:
            num_shares = st.number_input(f"Number of Shares for {symbol}", min_value=0, value=0, key=f"{symbol}_num_shares")
        portfolio_info[symbol] = {'avg_price': avg_price, 'num_shares': num_shares}
    
    analyze_portfolio = st.button("Analyze Portfolio")

# Main content
if stock_symbol:
    @st.cache_data(ttl=2)
    def get_realtime_data_cached(symbol):
        return get_realtime_data(symbol)
    
    realtime_data = get_realtime_data_cached(stock_symbol)
    data = get_stock_data(stock_symbol, start_date, end_date)
    
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
    
    if data.empty:
        st.error(f"No historical data available for {stock_symbol} between {start_date} and {end_date}. Please try a different date range or stock symbol.")
    else:
        data = calculate_technical_indicators(data)
        
        st.subheader(f"{stock_symbol} Historical Analysis")
        fig = plot_stock_analysis(data, stock_symbol, indicators)
        st.plotly_chart(fig, use_container_width=True)
    
    if advanced_analysis:
        st.subheader("Advanced Analysis with Predictions")
        advanced_fig = plot_advanced_analysis(data, stock_symbol, indicators)
        st.plotly_chart(advanced_fig, use_container_width=True)
    
    st.subheader(f"{stock_symbol} News Sentiment Analysis")
    news_df, overall_sentiment = get_news_sentiment(stock_symbol)
    if news_df is not None:
        st.write(f"**Overall Market Sentiment:** {overall_sentiment}")
        
        sentiment_counts = news_df['sentiment'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hole=.3)])
        fig.update_layout(title_text="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Recent News Articles")
        for _, row in news_df.iterrows():
            st.markdown(f"**{row['title']}** - {row['date']}")
            st.write(row['desc'])
            st.markdown(f"Sentiment: {row['sentiment']} | [Read more]({row['link']})")
            st.markdown("---")
    else:
        st.warning(f"No recent news found for {stock_symbol}")

# Portfolio Analysis
if analyze_portfolio:
    if not portfolio:
        st.warning("Please add stocks to your portfolio.")
    else:
        portfolio_data = get_portfolio_data(portfolio, start_date, end_date)
        if not portfolio_data:
            st.warning("No data available for the stocks in your portfolio.")
        else:
            st.subheader("Portfolio Analysis")
            total_return, annualized_return, volatility, total_return_value, total_initial_value, total_current_value = calculate_portfolio_performance(portfolio_data, portfolio_info)
            st.write(f"**Total Return:** {total_return.mean():.2%}")
            st.write(f"**Annualized Return:** {annualized_return.mean():.2%}")
            st.write(f"**Volatility:** {volatility.mean():.2%}")
            st.write(f"**Total Return Value:** {total_return_value:.2%}")
            st.write(f"**Total Initial Value:** ₹{total_initial_value:,.2f}")
            st.write(f"**Total Current Value:** ₹{total_current_value:,.2f}")
            
            composition_fig = plot_portfolio_composition(portfolio_data)
            st.plotly_chart(composition_fig, use_container_width=True)
            
            performance_fig = plot_portfolio_performance(total_return, annualized_return, volatility, total_return_value, total_initial_value, total_current_value)
            st.plotly_chart(performance_fig, use_container_width=True)

else:
    st.info("Please search and select a stock to begin analysis.")

# Add footer
add_footer()
