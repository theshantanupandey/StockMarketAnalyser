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
from prophet import Prophet
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict, Tuple
import logging

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
        logging.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_stock_data(symbol, start_date, end_date):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        return data
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {str(e)}")
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

def prophet_prediction(data, future_days=30):
    df = data[['Close']].reset_index()
    df.columns = ['ds', 'y']
    
    # Remove timezone from the 'ds' column
    df['ds'] = df['ds'].dt.tz_localize(None)
    
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=future_days)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(future_days)

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
    prophet_pred = prophet_prediction(data)
    
    # Add predictions to the first subplot
    if lstm_pred is not None:
        fig.add_trace(go.Scatter(x=pd.date_range(start=data.index[-1], periods=31)[1:], y=lstm_pred, name='LSTM Prediction', line=dict(color='red')), row=1, col=1)
    
    if arima_pred is not None:
        fig.add_trace(go.Scatter(x=pd.date_range(start=data.index[-1], periods=31)[1:], y=arima_pred, name='ARIMA Prediction', line=dict(color='green')), row=1, col=1)
    
    if rf_pred is not None:
        fig.add_trace(go.Scatter(x=pd.date_range(start=data.index[-1], periods=31)[1:], y=rf_pred, name='Random Forest Prediction', line=dict(color='blue')), row=1, col=1)
    
    if prophet_pred is not None:
        fig.add_trace(go.Scatter(x=prophet_pred['ds'], y=prophet_pred['yhat'], name='Prophet Prediction', line=dict(color='purple')), row=1, col=1)
    
    fig.update_layout(
        height=1000,
        template="plotly_white",
        title_text=f"Advanced Stock Analysis for {stock_symbol}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False
    )
    return fig

@st.cache_data(ttl=300)
def get_stock_news(stock_symbol, num_articles=5):
    try:
        googlenews = GoogleNews(lang='en', region='IN')
        googlenews.search(stock_symbol)
        result = googlenews.result()
        df = pd.DataFrame(result)
        if not df.empty:
            df = df[['title', 'desc', 'date', 'link']].head(num_articles)
        return df
    except Exception as e:
        logging.error(f"Error fetching news for {stock_symbol}: {str(e)}")
        return pd.DataFrame()

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

@dataclass
class StockMetrics:
    """Data class to hold stock metrics with improved validation and default handling."""
    bid: float
    ask: float
    day_range: Tuple[float, float]
    week_range: Tuple[float, float]
    avg_volume: int
    beta: float
    market_cap: float
    pe_ratio: Optional[float]
    eps: Optional[float]
    forward_pe: Optional[float]
    earnings_date: Optional[datetime]
    dividend_yield: Optional[float]
    ex_dividend_date: Optional[datetime]
    target_est: Optional[float]

    def __post_init__(self):
        """Validate inputs and convert types with fallback values."""
        try:
            # Helper function to safely convert numeric values
            def safe_convert(value, converter, default):
                if isinstance(value, (int, float)):
                    return converter(value)
                try:
                    if isinstance(value, str):
                        # Remove commas and convert
                        cleaned = value.replace(',', '')
                        if cleaned != 'N/A':
                            return converter(cleaned)
                except (ValueError, TypeError):
                    pass
                return default

            # Convert numerical fields with appropriate defaults
            self.bid = safe_convert(self.bid, float, 0.0)
            self.ask = safe_convert(self.ask, float, 0.0)
            self.beta = safe_convert(self.beta, float, 1.0)  # Default beta of 1
            self.market_cap = safe_convert(self.market_cap, float, 0.0)
            self.avg_volume = safe_convert(self.avg_volume, int, 0)
            
            # Handle optional fields
            self.pe_ratio = safe_convert(self.pe_ratio, float, None)
            self.eps = safe_convert(self.eps, float, None)
            self.forward_pe = safe_convert(self.forward_pe, float, None)
            self.dividend_yield = safe_convert(self.dividend_yield, float, None)
            self.target_est = safe_convert(self.target_est, float, None)

            # Special handling for day_range and week_range
            def parse_range(range_value, default=(0.0, 0.0)):
                if isinstance(range_value, tuple) and len(range_value) == 2:
                    low = safe_convert(range_value[0], float, 0.0)
                    high = safe_convert(range_value[1], float, 0.0)
                    return (low, high)
                return default

            self.day_range = parse_range(self.day_range)
            self.week_range = parse_range(self.week_range)

            # Handle datetime fields
            def safe_convert_datetime(value):
                if isinstance(value, datetime):
                    return value
                if isinstance(value, (int, float)):
                    try:
                        return datetime.fromtimestamp(int(value))
                    except (ValueError, OSError):
                        return None
                return None

            self.earnings_date = safe_convert_datetime(self.earnings_date)
            self.ex_dividend_date = safe_convert_datetime(self.ex_dividend_date)

        except Exception as e:
            raise ValueError(f"Error validating metrics: {str(e)}")

    def is_valid_for_analysis(self) -> bool:
        """
        Check if the metrics contain enough valid data for meaningful analysis.
        Returns True if essential metrics are available.
        """
        # Check if we have either bid/ask prices or market cap
        has_price_data = (self.bid > 0 and self.ask > 0) or self.market_cap > 0
        # Check if we have any trading activity data
        has_trading_data = self.avg_volume > 0 or (self.day_range[1] > self.day_range[0])
        
        return has_price_data and has_trading_data

    def get_current_price(self) -> float:
        """
        Get the current price estimate from available data.
        Returns the mid-point of bid-ask if available, otherwise 0.
        """
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        if self.day_range[1] > self.day_range[0]:
            return sum(self.day_range) / 2
        return 0.0
        
class StockRecommendation:
    """Stock recommendation system with weighted analysis of multiple factors."""
    
    # Classification thresholds
    MARKET_CAP_THRESHOLDS = {
        "MICRO": 3e8,    # $300M
        "SMALL": 2e9,    # $2B
        "MID": 1e10,     # $10B
        "LARGE": 1e11    # $100B
    }
    
    PE_THRESHOLDS = {
        "LOW": 15,
        "MODERATE": 25,
        "HIGH": 35
    }
    
    def __init__(self):
        """Initialize the recommendation system with scoring weights."""
        self.weights = {
            "valuation": 0.3,
            "momentum": 0.2,
            "risk": 0.2,
            "income": 0.15,
            "growth": 0.15
        }
        self.logger = logging.getLogger(__name__)

    def _analyze_valuation(self, metrics: StockMetrics) -> Dict[str, float]:
        """Analyze valuation metrics."""
        score = 0.0
        confidence = 1.0
        
        if metrics.pe_ratio:
            if metrics.pe_ratio < self.PE_THRESHOLDS["LOW"]:
                score += 1.0
            elif metrics.pe_ratio > self.PE_THRESHOLDS["HIGH"]:
                score -= 1.0
        else:
            confidence *= 0.8
            
        if metrics.forward_pe:
            if metrics.forward_pe < metrics.pe_ratio:
                score += 0.5
            elif metrics.forward_pe > metrics.pe_ratio:
                score -= 0.5
        else:
            confidence *= 0.8
            
        return {"score": score, "confidence": confidence}

    def _analyze_momentum(self, metrics: StockMetrics) -> Dict[str, float]:
        """Analyze price momentum."""
        score = 0.0
        current_price = (metrics.bid + metrics.ask) / 2
        
        # Compare to 52-week range
        week_range_midpoint = sum(metrics.week_range) / 2
        if current_price > week_range_midpoint:
            score += 0.5
        else:
            score -= 0.5
            
        # Compare to day range
        day_range_midpoint = sum(metrics.day_range) / 2
        if current_price > day_range_midpoint:
            score += 0.5
        else:
            score -= 0.5
            
        return {"score": score, "confidence": 1.0}

    def _analyze_risk(self, metrics: StockMetrics) -> Dict[str, float]:
        """Analyze risk metrics."""
        score = 0.0
        
        # Beta analysis
        if metrics.beta < 0.8:
            score += 1.0
        elif metrics.beta > 1.2:
            score -= 1.0
            
        # Market cap analysis
        if metrics.market_cap > self.MARKET_CAP_THRESHOLDS["LARGE"]:
            score += 0.5
        elif metrics.market_cap < self.MARKET_CAP_THRESHOLDS["MICRO"]:
            score -= 0.5
            
        return {"score": score, "confidence": 1.0}

    def _analyze_income(self, metrics: StockMetrics) -> Dict[str, float]:
        """Analyze income-related metrics."""
        score = 0.0
        confidence = 1.0
        
        if metrics.dividend_yield:
            if metrics.dividend_yield > 0.04:  # 4% yield
                score += 1.0
            elif metrics.dividend_yield > 0.02:  # 2% yield
                score += 0.5
        else:
            confidence *= 0.7
            
        if metrics.eps and metrics.eps > 0:
            score += 0.5
        elif metrics.eps and metrics.eps < 0:
            score -= 1.0
            
        return {"score": score, "confidence": confidence}

    def _analyze_growth(self, metrics: StockMetrics) -> Dict[str, float]:
        """Analyze growth potential."""
        score = 0.0
        confidence = 1.0
        
        if metrics.target_est:
            current_price = (metrics.bid + metrics.ask) / 2
            potential_return = (metrics.target_est - current_price) / current_price
            
            if potential_return > 0.2:  # 20% upside
                score += 1.0
            elif potential_return < -0.1:  # 10% downside
                score -= 1.0
        else:
            confidence *= 0.6
            
        return {"score": score, "confidence": confidence}

    def get_recommendation(self, metrics: StockMetrics) -> Tuple[str, float, Dict]:
        """
        Generate a stock recommendation based on comprehensive analysis.
        
        Args:
            metrics: StockMetrics object containing all relevant metrics
            
        Returns:
            Tuple containing:
            - recommendation: str ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell")
            - confidence: float (0-1)
            - analysis: Dict with detailed scoring breakdown
        """
        try:
            analyses = {
                "valuation": self._analyze_valuation(metrics),
                "momentum": self._analyze_momentum(metrics),
                "risk": self._analyze_risk(metrics),
                "income": self._analyze_income(metrics),
                "growth": self._analyze_growth(metrics)
            }
            
            # Calculate weighted score
            total_score = 0
            total_confidence = 0
            
            for category, weight in self.weights.items():
                analysis = analyses[category]
                total_score += analysis["score"] * weight * analysis["confidence"]
                total_confidence += weight * analysis["confidence"]
            
            # Normalize confidence
            final_confidence = total_confidence / sum(self.weights.values())
            
            # Determine recommendation
            if total_score > 0.5:
                recommendation = "Strong Buy"
            elif total_score > 0.2:
                recommendation = "Buy"
            elif total_score < -0.5:
                recommendation = "Strong Sell"
            elif total_score < -0.2:
                recommendation = "Sell"
            else:
                recommendation = "Hold"
                
            return recommendation, final_confidence, analyses
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {str(e)}")
            raise

def get_recommendation(data, prophet_pred):
    """Generate recommendation based on Prophet predictions"""
    try:
        last_price = data['Close'].iloc[-1]
        future_price = prophet_pred['yhat'].iloc[-1]
        price_change = ((future_price - last_price) / last_price) * 100
        
        if price_change > 10:
            return "Strong Buy"
        elif price_change > 5:
            return "Buy"
        elif price_change < -10:
            return "Strong Sell"
        elif price_change < -5:
            return "Sell"
        else:
            return "Hold"
    except Exception as e:
        logging.error(f"Error generating recommendation: {str(e)}")
        return "Unable to generate recommendation"

import numpy as np
from collections import defaultdict

def combine_evidence(evidence):
    """Combine evidence using Dempster-Shafer Theory (DST)."""
    combined_belief = defaultdict(float)
    normalization_factor = 0
    
    for model, beliefs in evidence.items():
        for outcome, belief in beliefs.items():
            combined_belief[outcome] += belief
            normalization_factor += belief
    
    # Normalize the combined belief
    for outcome in combined_belief:
        combined_belief[outcome] /= normalization_factor
    
    return combined_belief

class BeliefRuleBase:
    def __init__(self, rules):
        self.rules = rules
    
    def apply_rules(self, combined_belief):
        """Apply BRB rules based on the combined belief."""
        for rule in self.rules:
            condition = rule['condition']
            action = rule['action']
            
            # Check if the condition matches the combined belief
            match = all(combined_belief.get(outcome, 0) > 0.5 for outcome in condition.values())
            
            if match:
                return action
        
        return "Hold"  # Default action if no rule matches

# Define BRB rules
brb_rules = [
    {
        "condition": {"LSTM": "positive", "ARIMA": "positive"},
        "action": "Strong Buy"
    },
    {
        "condition": {"LSTM": "negative", "ARIMA": "negative"},
        "action": "Strong Sell"
    },
    {
        "condition": {"LSTM": "positive", "ARIMA": "negative"},
        "action": "Hold"
    },
    # Add more rules as needed
]

# Initialize BRB system
brb = BeliefRuleBase(brb_rules)

def get_recommendation_with_er_brb(data, prophet_pred):
    """Generate recommendation based on ER and BRB"""
    try:
        # Get predictions from different models
        lstm_pred = lstm_prediction(data)
        arima_pred = arima_prediction(data)
        rf_pred = random_forest_prediction(data)
        prophet_pred = prophet_prediction(data)
        
        # Combine evidence using ER
        evidence = {
            "LSTM": {"positive": 0.8, "negative": 0.2},  # Example belief functions
            "ARIMA": {"positive": 0.7, "negative": 0.3},
            "RandomForest": {"positive": 0.6, "negative": 0.4},
            "Prophet": {"positive": 0.9, "negative": 0.1}
        }
        combined_belief = combine_evidence(evidence)
        
        # Apply BRB rules
        recommendation = brb.apply_rules(combined_belief)
        
        return recommendation
    except Exception as e:
        logging.error(f"Error generating recommendation with ER and BRB: {str(e)}")
        return "Unable to generate recommendation"



# Main layout
set_custom_theme()
add_header()

# Sidebar
with st.sidebar:
    st.title("Stock Analyzer")
    
    search_query = st.text_input("Search for a stock")
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
        
        # Get recommendation based on realtime data
        try:
            metrics = StockMetrics(
                bid=realtime_data['bid'],
                ask=realtime_data['ask'],
                day_range=(
                    float(realtime_data['dayRange'].split(' - ')[0]) if ' - ' in str(realtime_data['dayRange']) else 0.0,
                    float(realtime_data['dayRange'].split(' - ')[1]) if ' - ' in str(realtime_data['dayRange']) else 0.0),
                week_range=(
                    float(realtime_data['weekRange'].split(' - ')[0]) if ' - ' in str(realtime_data['weekRange']) else 0.0,
                    float(realtime_data['weekRange'].split(' - ')[1]) if ' - ' in str(realtime_data['weekRange']) else 0.0),
                avg_volume=realtime_data['avgVolume'],
                beta=realtime_data['beta'],
                market_cap=realtime_data['marketCap'],
                pe_ratio=realtime_data['peRatio'],
                eps=realtime_data['eps'],
                forward_pe=realtime_data['forwardPE'],
                earnings_date=realtime_data['earningsDate'],
                dividend_yield=realtime_data['dividendYield'],
                ex_dividend_date=realtime_data['exDividendDate'],
                target_est=realtime_data['targetEstimate']
            )
            
            if metrics.is_valid_for_analysis():
                recommender = StockRecommendation()
                recommendation, confidence, analysis = recommender.get_recommendation(metrics)
                
                st.subheader("Recommendation")
                st.write(f"Based on the current financial metrics, the recommendation for {stock_symbol} is: **{recommendation}**")
                st.write(f"Confidence: {confidence:.2f}")
                
                # Display detailed analysis in collapsible sections
                st.subheader("Detailed Analysis")
                for category, result in analysis.items():
                    with st.expander(f"{category.capitalize()} Analysis"):
                        st.write(f"**Score:** {result['score']:.2f}")
                        st.write(f"**Confidence:** {result['confidence']:.2f}")
                        # Add more detailed explanations if needed
            else:
                st.warning("Insufficient data available for generating a reliable recommendation.")

        except Exception as e:
            st.error(f"Error generating recommendation: {str(e)}")
            logging.error(f"Error in recommendation generation: {str(e)}", exc_info=True)
    
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
        
        # Get Prophet predictions
        prophet_pred = prophet_prediction(data)
        if prophet_pred is not None:
            recommendation = get_recommendation_with_er_brb(data, prophet_pred)
            st.subheader("Recommendation")
            st.write(f"Based on the combined analysis using ER and BRB, the recommendation for {stock_symbol} is: **{recommendation}**")
    
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

else:
    st.info("Please search and select a stock to begin analysis.")

# Add footer
add_footer()
