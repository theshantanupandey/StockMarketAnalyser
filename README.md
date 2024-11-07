# StockMarketAnalyser
# MarketMind - Advanced Stock Market Analysis Dashboard

## Overview

**MarketMind** is an advanced stock market analysis dashboard built with Python and Streamlit. It provides real-time stock data, historical analysis, technical indicators, and predictive models to help users make informed investment decisions. The dashboard integrates Evidential Reasoning (ER) and Belief Rule-Based (BRB) systems to provide robust and reliable stock recommendations.

## Features

- **Real-time Stock Data**: Fetch and display real-time stock data including current price, bid-ask spread, volume, and more.
- **Historical Analysis**: Analyze historical stock data with customizable date ranges.
- **Technical Indicators**: Calculate and plot various technical indicators such as SMA, EMA, RSI, MACD, Bollinger Bands, and more.
- **Predictive Models**: Utilize LSTM, ARIMA, Random Forest, and Prophet models for stock price predictions.
- **News Sentiment Analysis**: Analyze sentiment from recent news articles to gauge market sentiment.
- **Advanced Analysis**: Combine multiple predictive models and technical indicators for a comprehensive analysis.
- **Evidential Reasoning (ER)**: Combine evidence from different models using Dempster-Shafer Theory (DST).
- **Belief Rule-Based (BRB)**: Apply predefined rules to make final stock recommendations based on combined evidence.

## Installation

To run **MarketMind** locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MarketMind.git
   cd MarketMind
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Search for a Stock**: Enter the stock symbol in the search bar (e.g., RELIANCE, TCS).
2. **Select a Stock**: Choose a stock from the dropdown list.
3. **Set Date Range**: Select the start and end dates for historical data analysis.
4. **Choose Indicators**: Select the technical indicators you want to plot.
5. **Run Advanced Analysis**: Click the "Run Advanced Analysis" button to get a comprehensive analysis and stock recommendation.

## Code Structure

- **app.py**: The main Streamlit application file.
- **helpers.py**: Contains helper functions for fetching stock data, calculating technical indicators, and generating predictions.
- **evidential_reasoning.py**: Implements the Evidential Reasoning (ER) system.
- **belief_rule_base.py**: Implements the Belief Rule-Based (BRB) system.
- **requirements.txt**: Lists all the required Python packages.

## Dependencies

- **Streamlit**: For building the web application.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Plotly**: For interactive data visualization.
- **yfinance**: For fetching real-time and historical stock data.
- **scikit-learn**: For machine learning models.
- **Keras**: For LSTM model implementation.
- **statsmodels**: For ARIMA model implementation.
- **Prophet**: For time series forecasting.
- **TextBlob**: For sentiment analysis.
- **GoogleNews**: For fetching recent news articles.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

**MarketMind** is for educational purposes only. It is not intended to provide financial advice. Always do your own research and consult with a financial advisor before making investment decisions.

## Contact

For any questions or feedback, please contact [Your Name](mailto:your.email@example.com).

---

© 2024 StockInsight Pro. All rights reserved.
