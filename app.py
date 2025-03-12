import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive Agg before importing pyplot

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import io
import base64
import json
from matplotlib.dates import DateFormatter



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class StockSignalGenerator:
    def __init__(self, ticker, start_date=None, end_date=None):
        """
        Initialize the StockSignalGenerator with a ticker symbol and date range.
        
        Parameters:
        - ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        - start_date (str): Start date in 'YYYY-MM-DD' format. Default is 2 years ago.
        - end_date (str): End date in 'YYYY-MM-DD' format. Default is today.
        """
        self.ticker = ticker
        
        # Set default dates if not provided
        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date
            
        if start_date is None:
            # Default to 2 years of data
            start = datetime.now() - timedelta(days=2*365)
            self.start_date = start.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
            
        self.data = None
        self.signals = None
    
    def fetch_data(self):
        """Fetch historical stock data using yfinance"""
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")
        
        # Download data from Yahoo Finance
        self.data = yf.download(
            self.ticker, 
            start=self.start_date, 
            end=self.end_date
        )
        
        if self.data.empty:
            raise ValueError(f"No data found for ticker {self.ticker} in the specified date range.")
            
        print(f"Successfully downloaded {len(self.data)} days of data.")
        return self.data
    
    def calculate_indicators(self, short_window=50, long_window=200):
        """
        Calculate technical indicators:
        - Short-term moving average (default: 50-day)
        - Long-term moving average (default: 200-day)
        - RSI (Relative Strength Index)
        
        Parameters:
        - short_window (int): Short-term moving average window
        - long_window (int): Long-term moving average window
        """
        if self.data is None:
            self.fetch_data()
            
        # Make a copy to avoid SettingWithCopyWarning
        df = self.data.copy()
        
        # Calculate moving averages
        df[f'SMA_{short_window}'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
        df[f'SMA_{long_window}'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
        
        # Calculate RSI (14-day period is standard)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        
        # Calculate relative strength (RS)
        rs = gain / loss
        
        # Calculate RSI
        df['RSI'] = 100 - (100 / (1 + rs))
        
        self.data = df
        return df
    
    def generate_signals(self, short_window=50, long_window=200):
        """
        Generate buy/sell signals based on moving average crossover strategy.
        
        Buy signal: Short-term MA crosses above Long-term MA
        Sell signal: Short-term MA crosses below Long-term MA
        
        Parameters:
        - short_window (int): Short-term moving average window
        - long_window (int): Long-term moving average window
        """
        if f'SMA_{short_window}' not in self.data.columns or f'SMA_{long_window}' not in self.data.columns:
            self.calculate_indicators(short_window, long_window)
            
        # Make a copy to avoid SettingWithCopyWarning
        df = self.data.copy()
        
        # Create signals DataFrame
        signals = pd.DataFrame(index=df.index)
        signals['price'] = df['Close']
        signals[f'SMA_{short_window}'] = df[f'SMA_{short_window}']
        signals[f'SMA_{long_window}'] = df[f'SMA_{long_window}']
        
        # Create signal when short MA crosses above long MA
        signals['signal'] = 0.0
        signals['signal'] = np.where(
            signals[f'SMA_{short_window}'] > signals[f'SMA_{long_window}'], 1.0, 0.0
        )
        
        # Generate trading orders
        signals['position'] = signals['signal'].diff()
        
        # 1.0 indicates buy, -1.0 indicates sell
        self.signals = signals
        return signals
    
    def calculate_returns(self):
        """Calculate returns based on the generated signals"""
        if self.signals is None:
            self.generate_signals()
            
        # Create a copy of the signals DataFrame
        returns = self.signals.copy()
        
        # Calculate daily returns
        returns['daily_return'] = self.data['Close'].pct_change()
        
        # Calculate strategy returns (position taken at previous day's close)
        returns['strategy_return'] = returns['daily_return'] * returns['signal'].shift(1)
        
        # Calculate cumulative returns
        returns['cumulative_market_return'] = (1 + returns['daily_return']).cumprod() - 1
        returns['cumulative_strategy_return'] = (1 + returns['strategy_return']).cumprod() - 1
        
        return returns
    
    def plot_signals(self):
        """
        Plot the stock price, moving averages, and buy/sell signals.
        Returns the plot as base64 encoded image.
        """
        if self.signals is None:
            self.generate_signals()
        
        # Get column names for moving averages
        short_window = [col for col in self.signals.columns if 'SMA_' in col][0].split('_')[1]
        long_window = [col for col in self.signals.columns if 'SMA_' in col][1].split('_')[1]
            
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the stock price
        ax.plot(self.signals['price'], label='Price', alpha=0.5)
        
        # Plot the moving averages
        ax.plot(self.signals[f'SMA_{short_window}'], label=f'{short_window}-day SMA', alpha=0.8)
        ax.plot(self.signals[f'SMA_{long_window}'], label=f'{long_window}-day SMA', alpha=0.8)
        
        # Plot buy signals
        buy_signals = self.signals.loc[self.signals['position'] == 1.0]
        if not buy_signals.empty:
            ax.plot(
                buy_signals.index,
                buy_signals['price'],
                '^', markersize=10, color='g', label='Buy Signal'
            )
        
        # Plot sell signals
        sell_signals = self.signals.loc[self.signals['position'] == -1.0]
        if not sell_signals.empty:
            ax.plot(
                sell_signals.index,
                sell_signals['price'],
                'v', markersize=10, color='r', label='Sell Signal'
            )
        
        # Add labels and title
        plt.title(f'{self.ticker} Stock Price and Moving Average Crossover Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Format x-axis dates for better readability
        date_format = DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45)
        
        # Save plot to a BytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        
        # Encode the plot as base64
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    def plot_returns(self):
        """
        Plot the cumulative returns of the strategy vs. the market.
        Returns the plot as base64 encoded image.
        """
        returns = self.calculate_returns()
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot cumulative returns
        ax.plot(returns['cumulative_market_return'], label='Market Returns', alpha=0.7)
        ax.plot(returns['cumulative_strategy_return'], label='Strategy Returns', alpha=0.7)
        
        # Add labels and title
        plt.title(f'{self.ticker} Cumulative Returns: Strategy vs. Market')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Format x-axis dates for better readability
        date_format = DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Save plot to a BytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        
        # Encode the plot as base64
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    def summarize_performance(self):
        """Summarize the performance of the trading strategy"""
        returns = self.calculate_returns()
        
        # Calculate performance metrics
        total_trades = len(returns[returns['position'] != 0])
        buy_signals = len(returns[returns['position'] == 1.0])
        sell_signals = len(returns[returns['position'] == -1.0])
        
        # Calculate annualized returns
        days = (returns.index[-1] - returns.index[0]).days
        years = days / 365.25
        
        market_return = returns['cumulative_market_return'].iloc[-1]
        strategy_return = returns['cumulative_strategy_return'].iloc[-1]
        
        market_annual_return = (1 + market_return) ** (1 / years) - 1 if years > 0 else 0
        strategy_annual_return = (1 + strategy_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate other metrics
        sharpe_ratio = returns['strategy_return'].mean() / returns['strategy_return'].std() * (252 ** 0.5) if returns['strategy_return'].std() > 0 else 0
        
        summary = {
            'ticker': self.ticker,
            'period_start': returns.index[0].strftime('%Y-%m-%d'),
            'period_end': returns.index[-1].strftime('%Y-%m-%d'),
            'period_days': days,
            'total_trades': total_trades,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'market_return': round(market_return * 100, 2),
            'strategy_return': round(strategy_return * 100, 2),
            'market_annual_return': round(market_annual_return * 100, 2),
            'strategy_annual_return': round(strategy_annual_return * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 2)
        }
        
        return summary

@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    try:
        # Parse request data
        data = request.json
        ticker = data.get('ticker', '').upper()
        start_date = data.get('startDate')
        end_date = data.get('endDate')
        short_window = int(data.get('shortWindow', 50))
        long_window = int(data.get('longWindow', 200))
        
        # Validate inputs
        if not ticker:
            return jsonify({'error': 'Ticker symbol is required'}), 400
            
        if short_window >= long_window:
            return jsonify({'error': 'Short-term window must be smaller than long-term window'}), 400
            
        # Create signal generator
        generator = StockSignalGenerator(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        try:
            # Fetch data and generate signals
            generator.fetch_data()
            generator.generate_signals(short_window, long_window)
        except ValueError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            return jsonify({'error': f"Error processing stock data: {str(e)}"}), 500
        
        # Generate plots (keep for backward compatibility)
        signals_plot = generator.plot_signals()
        returns_plot = generator.plot_returns()
        
        # Get performance summary
        summary = generator.summarize_performance()
        
        # Extract data for interactive charts
        returns = generator.calculate_returns()
        
        # Convert dates to string format for JSON serialization
        dates = [date.strftime('%Y-%m-%d') for date in returns.index]
        
        # Get buy and sell signals
        buy_signals = generator.signals.loc[generator.signals['position'] == 1.0]
        sell_signals = generator.signals.loc[generator.signals['position'] == -1.0]
        
        # Prepare data for frontend
        chart_data = {
            'dates': dates,
            'prices': returns['price'].tolist(),
            'short_ma': returns[f'SMA_{short_window}'].tolist(),
            'long_ma': returns[f'SMA_{long_window}'].tolist(),
            'buy_signals': {
                'dates': [date.strftime('%Y-%m-%d') for date in buy_signals.index],
                'prices': buy_signals['price'].tolist()
            },
            'sell_signals': {
                'dates': [date.strftime('%Y-%m-%d') for date in sell_signals.index],
                'prices': sell_signals['price'].tolist()
            },
            'market_returns': returns['cumulative_market_return'].tolist(),
            'strategy_returns': returns['cumulative_strategy_return'].tolist()
        }
        
        # Return the results
        return jsonify({
            'success': True,
            'summary': summary,
            'signals_plot': signals_plot,  # Keep for backward compatibility
            'returns_plot': returns_plot,  # Keep for backward compatibility
            **chart_data  # Include all chart data
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stocks', methods=['GET'])
def get_popular_stocks():
    # Expanded list of popular stocks to match autocomplete functionality in React
    popular_stocks = [
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corporation"},
        {"symbol": "GOOGL", "name": "Alphabet Inc. (Google)"},
        {"symbol": "AMZN", "name": "Amazon.com Inc."},
        {"symbol": "TSLA", "name": "Tesla, Inc."},
        {"symbol": "META", "name": "Meta Platforms, Inc. (Facebook)"},
        {"symbol": "NVDA", "name": "NVIDIA Corporation"},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
        {"symbol": "V", "name": "Visa Inc."},
        {"symbol": "JNJ", "name": "Johnson & Johnson"},
        {"symbol": "WMT", "name": "Walmart Inc."},
        {"symbol": "PG", "name": "Procter & Gamble Co."},
        {"symbol": "MA", "name": "Mastercard Inc."},
        {"symbol": "UNH", "name": "UnitedHealth Group Inc."},
        {"symbol": "HD", "name": "Home Depot Inc."},
        {"symbol": "DIS", "name": "Walt Disney Co."},
        {"symbol": "BAC", "name": "Bank of America Corp."},
        {"symbol": "INTC", "name": "Intel Corporation"},
        {"symbol": "VZ", "name": "Verizon Communications Inc."},
        {"symbol": "CSCO", "name": "Cisco Systems Inc."},
        {"symbol": "ADBE", "name": "Adobe Inc."},
        {"symbol": "NFLX", "name": "Netflix Inc."},
        {"symbol": "CRM", "name": "Salesforce Inc."},
        {"symbol": "KO", "name": "Coca-Cola Co."},
        {"symbol": "PEP", "name": "PepsiCo Inc."}
    ]
    
    return jsonify(popular_stocks)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/api/stock-info/<ticker>', methods=['GET'])
def get_stock_info(ticker):
    """Get basic information about a stock"""
    try:
        ticker = ticker.upper()
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant information
        basic_info = {
            'symbol': ticker,
            'name': info.get('shortName', 'Unknown'),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'marketCap': info.get('marketCap', 0),
            'currentPrice': info.get('currentPrice', 0),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0),
            'averageVolume': info.get('averageVolume', 0),
            'dividendYield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'beta': info.get('beta', 0)
        }
        
        return jsonify(basic_info)
    
    except Exception as e:
        return jsonify({'error': f"Could not fetch stock info: {str(e)}"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)