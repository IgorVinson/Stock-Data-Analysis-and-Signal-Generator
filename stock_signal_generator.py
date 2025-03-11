import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import argparse

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
        print("This may take a moment, please wait...")
        
        try:
            # Download data from Yahoo Finance
            self.data = yf.download(
                self.ticker, 
                start=self.start_date, 
                end=self.end_date,
                progress=True  # Add progress bar
            )
            
            if self.data.empty:
                raise ValueError(f"No data found for ticker {self.ticker} in the specified date range.")
            
            print(f"Successfully downloaded {len(self.data)} days of data.")
            return self.data
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            raise
    
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
            
        print(f"Calculating technical indicators (SMA_{short_window}, SMA_{long_window}, RSI)...")
        
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
        print("Indicators calculated successfully.")
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
            
        print("Generating trading signals...")
        
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
        print("Signals generated successfully.")
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
    
    def plot_signals(self, save_path=None):
        """
        Plot the stock price, moving averages, and buy/sell signals.
        
        Parameters:
        - save_path (str): Path to save the plot. If None, the plot is displayed.
        """
        if self.signals is None:
            self.generate_signals()
            
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the stock price
        ax.plot(self.signals['price'], label='Price', alpha=0.5)
        
        # Plot the moving averages
        short_window = [col for col in self.signals.columns if 'SMA_' in col][0].split('_')[1]
        long_window = [col for col in self.signals.columns if 'SMA_' in col][1].split('_')[1]
        
        ax.plot(self.signals[f'SMA_{short_window}'], label=f'{short_window}-day SMA', alpha=0.8)
        ax.plot(self.signals[f'SMA_{long_window}'], label=f'{long_window}-day SMA', alpha=0.8)
        
        # Plot buy signals
        ax.plot(
            self.signals.loc[self.signals['position'] == 1.0].index,
            self.signals.loc[self.signals['position'] == 1.0]['price'],
            '^', markersize=10, color='g', label='Buy Signal'
        )
        
        # Plot sell signals
        ax.plot(
            self.signals.loc[self.signals['position'] == -1.0].index,
            self.signals.loc[self.signals['position'] == -1.0]['price'],
            'v', markersize=10, color='r', label='Sell Signal'
        )
        
        # Add labels and title
        plt.title(f'{self.ticker} Stock Price and Moving Average Crossover Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()
            
    def plot_returns(self, save_path=None):
        """
        Plot the cumulative returns of the strategy vs. the market.
        
        Parameters:
        - save_path (str): Path to save the plot. If None, the plot is displayed.
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
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            print(f"Returns plot saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()
    
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
        
        market_annual_return = (1 + market_return) ** (1 / years) - 1
        strategy_annual_return = (1 + strategy_return) ** (1 / years) - 1
        
        # Calculate other metrics
        sharpe_ratio = returns['strategy_return'].mean() / returns['strategy_return'].std() * (252 ** 0.5)
        
        # Print performance summary
        print("\n===== PERFORMANCE SUMMARY =====")
        print(f"Ticker: {self.ticker}")
        print(f"Period: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')} ({days} days)")
        print(f"Total Trades: {total_trades} (Buy: {buy_signals}, Sell: {sell_signals})")
        print(f"Cumulative Market Return: {market_return:.2%}")
        print(f"Cumulative Strategy Return: {strategy_return:.2%}")
        print(f"Annualized Market Return: {market_annual_return:.2%}")
        print(f"Annualized Strategy Return: {strategy_annual_return:.2%}")
        print(f"Strategy Sharpe Ratio: {sharpe_ratio:.2f}")
        print("===============================")
        
        return {
            'ticker': self.ticker,
            'period_days': days,
            'total_trades': total_trades,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'market_return': market_return,
            'strategy_return': strategy_return,
            'market_annual_return': market_annual_return,
            'strategy_annual_return': strategy_annual_return,
            'sharpe_ratio': sharpe_ratio
        }


def main():
    """Main function to run the stock signal generator"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Data Analysis and Signal Generator')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL, MSFT)')
    parser.add_argument('--start', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('--short', type=int, default=50, help='Short-term moving average window (default: 50)')
    parser.add_argument('--long', type=int, default=200, help='Long-term moving average window (default: 200)')
    parser.add_argument('--save', type=str, help='Path to save the plots')
    
    args = parser.parse_args()
    
    print(f"\n===== Stock Signal Generator for {args.ticker} =====")
    
    try:
        # Create the signal generator
        generator = StockSignalGenerator(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end
        )
        
        # Fetch data
        generator.fetch_data()
        
        # Generate signals
        generator.generate_signals(args.short, args.long)
        
        print("Generating plots...")
        # Plot signals
        if args.save:
            generator.plot_signals(f"{args.save}_{args.ticker}_signals.png")
            generator.plot_returns(f"{args.save}_{args.ticker}_returns.png")
        else:
            generator.plot_signals()
            generator.plot_returns()
        
        # Print performance summary
        generator.summarize_performance()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting...")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()