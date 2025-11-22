"""
Data loading and preprocessing module for portfolio optimization.

This module handles:
- Downloading financial data from Yahoo Finance
- Computing technical indicators (SMA, EMA, Momentum)
- Feature engineering and normalization
- Train/test splitting

Mathematical Formulations:
- Returns: r_t = (p_t - p_{t-1}) / p_{t-1}
- Log Returns: r_t = log(p_t / p_{t-1})
- SMA: SMA(L)_t = (1/L) * sum_{i=0}^{L-1} p_{t-i}
- EMA: EMA(α)_t = α * p_t + (1-α) * EMA_{t-1}
- Momentum: Mom(L)_t = (p_t - p_{t-L}) / p_{t-L}
"""

import os
import warnings
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")


class DataLoader:
    """
    Data loader for financial time series with feature engineering.
    
    Attributes:
        assets (List[str]): List of asset tickers.
        start_date (str): Start date for data download.
        end_date (str): End date for data download.
        data_dir (str): Directory to cache downloaded data.
    """
    
    def __init__(
        self,
        assets: List[str],
        start_date: str = "2015-01-01",
        end_date: str = "2024-12-31",
        data_dir: str = "data",
        frequency: str = "1d",
    ):
        """
        Initialize DataLoader.
        
        Args:
            assets: List of ticker symbols (e.g., ['AAPL', 'GOOGL']).
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            data_dir: Directory to save/load cached data.
            frequency: Data frequency ('1d' for daily, '1wk' for weekly).
        """
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        self.frequency = frequency
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Placeholders for data
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.log_returns: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        
    def download_data(self, force_download: bool = False) -> pd.DataFrame:
        """
        Download price data from Yahoo Finance.
        
        Args:
            force_download: If True, re-download even if cached data exists.
            
        Returns:
            DataFrame with adjusted closing prices for each asset.
        """
        cache_file = os.path.join(
            self.data_dir,
            f"prices_{'_'.join(self.assets)}_{self.start_date}_{self.end_date}.csv"
        )
        
        # Load from cache if available
        if os.path.exists(cache_file) and not force_download:
            print(f"Loading cached data from {cache_file}")
            self.prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return self.prices
        
        print(f"Downloading data for {len(self.assets)} assets from Yahoo Finance...")
        
        # Download data
        data = yf.download(
            self.assets,
            start=self.start_date,
            end=self.end_date,
            interval=self.frequency,
            progress=True,
            auto_adjust=True,  # Use adjusted close prices
        )
        
        # Extract closing prices
        if len(self.assets) == 1:
            prices = data[['Close']].copy()
            prices.columns = self.assets
        else:
            prices = data['Close'].copy()
        
        # Handle missing values
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        
        # Drop rows with any remaining NaN values
        prices = prices.dropna()
        
        if prices.empty:
            raise ValueError("No valid data downloaded. Check asset tickers and date range.")
        
        # Save to cache
        prices.to_csv(cache_file)
        print(f"Data saved to {cache_file}")
        print(f"Downloaded {len(prices)} periods for {len(prices.columns)} assets")
        
        self.prices = prices
        return prices
    
    def compute_returns(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute simple and log returns from prices.
        
        Returns:
            Tuple of (simple_returns, log_returns) DataFrames.
            
        Mathematical Formulas:
            Simple return: r_t = (p_t - p_{t-1}) / p_{t-1}
            Log return: r_t = log(p_t / p_{t-1})
        """
        if self.prices is None:
            raise ValueError("Prices not loaded. Call download_data() first.")
        
        # Simple returns: (p_t - p_{t-1}) / p_{t-1}
        self.returns = self.prices.pct_change().dropna()
        
        # Log returns: log(p_t / p_{t-1})
        self.log_returns = np.log(self.prices / self.prices.shift(1)).dropna()
        
        return self.returns, self.log_returns
    
    def compute_sma(self, periods: List[int]) -> pd.DataFrame:
        """
        Compute Simple Moving Averages.
        
        Args:
            periods: List of lookback periods (e.g., [5, 10, 20]).
            
        Returns:
            DataFrame with SMA features for each asset and period.
            
        Formula:
            SMA(L)_t = (1/L) * sum_{i=0}^{L-1} p_{t-i}
        """
        if self.prices is None:
            raise ValueError("Prices not loaded. Call download_data() first.")
        
        sma_features = {}
        for period in periods:
            for asset in self.assets:
                col_name = f"{asset}_SMA_{period}"
                sma_features[col_name] = self.prices[asset].rolling(window=period).mean()
        
        return pd.DataFrame(sma_features, index=self.prices.index)
    
    def compute_ema(
        self,
        periods: List[int],
        alpha: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Compute Exponential Moving Averages.
        
        Args:
            periods: List of lookback periods (e.g., [5, 10, 20]).
            alpha: Smoothing factor (if None, uses span-based alpha).
            
        Returns:
            DataFrame with EMA features for each asset and period.
            
        Formula:
            EMA(α)_t = α * p_t + (1-α) * EMA_{t-1}
            where α = 2 / (period + 1) if not specified
        """
        if self.prices is None:
            raise ValueError("Prices not loaded. Call download_data() first.")
        
        ema_features = {}
        for period in periods:
            for asset in self.assets:
                col_name = f"{asset}_EMA_{period}"
                if alpha is not None:
                    # Use custom alpha
                    ema_features[col_name] = self.prices[asset].ewm(
                        alpha=alpha, adjust=False
                    ).mean()
                else:
                    # Use span-based alpha: α = 2/(span+1)
                    ema_features[col_name] = self.prices[asset].ewm(
                        span=period, adjust=False
                    ).mean()
        
        return pd.DataFrame(ema_features, index=self.prices.index)
    
    def compute_momentum(self, periods: List[int]) -> pd.DataFrame:
        """
        Compute price momentum indicators.
        
        Args:
            periods: List of lookback periods (e.g., [5, 10, 20]).
            
        Returns:
            DataFrame with momentum features for each asset and period.
            
        Formula:
            Momentum(L)_t = (p_t - p_{t-L}) / p_{t-L}
        """
        if self.prices is None:
            raise ValueError("Prices not loaded. Call download_data() first.")
        
        momentum_features = {}
        for period in periods:
            for asset in self.assets:
                col_name = f"{asset}_MOM_{period}"
                momentum_features[col_name] = (
                    (self.prices[asset] - self.prices[asset].shift(period))
                    / self.prices[asset].shift(period)
                )
        
        return pd.DataFrame(momentum_features, index=self.prices.index)
    
    def compute_volatility(self, window: int = 20) -> pd.DataFrame:
        """
        Compute rolling volatility.
        
        Args:
            window: Lookback window for volatility calculation.
            
        Returns:
            DataFrame with volatility for each asset.
        """
        if self.returns is None:
            self.compute_returns()
        
        volatility = self.returns.rolling(window=window).std()
        volatility.columns = [f"{col}_VOL" for col in volatility.columns]
        
        return volatility
    
    def normalize_features(
        self,
        data: pd.DataFrame,
        method: str = "zscore",
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Normalize features for better training stability.
        
        Args:
            data: DataFrame to normalize.
            method: Normalization method ('zscore', 'minmax', or 'none').
            window: Rolling window size (None for global normalization).
            
        Returns:
            Normalized DataFrame.
        """
        if method == "none":
            return data
        
        normalized = data.copy()
        
        if method == "zscore":
            if window is not None:
                # Rolling z-score normalization
                mean = data.rolling(window=window, min_periods=1).mean()
                std = data.rolling(window=window, min_periods=1).std()
                normalized = (data - mean) / (std + 1e-8)
            else:
                # Global z-score normalization
                normalized = (data - data.mean()) / (data.std() + 1e-8)
                
        elif method == "minmax":
            if window is not None:
                # Rolling min-max normalization
                min_val = data.rolling(window=window, min_periods=1).min()
                max_val = data.rolling(window=window, min_periods=1).max()
                normalized = (data - min_val) / (max_val - min_val + 1e-8)
            else:
                # Global min-max normalization
                normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        return normalized
    
    def build_features(
        self,
        sma_periods: List[int] = [5, 10, 20],
        ema_periods: List[int] = [5, 10, 20],
        momentum_periods: List[int] = [5, 10, 20],
        include_volatility: bool = True,
        volatility_window: int = 20,
        normalize: bool = True,
        normalize_method: str = "zscore",
        rolling_window: Optional[int] = 60,
    ) -> pd.DataFrame:
        """
        Build complete feature set for the portfolio environment.
        
        Args:
            sma_periods: Periods for SMA calculation.
            ema_periods: Periods for EMA calculation.
            momentum_periods: Periods for momentum calculation.
            include_volatility: Whether to include volatility features.
            volatility_window: Window for volatility calculation.
            normalize: Whether to normalize features.
            normalize_method: Normalization method.
            rolling_window: Rolling window for normalization.
            
        Returns:
            DataFrame with all engineered features.
        """
        print("Building feature set...")
        
        # Ensure returns are computed
        if self.returns is None:
            self.compute_returns()
        
        feature_dfs = []
        
        # Add normalized prices
        if normalize:
            normalized_prices = self.normalize_features(
                self.prices, normalize_method, rolling_window
            )
            normalized_prices.columns = [f"{col}_PRICE_NORM" for col in self.prices.columns]
            feature_dfs.append(normalized_prices)
        else:
            price_features = self.prices.copy()
            price_features.columns = [f"{col}_PRICE" for col in self.prices.columns]
            feature_dfs.append(price_features)
        
        # Add returns
        return_features = self.returns.copy()
        return_features.columns = [f"{col}_RETURN" for col in self.returns.columns]
        feature_dfs.append(return_features)
        
        # Add SMA features
        sma = self.compute_sma(sma_periods)
        if normalize:
            sma = self.normalize_features(sma, normalize_method, rolling_window)
        feature_dfs.append(sma)
        
        # Add EMA features
        ema = self.compute_ema(ema_periods)
        if normalize:
            ema = self.normalize_features(ema, normalize_method, rolling_window)
        feature_dfs.append(ema)
        
        # Add Momentum features
        momentum = self.compute_momentum(momentum_periods)
        if normalize:
            momentum = self.normalize_features(momentum, normalize_method, rolling_window)
        feature_dfs.append(momentum)
        
        # Add Volatility features
        if include_volatility:
            volatility = self.compute_volatility(volatility_window)
            if normalize:
                volatility = self.normalize_features(volatility, normalize_method, rolling_window)
            feature_dfs.append(volatility)
        
        # Combine all features
        self.features = pd.concat(feature_dfs, axis=1)
        
        # Drop rows with NaN values (from indicator calculations)
        self.features = self.features.dropna()
        
        print(f"Built {self.features.shape[1]} features over {self.features.shape[0]} periods")
        
        return self.features
    
    def train_test_split(
        self,
        train_ratio: float = 0.7,
        train_end: Optional[str] = None
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Split data into training and testing sets.
        
        Args:
            train_ratio: Proportion of data for training (e.g., 0.7 = 70%).
                         Ignored if train_end is specified.
            train_end: Optional explicit end date for training period (YYYY-MM-DD).
                      If specified, data up to and including this date is used for training,
                      and data after this date is used for testing.
            
        Returns:
            Tuple of (train_data, test_data) dictionaries containing:
                - 'prices': Price data
                - 'returns': Return data
                - 'features': Feature data
        """
        if self.features is None:
            raise ValueError("Features not built. Call build_features() first.")
        
        # Determine split index
        if train_end is not None:
            # Use explicit date for split
            train_end_date = pd.to_datetime(train_end)
            split_idx = (self.features.index <= train_end_date).sum()
            print(f"Using explicit train_end date: {train_end}")
            print(f"Training samples: {split_idx}, Test samples: {len(self.features) - split_idx}")
        else:
            # Use ratio-based split
            split_idx = int(len(self.features) * train_ratio)
        
        # Align all dataframes to feature index
        aligned_prices = self.prices.loc[self.features.index]
        aligned_returns = self.returns.loc[self.features.index]
        
        # Split data
        train_data = {
            'prices': aligned_prices.iloc[:split_idx],
            'returns': aligned_returns.iloc[:split_idx],
            'features': self.features.iloc[:split_idx],
        }
        
        test_data = {
            'prices': aligned_prices.iloc[split_idx:],
            'returns': aligned_returns.iloc[split_idx:],
            'features': self.features.iloc[split_idx:],
        }
        
        print(f"Train period: {train_data['prices'].index[0]} to {train_data['prices'].index[-1]}")
        print(f"Test period: {test_data['prices'].index[0]} to {test_data['prices'].index[-1]}")
        print(f"Train samples: {len(train_data['prices'])}")
        print(f"Test samples: {len(test_data['prices'])}")
        
        return train_data, test_data
    
    def get_asset_statistics(self) -> pd.DataFrame:
        """
        Compute descriptive statistics for each asset.
        
        Returns:
            DataFrame with statistics (mean return, volatility, Sharpe, etc.).
        """
        if self.returns is None:
            self.compute_returns()
        
        stats_dict = {}
        
        for asset in self.assets:
            asset_returns = self.returns[asset]
            
            stats_dict[asset] = {
                'Mean Daily Return': asset_returns.mean(),
                'Std Daily Return': asset_returns.std(),
                'Annualized Return': (1 + asset_returns.mean()) ** 252 - 1,
                'Annualized Volatility': asset_returns.std() * np.sqrt(252),
                'Sharpe Ratio': asset_returns.mean() / asset_returns.std() * np.sqrt(252),
                'Skewness': stats.skew(asset_returns.dropna()),
                'Kurtosis': stats.kurtosis(asset_returns.dropna()),
                'Min Return': asset_returns.min(),
                'Max Return': asset_returns.max(),
            }
        
        return pd.DataFrame(stats_dict).T


if __name__ == "__main__":
    """Example usage of DataLoader."""
    
    # Define assets
    assets = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL', 'SPY', 'GLD']
    
    # Initialize loader
    loader = DataLoader(
        assets=assets,
        start_date="2015-01-01",
        end_date="2024-12-31",
        data_dir="data"
    )
    
    # Download data
    prices = loader.download_data()
    print("\nPrice data shape:", prices.shape)
    print("\nFirst few rows:")
    print(prices.head())
    
    # Compute returns
    returns, log_returns = loader.compute_returns()
    print("\nReturns shape:", returns.shape)
    
    # Build features
    features = loader.build_features(
        sma_periods=[5, 10, 20],
        ema_periods=[5, 10, 20],
        momentum_periods=[5, 10, 20],
        normalize=True
    )
    print("\nFeature data shape:", features.shape)
    
    # Train-test split
    train_data, test_data = loader.train_test_split(train_ratio=0.7)
    
    # Get statistics
    stats = loader.get_asset_statistics()
    print("\nAsset Statistics:")
    print(stats)
