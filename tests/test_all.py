"""
Test suite for DRL Portfolio Optimization project.

Run with: pytest tests/test_all.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
from portfolio_env import PortfolioEnv
from benchmarks import EqualWeightStrategy, MomentumStrategy
from metrics import PerformanceMetrics


class TestDataLoader:
    """Test data loading and preprocessing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.assets = ['AAPL', 'MSFT']
        self.loader = DataLoader(
            assets=self.assets,
            start_date='2020-01-01',
            end_date='2020-12-31',
            data_dir='../data'
        )
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        assert self.loader.assets == self.assets
        assert self.loader.start_date == '2020-01-01'
        assert self.loader.end_date == '2020-12-31'
    
    def test_download_data(self):
        """Test data download."""
        prices = self.loader.download_data()
        assert isinstance(prices, pd.DataFrame)
        assert len(prices.columns) == len(self.assets)
        assert len(prices) > 0
    
    def test_compute_returns(self):
        """Test return computation."""
        self.loader.download_data()
        returns, log_returns = self.loader.compute_returns()
        assert isinstance(returns, pd.DataFrame)
        assert isinstance(log_returns, pd.DataFrame)
        assert returns.shape == log_returns.shape
    
    def test_technical_indicators(self):
        """Test technical indicator computation."""
        self.loader.download_data()
        
        # SMA
        sma = self.loader.compute_sma([5, 10])
        assert isinstance(sma, pd.DataFrame)
        assert sma.shape[1] == len(self.assets) * 2
        
        # EMA
        ema = self.loader.compute_ema([5, 10])
        assert isinstance(ema, pd.DataFrame)
        
        # Momentum
        momentum = self.loader.compute_momentum([5, 10])
        assert isinstance(momentum, pd.DataFrame)
    
    def test_feature_building(self):
        """Test complete feature building."""
        self.loader.download_data()
        features = self.loader.build_features()
        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] > len(self.assets)  # More features than assets
    
    def test_train_test_split(self):
        """Test train-test split."""
        self.loader.download_data()
        self.loader.build_features()
        train_data, test_data = self.loader.train_test_split(train_ratio=0.7)
        
        assert 'prices' in train_data
        assert 'returns' in train_data
        assert 'features' in train_data
        assert len(train_data['prices']) > len(test_data['prices'])


class TestPortfolioEnv:
    """Test portfolio environment."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create dummy data
        np.random.seed(42)
        n_periods = 100
        n_assets = 3
        
        dates = pd.date_range('2020-01-01', periods=n_periods)
        self.prices = pd.DataFrame(
            np.random.randn(n_periods, n_assets).cumsum(axis=0) + 100,
            index=dates,
            columns=['Asset_A', 'Asset_B', 'Asset_C']
        )
        self.returns = self.prices.pct_change().fillna(0)
        self.features = self.returns.copy()
        
        self.env = PortfolioEnv(
            prices=self.prices,
            returns=self.returns,
            features=self.features,
            initial_balance=100000,
            transaction_cost=0.001,
            lookback_window=10
        )
    
    def test_initialization(self):
        """Test environment initialization."""
        assert self.env.n_assets == 3
        assert self.env.initial_balance == 100000
        assert self.env.transaction_cost == 0.001
    
    def test_reset(self):
        """Test environment reset."""
        obs, info = self.env.reset()
        assert obs.shape == (self.env.observation_dim,)
        assert 'portfolio_value' in info
        assert info['portfolio_value'] == 100000
    
    def test_step(self):
        """Test environment step."""
        obs, info = self.env.reset()
        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        assert next_obs.shape == obs.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert 'portfolio_value' in info
    
    def test_episode(self):
        """Test complete episode."""
        obs, info = self.env.reset()
        done = False
        steps = 0
        
        while not done and steps < 10:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            steps += 1
        
        assert steps > 0
        history = self.env.get_portfolio_history()
        assert len(history['returns']) == steps
    
    def test_softmax_weights(self):
        """Test softmax weight conversion."""
        logits = np.array([1.0, 2.0, 3.0])
        weights = self.env._softmax_weights(logits)
        
        assert np.allclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
        assert np.all(weights <= 1)


class TestBenchmarks:
    """Test benchmark strategies."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        n_periods = 100
        n_assets = 3
        
        self.returns = pd.DataFrame(
            np.random.randn(n_periods, n_assets) * 0.01,
            columns=['Asset_A', 'Asset_B', 'Asset_C']
        )
    
    def test_equal_weight(self):
        """Test equal-weight strategy."""
        strategy = EqualWeightStrategy(n_assets=3)
        weights = strategy.get_weights()
        
        assert np.allclose(weights, 1/3)
        assert np.allclose(np.sum(weights), 1.0)
    
    def test_equal_weight_backtest(self):
        """Test equal-weight backtest."""
        strategy = EqualWeightStrategy(n_assets=3)
        results = strategy.run_backtest(
            self.returns,
            transaction_cost=0.001,
            initial_value=100000
        )
        
        assert 'returns' in results
        assert 'values' in results
        assert len(results['returns']) == len(self.returns)
    
    def test_momentum(self):
        """Test momentum strategy."""
        strategy = MomentumStrategy(n_assets=3, lookback=20, top_k=2)
        
        # Need enough data for momentum calculation
        weights = strategy.get_weights(returns=self.returns)
        
        assert np.allclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
        assert np.sum(weights > 0) <= 2  # Top 2 assets


class TestMetrics:
    """Test performance metrics."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.01, 252)  # One year of daily returns
        self.values = np.cumprod(1 + self.returns) * 100000
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = PerformanceMetrics(
            returns=self.returns,
            values=self.values,
            risk_free_rate=0.02
        )
        assert len(metrics.returns) == len(self.returns)
    
    def test_annualized_return(self):
        """Test annualized return calculation."""
        metrics = PerformanceMetrics(returns=self.returns)
        ar = metrics.annualized_return()
        assert isinstance(ar, (int, float))
    
    def test_volatility(self):
        """Test volatility calculation."""
        metrics = PerformanceMetrics(returns=self.returns)
        vol = metrics.annualized_volatility()
        assert isinstance(vol, (int, float))
        assert vol > 0
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        metrics = PerformanceMetrics(returns=self.returns, risk_free_rate=0.02)
        sharpe = metrics.sharpe_ratio()
        assert isinstance(sharpe, (int, float))
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        metrics = PerformanceMetrics(returns=self.returns, values=self.values)
        mdd = metrics.max_drawdown()
        assert isinstance(mdd, (int, float))
        assert 0 <= mdd <= 1
    
    def test_all_metrics(self):
        """Test getting all metrics."""
        metrics = PerformanceMetrics(returns=self.returns, values=self.values)
        all_metrics = metrics.get_all_metrics()
        
        assert isinstance(all_metrics, dict)
        assert 'Sharpe Ratio' in all_metrics
        assert 'Max Drawdown' in all_metrics
        assert 'Annualized Return' in all_metrics


def test_imports():
    """Test that all modules can be imported."""
    try:
        from data_loader import DataLoader
        from portfolio_env import PortfolioEnv
        from agents import create_agent
        from benchmarks import EqualWeightStrategy, MomentumStrategy
        from metrics import PerformanceMetrics
        from visualization import plot_cumulative_returns
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, '-v'])
