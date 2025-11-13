"""
Performance evaluation metrics for portfolio optimization.

Implements financial metrics:
- Annualized Return (AR)
- Sharpe Ratio
- Maximum Drawdown (MDD)
- Annualized Volatility
- Average Turnover
- Sortino Ratio
- Calmar Ratio
- Value at Risk (VaR)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from scipy import stats


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for portfolio strategies.
    
    Attributes:
        returns (np.ndarray): Array of portfolio returns.
        values (np.ndarray): Array of portfolio values.
        weights (np.ndarray): Array of portfolio weights over time.
        risk_free_rate (float): Annual risk-free rate.
        periods_per_year (int): Number of periods per year (252 for daily).
    """
    
    def __init__(
        self,
        returns: np.ndarray,
        values: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        turnover: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ):
        """
        Initialize PerformanceMetrics.
        
        Args:
            returns: Array of portfolio returns.
            values: Array of portfolio values (computed if not provided).
            weights: Array of portfolio weights over time.
            turnover: Array of turnover values.
            risk_free_rate: Annual risk-free rate.
            periods_per_year: Trading periods per year.
        """
        self.returns = np.array(returns)
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.daily_rf = risk_free_rate / periods_per_year
        
        # Compute values if not provided
        if values is not None:
            self.values = np.array(values)
        else:
            self.values = np.cumprod(1 + self.returns) * 100000  # Assume $100k start
        
        self.weights = np.array(weights) if weights is not None else None
        self.turnover = np.array(turnover) if turnover is not None else None
    
    def annualized_return(self) -> float:
        """
        Calculate annualized return.
        
        Formula:
            AR = [prod_t (1 + R_t)]^{252/T} - 1
            
        Returns:
            Annualized return as decimal.
        """
        if len(self.returns) == 0:
            return 0.0
        
        total_return = np.prod(1 + self.returns)
        n_periods = len(self.returns)
        annualized = total_return ** (self.periods_per_year / n_periods) - 1
        
        return annualized
    
    def annualized_volatility(self) -> float:
        """
        Calculate annualized volatility (standard deviation).
        
        Formula:
            σ_ann = std(R_t) * sqrt(252)
            
        Returns:
            Annualized volatility as decimal.
        """
        if len(self.returns) <= 1:
            return 0.0
        
        volatility = np.std(self.returns, ddof=1) * np.sqrt(self.periods_per_year)
        return volatility
    
    def sharpe_ratio(self) -> float:
        """
        Calculate Sharpe ratio.
        
        Formula:
            Sharpe = (AR - r_f) / σ_ann
            
        Returns:
            Sharpe ratio.
        """
        ann_return = self.annualized_return()
        ann_vol = self.annualized_volatility()
        
        if ann_vol == 0:
            return 0.0
        
        sharpe = (ann_return - self.risk_free_rate) / ann_vol
        return sharpe
    
    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.
        
        Formula:
            MDD = max_{t' < t} [(V_{t'} - V_t) / V_{t'}]
            
        Returns:
            Maximum drawdown as positive decimal.
        """
        if len(self.values) == 0:
            return 0.0
        
        cummax = np.maximum.accumulate(self.values)
        drawdown = (cummax - self.values) / cummax
        max_dd = np.max(drawdown)
        
        return max_dd
    
    def calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).
        
        Formula:
            Calmar = AR / MDD
            
        Returns:
            Calmar ratio.
        """
        ann_return = self.annualized_return()
        mdd = self.max_drawdown()
        
        if mdd == 0:
            return 0.0
        
        calmar = ann_return / mdd
        return calmar
    
    def sortino_ratio(self, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (penalizes only downside volatility).
        
        Formula:
            Sortino = (AR - target) / σ_downside
            where σ_downside = std(min(R_t - target, 0))
            
        Args:
            target_return: Target return threshold.
            
        Returns:
            Sortino ratio.
        """
        ann_return = self.annualized_return()
        
        # Calculate downside deviation
        downside_returns = np.minimum(self.returns - target_return, 0)
        downside_vol = np.std(downside_returns, ddof=1) * np.sqrt(self.periods_per_year)
        
        if downside_vol == 0:
            return 0.0
        
        sortino = (ann_return - target_return) / downside_vol
        return sortino
    
    def value_at_risk(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            confidence: Confidence level (e.g., 0.95 for 95% VaR).
            
        Returns:
            VaR as positive decimal (loss threshold).
        """
        if len(self.returns) == 0:
            return 0.0
        
        var = -np.percentile(self.returns, (1 - confidence) * 100)
        return var
    
    def conditional_var(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
        
        Args:
            confidence: Confidence level.
            
        Returns:
            CVaR as positive decimal.
        """
        if len(self.returns) == 0:
            return 0.0
        
        var = self.value_at_risk(confidence)
        cvar = -np.mean(self.returns[self.returns <= -var])
        
        return cvar if not np.isnan(cvar) else 0.0
    
    def average_turnover(self) -> float:
        """
        Calculate average portfolio turnover.
        
        Formula:
            Avg_Turn = (1/T) * sum_t Turn_t
            
        Returns:
            Average turnover.
        """
        if self.turnover is None or len(self.turnover) == 0:
            if self.weights is not None and len(self.weights) > 1:
                # Calculate turnover from weights
                turnover = np.sum(np.abs(np.diff(self.weights, axis=0)), axis=1)
                return np.mean(turnover)
            return 0.0
        
        return np.mean(self.turnover)
    
    def total_return(self) -> float:
        """
        Calculate total cumulative return.
        
        Returns:
            Total return as decimal.
        """
        if len(self.returns) == 0:
            return 0.0
        
        return np.prod(1 + self.returns) - 1
    
    def hit_ratio(self) -> float:
        """
        Calculate hit ratio (proportion of positive returns).
        
        Returns:
            Hit ratio as decimal.
        """
        if len(self.returns) == 0:
            return 0.0
        
        return np.mean(self.returns > 0)
    
    def skewness(self) -> float:
        """
        Calculate return skewness.
        
        Returns:
            Skewness coefficient.
        """
        if len(self.returns) <= 2:
            return 0.0
        
        return stats.skew(self.returns)
    
    def kurtosis(self) -> float:
        """
        Calculate return kurtosis (excess kurtosis).
        
        Returns:
            Excess kurtosis coefficient.
        """
        if len(self.returns) <= 3:
            return 0.0
        
        return stats.kurtosis(self.returns)
    
    def information_ratio(self, benchmark_returns: np.ndarray) -> float:
        """
        Calculate information ratio vs benchmark.
        
        Formula:
            IR = mean(R_p - R_b) / std(R_p - R_b)
            
        Args:
            benchmark_returns: Benchmark return series.
            
        Returns:
            Information ratio.
        """
        if len(self.returns) != len(benchmark_returns):
            return 0.0
        
        active_returns = self.returns - benchmark_returns
        
        if len(active_returns) <= 1:
            return 0.0
        
        active_mean = np.mean(active_returns)
        tracking_error = np.std(active_returns, ddof=1)
        
        if tracking_error == 0:
            return 0.0
        
        ir = active_mean / tracking_error * np.sqrt(self.periods_per_year)
        return ir
    
    def get_all_metrics(
        self,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Args:
            benchmark_returns: Optional benchmark returns for comparison.
            
        Returns:
            Dictionary of all metrics.
        """
        metrics = {
            'Total Return': self.total_return(),
            'Annualized Return': self.annualized_return(),
            'Annualized Volatility': self.annualized_volatility(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Max Drawdown': self.max_drawdown(),
            'Calmar Ratio': self.calmar_ratio(),
            'VaR (95%)': self.value_at_risk(0.95),
            'CVaR (95%)': self.conditional_var(0.95),
            'Hit Ratio': self.hit_ratio(),
            'Skewness': self.skewness(),
            'Kurtosis': self.kurtosis(),
            'Average Turnover': self.average_turnover(),
        }
        
        if benchmark_returns is not None:
            metrics['Information Ratio'] = self.information_ratio(benchmark_returns)
        
        return metrics
    
    def print_metrics(
        self,
        name: str = "Strategy",
        benchmark_returns: Optional[np.ndarray] = None
    ) -> None:
        """
        Print all metrics in a formatted table.
        
        Args:
            name: Strategy name for display.
            benchmark_returns: Optional benchmark returns.
        """
        metrics = self.get_all_metrics(benchmark_returns)
        
        print(f"\n{'='*60}")
        print(f" Performance Metrics: {name}")
        print(f"{'='*60}")
        
        for metric_name, value in metrics.items():
            if 'Return' in metric_name or 'Volatility' in metric_name or \
               'Drawdown' in metric_name or 'VaR' in metric_name or \
               'CVaR' in metric_name or 'Turnover' in metric_name:
                print(f"{metric_name:.<40} {value:>12.2%}")
            else:
                print(f"{metric_name:.<40} {value:>12.4f}")
        
        print(f"{'='*60}\n")


def compare_strategies(
    strategies: Dict[str, Dict[str, np.ndarray]],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Compare multiple strategies using performance metrics.
    
    Args:
        strategies: Dictionary mapping strategy names to data dictionaries
                   (must contain 'returns', optionally 'values', 'weights', 'turnover').
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Trading periods per year.
        
    Returns:
        DataFrame with metrics for each strategy.
    """
    results = {}
    
    for name, data in strategies.items():
        metrics_obj = PerformanceMetrics(
            returns=data['returns'],
            values=data.get('values'),
            weights=data.get('weights'),
            turnover=data.get('turnover'),
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        
        results[name] = metrics_obj.get_all_metrics()
    
    # Convert to DataFrame
    df = pd.DataFrame(results).T
    
    # Sort by Sharpe Ratio (descending)
    df = df.sort_values('Sharpe Ratio', ascending=False)
    
    return df


if __name__ == "__main__":
    """Example usage of PerformanceMetrics."""
    
    # Generate sample returns
    np.random.seed(42)
    n_periods = 252  # One year of daily data
    
    # Strategy 1: Positive drift with volatility
    returns_1 = np.random.normal(0.0005, 0.01, n_periods)
    
    # Strategy 2: Higher returns, higher volatility
    returns_2 = np.random.normal(0.001, 0.02, n_periods)
    
    # Strategy 3: Lower returns, lower volatility
    returns_3 = np.random.normal(0.0003, 0.005, n_periods)
    
    # Calculate metrics for each strategy
    print("Strategy 1: Moderate Risk-Return")
    metrics_1 = PerformanceMetrics(returns_1, risk_free_rate=0.02)
    metrics_1.print_metrics("Strategy 1")
    
    print("Strategy 2: High Risk-Return")
    metrics_2 = PerformanceMetrics(returns_2, risk_free_rate=0.02)
    metrics_2.print_metrics("Strategy 2")
    
    print("Strategy 3: Low Risk-Return")
    metrics_3 = PerformanceMetrics(returns_3, risk_free_rate=0.02)
    metrics_3.print_metrics("Strategy 3")
    
    # Compare all strategies
    print("\nComparison of All Strategies:")
    comparison = compare_strategies({
        'Strategy 1': {'returns': returns_1},
        'Strategy 2': {'returns': returns_2},
        'Strategy 3': {'returns': returns_3},
    })
    print(comparison)
