"""
Benchmark strategies for portfolio optimization comparison.

Implements:
1. Equal-Weight Strategy: w_i = 1/N
2. Mean-Variance Optimization: Markowitz quadratic programming
3. Momentum Strategy: Allocate based on past returns
"""

import numpy as np
import pandas as pd
from typing import Optional, List
import warnings

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("cvxpy not available. Mean-Variance optimization will be disabled.")


class BaseStrategy:
    """Base class for portfolio strategies."""
    
    def __init__(self, n_assets: int):
        """
        Initialize strategy.
        
        Args:
            n_assets: Number of assets in portfolio.
        """
        self.n_assets = n_assets
        self.weights_history = []
    
    def get_weights(self, **kwargs) -> np.ndarray:
        """
        Get portfolio weights for current period.
        
        Returns:
            Array of portfolio weights.
        """
        raise NotImplementedError
    
    def run_backtest(
        self,
        returns: pd.DataFrame,
        transaction_cost: float = 0.0,
        initial_value: float = 100000.0
    ) -> dict:
        """
        Run backtest of strategy.
        
        Args:
            returns: DataFrame of asset returns.
            transaction_cost: Transaction cost per unit turnover.
            initial_value: Initial portfolio value.
            
        Returns:
            Dictionary with backtest results.
        """
        n_periods = len(returns)
        portfolio_values = [initial_value]
        portfolio_returns = []
        weights_history = []
        turnover_history = []
        
        # Initialize with equal weights
        prev_weights = np.ones(self.n_assets) / self.n_assets
        weights_history.append(prev_weights.copy())
        
        for t in range(n_periods):
            # Get new weights
            weights = self.get_weights(
                returns=returns.iloc[:t+1],
                current_step=t
            )
            
            # Calculate turnover
            turnover = np.sum(np.abs(weights - prev_weights))
            turnover_history.append(turnover)
            
            # Calculate costs
            cost = transaction_cost * turnover
            
            # Calculate gross return
            period_returns = returns.iloc[t].values
            gross_return = np.dot(prev_weights, period_returns)
            
            # Calculate net return
            net_return = gross_return - cost
            
            # Update portfolio value
            current_value = portfolio_values[-1] * (1 + net_return)
            portfolio_values.append(current_value)
            portfolio_returns.append(net_return)
            
            # Store weights
            weights_history.append(weights.copy())
            prev_weights = weights
        
        return {
            'values': np.array(portfolio_values[1:]),  # Exclude initial
            'returns': np.array(portfolio_returns),
            'weights': np.array(weights_history[1:]),  # Exclude initial
            'turnover': np.array(turnover_history),
        }


class EqualWeightStrategy(BaseStrategy):
    """
    Equal-weight (1/N) portfolio strategy.
    
    Formula:
        w_i = 1/N for all i
    """
    
    def __init__(self, n_assets: int):
        """
        Initialize equal-weight strategy.
        
        Args:
            n_assets: Number of assets.
        """
        super().__init__(n_assets)
        self.weights = np.ones(n_assets) / n_assets
    
    def get_weights(self, **kwargs) -> np.ndarray:
        """
        Get equal weights.
        
        Returns:
            Equal weights for all assets.
        """
        return self.weights.copy()


class MeanVarianceStrategy(BaseStrategy):
    """
    Mean-Variance Optimization (Markowitz) strategy.
    
    Solves:
        max_w  w^T μ - (λ/2) w^T Σ w
        s.t.   sum(w) = 1, w >= 0
        
    Or for target return:
        min_w  w^T Σ w
        s.t.   w^T μ >= target_return, sum(w) = 1, w >= 0
    """
    
    def __init__(
        self,
        n_assets: int,
        lookback: int = 60,
        target_return: Optional[float] = None,
        allow_short: bool = False,
        risk_aversion: float = 1.0
    ):
        """
        Initialize mean-variance strategy.
        
        Args:
            n_assets: Number of assets.
            lookback: Lookback period for estimating mean and covariance.
            target_return: Target return (None for max Sharpe).
            allow_short: Whether to allow short positions.
            risk_aversion: Risk aversion parameter (higher = more conservative).
        """
        super().__init__(n_assets)
        
        if not CVXPY_AVAILABLE:
            raise ImportError("cvxpy required for Mean-Variance optimization")
        
        self.lookback = lookback
        self.target_return = target_return
        self.allow_short = allow_short
        self.risk_aversion = risk_aversion
    
    def get_weights(
        self,
        returns: pd.DataFrame,
        **kwargs
    ) -> np.ndarray:
        """
        Calculate optimal weights using mean-variance optimization.
        
        Args:
            returns: Historical returns DataFrame.
            
        Returns:
            Optimal portfolio weights.
        """
        # Use last 'lookback' periods
        if len(returns) < self.lookback:
            # Not enough data, use equal weights
            return np.ones(self.n_assets) / self.n_assets
        
        recent_returns = returns.iloc[-self.lookback:].values
        
        # Estimate mean and covariance
        mean_returns = np.mean(recent_returns, axis=0)
        cov_matrix = np.cov(recent_returns, rowvar=False)
        
        # Add regularization to covariance matrix for numerical stability
        cov_matrix += np.eye(self.n_assets) * 1e-5
        
        try:
            # Define optimization problem
            w = cp.Variable(self.n_assets)
            
            # Objective: maximize return - risk_aversion * variance
            portfolio_return = mean_returns @ w
            portfolio_variance = cp.quad_form(w, cov_matrix)
            objective = cp.Maximize(portfolio_return - (self.risk_aversion / 2) * portfolio_variance)
            
            # Constraints
            constraints = [cp.sum(w) == 1]
            
            if not self.allow_short:
                constraints.append(w >= 0)
            
            if self.target_return is not None:
                constraints.append(portfolio_return >= self.target_return)
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if w.value is None:
                # Optimization failed, use equal weights
                return np.ones(self.n_assets) / self.n_assets
            
            weights = np.array(w.value).flatten()
            
            # Ensure weights are valid
            weights = np.maximum(weights, 0) if not self.allow_short else weights
            weights = weights / np.sum(weights)
            
            return weights
            
        except Exception as e:
            warnings.warn(f"Mean-variance optimization failed: {e}. Using equal weights.")
            return np.ones(self.n_assets) / self.n_assets


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based strategy.
    
    Allocates to assets with highest past returns.
    
    Formula:
        - Calculate momentum: Mom_i = (p_t - p_{t-L}) / p_{t-L}
        - Allocate to top K assets
        - w_i = Mom_i / sum(Mom_j) for top K, 0 otherwise
    """
    
    def __init__(
        self,
        n_assets: int,
        lookback: int = 20,
        top_k: Optional[int] = None,
        method: str = "proportional"
    ):
        """
        Initialize momentum strategy.
        
        Args:
            n_assets: Number of assets.
            lookback: Lookback period for momentum calculation.
            top_k: Number of top assets to invest in (None for all).
            method: Allocation method ('equal' or 'proportional').
        """
        super().__init__(n_assets)
        self.lookback = lookback
        self.top_k = top_k if top_k is not None else n_assets
        self.method = method
    
    def get_weights(
        self,
        returns: pd.DataFrame,
        **kwargs
    ) -> np.ndarray:
        """
        Calculate momentum-based weights.
        
        Args:
            returns: Historical returns DataFrame.
            
        Returns:
            Momentum-based portfolio weights.
        """
        # Check if enough data
        if len(returns) < self.lookback:
            return np.ones(self.n_assets) / self.n_assets
        
        # Calculate momentum as cumulative return over lookback period
        recent_returns = returns.iloc[-self.lookback:].values
        momentum = np.prod(1 + recent_returns, axis=0) - 1
        
        # Get top K assets
        top_indices = np.argsort(momentum)[-self.top_k:]
        
        # Allocate weights
        weights = np.zeros(self.n_assets)
        
        if self.method == "equal":
            # Equal weight among top K
            weights[top_indices] = 1.0 / self.top_k
        else:
            # Proportional to momentum (only positive momentum)
            top_momentum = np.maximum(momentum[top_indices], 0)
            
            if np.sum(top_momentum) > 0:
                weights[top_indices] = top_momentum / np.sum(top_momentum)
            else:
                # All negative momentum, use equal weights
                weights[top_indices] = 1.0 / self.top_k
        
        return weights


class BuyAndHoldStrategy(BaseStrategy):
    """
    Buy-and-hold strategy with initial allocation.
    
    Maintains initial weights without rebalancing.
    """
    
    def __init__(
        self,
        n_assets: int,
        initial_weights: Optional[np.ndarray] = None
    ):
        """
        Initialize buy-and-hold strategy.
        
        Args:
            n_assets: Number of assets.
            initial_weights: Initial allocation (equal weight if None).
        """
        super().__init__(n_assets)
        
        if initial_weights is not None:
            self.weights = initial_weights / np.sum(initial_weights)
        else:
            self.weights = np.ones(n_assets) / n_assets
    
    def get_weights(self, **kwargs) -> np.ndarray:
        """
        Get buy-and-hold weights (constant).
        
        Returns:
            Initial weights maintained throughout.
        """
        return self.weights.copy()


def run_all_benchmarks(
    returns: pd.DataFrame,
    transaction_cost: float = 0.001,
    initial_value: float = 100000.0,
    mv_lookback: int = 60,
    momentum_lookback: int = 20,
    momentum_top_k: Optional[int] = None,
) -> dict:
    """
    Run all benchmark strategies.
    
    Args:
        returns: DataFrame of asset returns.
        transaction_cost: Transaction cost per unit turnover.
        initial_value: Initial portfolio value.
        mv_lookback: Lookback for mean-variance optimization.
        momentum_lookback: Lookback for momentum strategy.
        momentum_top_k: Number of top assets for momentum.
        
    Returns:
        Dictionary mapping strategy names to backtest results.
    """
    n_assets = len(returns.columns)
    results = {}
    
    # Equal-Weight
    print("Running Equal-Weight benchmark...")
    eq_strategy = EqualWeightStrategy(n_assets)
    results['Equal-Weight'] = eq_strategy.run_backtest(
        returns, transaction_cost, initial_value
    )
    
    # Mean-Variance (if available)
    if CVXPY_AVAILABLE:
        print("Running Mean-Variance benchmark...")
        try:
            mv_strategy = MeanVarianceStrategy(
                n_assets,
                lookback=mv_lookback,
                allow_short=False
            )
            results['Mean-Variance'] = mv_strategy.run_backtest(
                returns, transaction_cost, initial_value
            )
        except Exception as e:
            print(f"Mean-Variance benchmark failed: {e}")
    
    # Momentum
    print("Running Momentum benchmark...")
    momentum_strategy = MomentumStrategy(
        n_assets,
        lookback=momentum_lookback,
        top_k=momentum_top_k,
        method="proportional"
    )
    results['Momentum'] = momentum_strategy.run_backtest(
        returns, transaction_cost, initial_value
    )
    
    # Buy-and-Hold
    print("Running Buy-and-Hold benchmark...")
    bh_strategy = BuyAndHoldStrategy(n_assets)
    results['Buy-and-Hold'] = bh_strategy.run_backtest(
        returns, transaction_cost, initial_value
    )
    
    return results


if __name__ == "__main__":
    """Example usage of benchmark strategies."""
    
    # Generate sample data
    np.random.seed(42)
    n_periods = 252
    n_assets = 5
    
    # Random returns with some momentum
    returns_data = np.random.randn(n_periods, n_assets) * 0.01
    returns_data[:, 0] += 0.0005  # Asset 0 has positive drift
    returns_data[:, 1] -= 0.0002  # Asset 1 has negative drift
    
    returns = pd.DataFrame(
        returns_data,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    print("Testing benchmark strategies...")
    print(f"Data shape: {returns.shape}")
    
    # Run all benchmarks
    results = run_all_benchmarks(
        returns=returns,
        transaction_cost=0.001,
        initial_value=100000.0,
        mv_lookback=60,
        momentum_lookback=20,
        momentum_top_k=3
    )
    
    # Display results
    print("\n" + "="*60)
    print("Benchmark Results Summary")
    print("="*60)
    
    for name, result in results.items():
        final_value = result['values'][-1]
        total_return = (final_value / 100000.0) - 1
        avg_turnover = np.mean(result['turnover'])
        
        print(f"\n{name}:")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Avg Turnover: {avg_turnover:.4f}")
