"""
Custom OpenAI Gym environment for portfolio optimization.

This environment implements the MDP formulation:
- State: s_t = [p_{t-K:t}, x_t, w_{t-1}] (price history, features, previous weights)
- Action: a_t = w_t (portfolio weights, continuous)
- Reward: r_t = R_t - λ σ̂²_t (risk-adjusted return)

Mathematical Formulations:
- Portfolio return: R_t^{gross} = w_{t-1}^T r_t
- Turnover: Turn_t = sum_i |w_{t,i} - w_{t-1,i}|
- Transaction cost: cost_t = c * Turn_t
- Net return: R_t = R_t^{gross} - cost_t
- Portfolio value: V_t = V_{t-1} * (1 + R_t)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any


class PortfolioEnv(gym.Env):
    """
    Portfolio optimization environment following OpenAI Gym interface.
    
    Attributes:
        prices (pd.DataFrame): Historical price data.
        returns (pd.DataFrame): Asset returns.
        features (pd.DataFrame): Engineered features.
        n_assets (int): Number of assets in portfolio.
        lookback_window (int): Number of historical periods in state.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        features: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        lookback_window: int = 20,
        reward_type: str = "risk_adjusted",
        risk_penalty_lambda: float = 0.5,
        volatility_window: int = 20,
        risk_free_rate: float = 0.02,
        allow_short: bool = False,
        max_leverage: float = 1.0,
        turnover_penalty: float = 0.0,
    ):
        """
        Initialize portfolio environment.
        
        Args:
            prices: DataFrame with historical prices.
            returns: DataFrame with asset returns.
            features: DataFrame with engineered features.
            initial_balance: Starting portfolio value.
            transaction_cost: Cost per unit turnover (e.g., 0.001 = 0.1%).
            lookback_window: Number of historical periods in state.
            reward_type: Type of reward ('risk_adjusted', 'sharpe', 'log_return').
            risk_penalty_lambda: Risk penalty coefficient.
            volatility_window: Window for computing volatility.
            risk_free_rate: Annual risk-free rate.
            allow_short: Whether to allow short positions.
            max_leverage: Maximum leverage (1.0 for long-only).
            turnover_penalty: Additional penalty for high turnover.
        """
        super(PortfolioEnv, self).__init__()
        
        # Store data
        self.prices = prices
        self.returns = returns
        self.features = features
        
        # Environment parameters
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.reward_type = reward_type
        self.risk_penalty_lambda = risk_penalty_lambda
        self.volatility_window = volatility_window
        self.risk_free_rate = risk_free_rate
        self.allow_short = allow_short
        self.max_leverage = max_leverage
        self.turnover_penalty = turnover_penalty
        self.frequency = 'weekly'  # Default, can be overridden
        
        # Portfolio parameters
        self.n_assets = len(prices.columns)
        self.asset_names = prices.columns.tolist()
        
        # Define action space: continuous weights for each asset
        # Action will be raw logits, converted to valid weights via softmax
        if allow_short:
            self.action_space = spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(self.n_assets,),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(self.n_assets,),
                dtype=np.float32
            )
        
        # Define observation space
        # State: [price_history, features, previous_weights, portfolio_metrics]
        n_price_features = self.n_assets * lookback_window
        n_technical_features = len(features.columns)
        n_weight_features = self.n_assets
        n_portfolio_metrics = 3  # recent_return, drawdown, volatility
        
        self.observation_dim = n_price_features + n_technical_features + n_weight_features + n_portfolio_metrics
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = len(returns) - lookback_window - 1
        
        # Portfolio state
        self.portfolio_value = initial_balance
        self.weights = np.ones(self.n_assets) / self.n_assets  # Start with equal weights
        self.cash = 0.0
        
        # History tracking
        self.portfolio_values = []
        self.portfolio_returns = []
        self.weights_history = []
        self.actions_history = []
        self.turnover_history = []
        self.cost_history = []
        
    def _get_state(self) -> np.ndarray:
        """
        Construct the state representation.
        
        State consists of:
        1. Price history: normalized prices for lookback window
        2. Technical features: current values
        3. Previous weights: portfolio allocation
        
        Returns:
            State vector as numpy array.
        """
        # Get current index in data
        data_idx = self.lookback_window + self.current_step
        
        # 1. Price history (last K periods)
        price_window = self.returns.iloc[
            data_idx - self.lookback_window:data_idx
        ].values.flatten()
        
        # Normalize price history (clip extreme values)
        price_window = np.clip(price_window, -0.5, 0.5)
        
        # 2. Current technical features
        current_features = self.features.iloc[data_idx].values
        
        # Normalize features (handle inf/nan)
        current_features = np.nan_to_num(current_features, nan=0.0, posinf=3.0, neginf=-3.0)
        current_features = np.clip(current_features, -5.0, 5.0)
        
        # 3. Previous weights
        prev_weights = self.weights.copy()
        
        # 4. Add portfolio performance metrics
        if len(self.portfolio_values) > 1:
            # Recent return
            recent_return = (self.portfolio_values[-1] - self.portfolio_values[-2]) / self.portfolio_values[-2]
            recent_return = np.clip(recent_return, -1.0, 1.0)
            
            # Drawdown from peak
            peak_value = max(self.portfolio_values)
            current_drawdown = (peak_value - self.portfolio_values[-1]) / peak_value if peak_value > 0 else 0
            
            # Recent volatility
            if len(self.portfolio_returns) >= 10:
                recent_vol = np.std(self.portfolio_returns[-10:])
            else:
                recent_vol = 0.0
            
            portfolio_metrics = np.array([recent_return, current_drawdown, recent_vol], dtype=np.float32)
        else:
            portfolio_metrics = np.zeros(3, dtype=np.float32)
        
        # Concatenate all components
        state = np.concatenate([
            price_window,
            current_features,
            prev_weights,
            portfolio_metrics
        ]).astype(np.float32)
        
        return state
    
    def _softmax_weights(self, logits: np.ndarray) -> np.ndarray:
        """
        Convert raw action logits to valid portfolio weights using softmax.
        Implements dynamic risk management based on current drawdown.
        
        Args:
            logits: Raw output from neural network.
            
        Returns:
            Valid portfolio weights with dynamic cash allocation based on risk.
            
        Formula:
            w_i = exp(z_i) / sum_j exp(z_j)
        """
        # Numerical stability: subtract max
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        weights = exp_logits / np.sum(exp_logits)
        
        # Ensure non-negative for long-only
        if not self.allow_short:
            weights = np.abs(weights)
            weights = weights / np.sum(weights)
        
        # CRITICAL: Dynamic risk management based on drawdown
        # Automatically reduce exposure when experiencing drawdown
        if len(self.portfolio_values) >= 2:
            current_value = self.portfolio_values[-1]
            peak_value = max(self.portfolio_values)
            if peak_value > 0:
                current_drawdown = (peak_value - current_value) / peak_value
                
                # AGGRESSIVE exposure reduction based on drawdown (target <10%)
                if current_drawdown > 0.02:  # 2% drawdown
                    risk_reduction = 0.9  # Reduce to 90% exposure
                elif current_drawdown > 0.04:  # 4% drawdown
                    risk_reduction = 0.7  # Reduce to 70% exposure
                elif current_drawdown > 0.06:  # 6% drawdown
                    risk_reduction = 0.5  # Reduce to 50% exposure
                elif current_drawdown > 0.08:  # 8% drawdown
                    risk_reduction = 0.3  # Reduce to 30% exposure
                elif current_drawdown > 0.10:  # 10% drawdown (defensive mode)
                    risk_reduction = 0.1  # Reduce to 10% exposure
                else:
                    risk_reduction = 1.0  # Full exposure
                
                # Scale down weights, remainder goes to cash
                weights = weights * risk_reduction
        
        # Also check recent volatility for aggressive risk management
        if len(self.portfolio_returns) >= 20:
            recent_vol = np.std(self.portfolio_returns[-20:])
            # If volatility is high, aggressively reduce exposure
            if recent_vol > 0.012:  # Moderate daily volatility
                vol_reduction = 0.8
                weights = weights * vol_reduction
            elif recent_vol > 0.015:  # High daily volatility
                vol_reduction = 0.6
                weights = weights * vol_reduction
            elif recent_vol > 0.020:  # Extreme volatility
                vol_reduction = 0.3
                weights = weights * vol_reduction
        
        return weights
    
    def _project_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Project weights onto feasible set: sum to <= 1, satisfy constraints.
        Allows dynamic cash allocation (remainder = cash position).
        
        Args:
            weights: Raw weights.
            
        Returns:
            Valid weights satisfying all constraints. Sum can be < 1 (cash).
        """
        # Clip weights to [0, 1] for each asset
        if not self.allow_short:
            weights = np.clip(weights, 0, 1)
        
        # Normalize only if sum exceeds max_leverage (typically 1.0)
        weight_sum = np.sum(np.abs(weights))
        if weight_sum > self.max_leverage:
            weights = weights * self.max_leverage / weight_sum
        
        # Otherwise allow sum < 1 (remainder is cash)
        # Cash allocation = 1 - sum(weights)
        # This enables dynamic risk management
        
        return weights
    
    def _calculate_turnover(self, new_weights: np.ndarray) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            new_weights: New portfolio weights.
            
        Returns:
            Turnover value.
            
        Formula:
            Turn_t = sum_i |w_{t,i} - w_{t-1,i}|
        """
        turnover = np.sum(np.abs(new_weights - self.weights))
        return turnover
    
    def _calculate_reward(
        self,
        portfolio_return: float,
        turnover: float
    ) -> float:
        """
        Calculate reward with strong emphasis on risk control and drawdown minimization.
        
        Args:
            portfolio_return: Net portfolio return for this step.
            turnover: Portfolio turnover.
            
        Returns:
            Reward value optimized for high Sharpe ratio and low drawdown.
        """
        # Start with base return
        reward = 0.0
        
        # Calculate current drawdown
        current_drawdown = 0.0
        if len(self.portfolio_values) >= 2:
            current_value = self.portfolio_values[-1]
            peak_value = max(self.portfolio_values)
            if peak_value > 0:
                current_drawdown = (peak_value - current_value) / peak_value
        
        # Calculate recent volatility
        recent_volatility = 0.0
        if len(self.portfolio_returns) >= self.volatility_window:
            recent_returns = self.portfolio_returns[-self.volatility_window:]
            recent_volatility = np.std(recent_returns)
        
        # === CORE REWARD COMPONENTS ===
        
        # 1. Base return reward (scaled appropriately)
        reward += portfolio_return * 1000.0
        
        # 2. Asymmetric return reward (upside/downside asymmetry)
        if portfolio_return > 0:
            # Reward positive returns
            reward += portfolio_return * 500.0
            # Extra bonus for returns above risk-free rate
            excess_return = portfolio_return - (self.risk_free_rate / 252)
            if excess_return > 0:
                reward += excess_return * 300.0
        else:
            # Penalty for negative returns (moderate downside protection)
            reward += portfolio_return * 1500.0  # 1.5x penalty for losses
        
        # 3. CRITICAL: Extremely aggressive drawdown penalties (target < 10%)
        if current_drawdown > 0.01:  # Start penalizing early at 1%
            reward -= 100.0 * (current_drawdown - 0.01)
        if current_drawdown > 0.03:  # Moderate penalty at 3%
            reward -= 500.0 * (current_drawdown - 0.03)
        if current_drawdown > 0.05:  # Strong penalty at 5%
            reward -= 1000.0 * (current_drawdown - 0.05)
        if current_drawdown > 0.08:  # Severe penalty at 8%
            reward -= 3000.0 * (current_drawdown - 0.08)
        if current_drawdown > 0.10:  # Catastrophic penalty at 10%
            reward -= 10000.0 * (current_drawdown - 0.10)
        
        # 4. Reward for staying near peak (low drawdown bonus)
        if current_drawdown < 0.02:
            reward += 100.0 * (0.02 - current_drawdown)  # Scaled bonus
        
        # Additional bonus for maintaining very low drawdown over time
        if len(self.portfolio_values) >= 50:
            # Calculate average drawdown over last 50 periods
            recent_values = self.portfolio_values[-50:]
            avg_drawdown = 0
            for i, val in enumerate(recent_values):
                peak = max(recent_values[:i+1]) if i > 0 else val
                if peak > 0:
                    dd = (peak - val) / peak
                    avg_drawdown += dd
            avg_drawdown /= 50
            
            # Strong reward for maintaining low average drawdown
            if avg_drawdown < 0.05:
                reward += 500.0 * (0.05 - avg_drawdown)
            if avg_drawdown < 0.03:
                reward += 1000.0 * (0.03 - avg_drawdown)
        
        # 5. Volatility control (encourage stability, but not too strict)
        if recent_volatility > 0:
            # Target annual volatility < 15% (daily vol < 0.01)
            if recent_volatility > 0.010:
                reward -= 200.0 * (recent_volatility - 0.010)
        
        # 6. Sharpe-based reward component (STRONG weight on this)
        if len(self.portfolio_returns) >= self.volatility_window:
            recent_returns = self.portfolio_returns[-self.volatility_window:]
            mean_return = np.mean(recent_returns)
            volatility = np.std(recent_returns) + 1e-8
            sharpe = (mean_return - self.risk_free_rate / 252) / volatility
            
            # Strong rewards for high Sharpe ratios
            if sharpe > 0.04:  # Reasonable Sharpe
                reward += 500.0 * sharpe
            if sharpe > 0.06:  # Good Sharpe (>1 annualized)
                reward += 1000.0 * (sharpe - 0.06)
            if sharpe > 0.08:  # Excellent Sharpe
                reward += 2000.0 * (sharpe - 0.08)
        
        # 7. Consistency bonus (reward stable performance)
        if len(self.portfolio_returns) >= 20:
            recent_20 = self.portfolio_returns[-20:]
            positive_ratio = sum(1 for r in recent_20 if r > 0) / 20
            # Reward high win rate
            if positive_ratio >= 0.55:
                reward += 300.0 * positive_ratio
            if positive_ratio >= 0.65:
                reward += 500.0 * (positive_ratio - 0.65)
        
        # 8. Turnover penalty (reduce transaction costs but not too strict)
        if self.turnover_penalty > 0:
            reward -= self.turnover_penalty * turnover
        
        # Moderate penalty for excessive turnover
        if turnover > 0.6:
            reward -= 100.0 * (turnover - 0.6)
        
        # 9. Value growth bonus (encourage compounding returns)
        if len(self.portfolio_values) >= 2:
            value_growth = (self.portfolio_values[-1] - self.initial_balance) / self.initial_balance
            if value_growth > 0.05:  # 5% growth
                reward += 200.0 * value_growth
            if value_growth > 0.15:  # 15% growth
                reward += 500.0 * (value_growth - 0.15)
        
        # 10. Risk-adjusted return over longer horizon (rolling Sharpe)
        if len(self.portfolio_returns) >= 50:
            returns_50 = self.portfolio_returns[-50:]
            cumulative_return = np.prod([1 + r for r in returns_50]) - 1
            volatility_50 = np.std(returns_50) + 1e-8
            risk_adj_return = cumulative_return / (volatility_50 * np.sqrt(50))
            if risk_adj_return > 0:
                reward += 300.0 * risk_adj_return
            if risk_adj_return > 0.5:  # Excellent risk-adjusted performance
                reward += 500.0 * (risk_adj_return - 0.5)
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step within the environment.
        
        Args:
            action: Portfolio weight logits from agent.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Convert action to valid weights
        new_weights = self._softmax_weights(action)
        new_weights = self._project_weights(new_weights)
        
        # Calculate turnover
        turnover = self._calculate_turnover(new_weights)
        
        # Calculate transaction cost
        transaction_cost = self.transaction_cost * turnover
        
        # Get current period returns
        data_idx = self.lookback_window + self.current_step + 1
        period_returns = self.returns.iloc[data_idx].values
        
        # Calculate cash allocation (remainder after asset allocation)
        cash_weight_old = 1.0 - np.sum(self.weights)
        cash_weight_new = 1.0 - np.sum(new_weights)
        
        # Cash return (assume risk-free rate / 52 for weekly, / 252 for daily)
        periods_per_year = 52 if hasattr(self, 'frequency') else 252
        cash_return = self.risk_free_rate / periods_per_year
        
        # Calculate gross portfolio return: w_{t-1}^T * r_t + cash_weight * r_cash
        gross_return = np.dot(self.weights, period_returns) + cash_weight_old * cash_return
        
        # Calculate net return after costs
        net_return = gross_return - transaction_cost
        
        # Update portfolio value: V_t = V_{t-1} * (1 + R_t)
        self.portfolio_value *= (1 + net_return)
        
        # Calculate reward
        reward = self._calculate_reward(net_return, turnover)
        
        # Update weights for next period
        self.weights = new_weights
        
        # Store history
        self.portfolio_values.append(self.portfolio_value)
        self.portfolio_returns.append(net_return)
        self.weights_history.append(new_weights.copy())
        self.actions_history.append(action.copy())
        self.turnover_history.append(turnover)
        self.cost_history.append(transaction_cost)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get next state
        if not terminated:
            next_state = self._get_state()
        else:
            next_state = np.zeros(self.observation_dim, dtype=np.float32)
        
        # Info dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'return': net_return,
            'gross_return': gross_return,
            'turnover': turnover,
            'transaction_cost': transaction_cost,
            'weights': new_weights.copy(),
            'cash_weight': cash_weight_new,  # Track cash allocation
        }
        
        return next_state, reward, terminated, truncated, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options.
            
        Returns:
            Tuple of (initial_observation, info).
        """
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.current_step = 0
        
        # Reset portfolio state
        self.portfolio_value = self.initial_balance
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.cash = 0.0
        
        # Reset history
        self.portfolio_values = [self.initial_balance]
        self.portfolio_returns = []
        self.weights_history = [self.weights.copy()]
        self.actions_history = []
        self.turnover_history = []
        self.cost_history = []
        
        # Get initial state
        initial_state = self._get_state()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy(),
        }
        
        return initial_state, info
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment state.
        
        Args:
            mode: Rendering mode.
        """
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            if len(self.portfolio_returns) > 0:
                print(f"Last Return: {self.portfolio_returns[-1]:.4f}")
                print(f"Last Turnover: {self.turnover_history[-1]:.4f}")
            print(f"Current Weights:")
            for i, (asset, weight) in enumerate(zip(self.asset_names, self.weights)):
                print(f"  {asset}: {weight:.4f}")
    
    def get_portfolio_history(self) -> Dict[str, Any]:
        """
        Get complete portfolio history.
        
        Returns:
            Dictionary with historical data.
        """
        return {
            'values': np.array(self.portfolio_values),
            'returns': np.array(self.portfolio_returns),
            'weights': np.array(self.weights_history),
            'actions': np.array(self.actions_history),
            'turnover': np.array(self.turnover_history),
            'costs': np.array(self.cost_history),
        }


if __name__ == "__main__":
    """Example usage of PortfolioEnv."""
    
    # Create sample data
    np.random.seed(42)
    n_periods = 500
    n_assets = 5
    
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    # Random walk prices
    prices = pd.DataFrame(
        np.cumsum(np.random.randn(n_periods, n_assets) * 0.02, axis=0) + 100,
        index=dates,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Calculate returns
    returns = prices.pct_change().fillna(0)
    
    # Simple features (just returns for this example)
    features = returns.copy()
    features.columns = [f'{col}_RETURN' for col in features.columns]
    
    # Create environment
    env = PortfolioEnv(
        prices=prices,
        returns=returns,
        features=features,
        initial_balance=100000,
        transaction_cost=0.001,
        lookback_window=20
    )
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Max steps: {env.max_steps}")
    
    # Test random episode
    print("\nTesting random episode...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 10:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        
        if steps % 5 == 0:
            env.render()
    
    print(f"\nEpisode finished after {steps} steps")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final portfolio value: ${info['portfolio_value']:,.2f}")
