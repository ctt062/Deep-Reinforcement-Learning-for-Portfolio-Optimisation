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
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        self.allow_short = allow_short
        self.max_leverage = max_leverage
        self.turnover_penalty = turnover_penalty
        
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
        # State: [price_history, features, previous_weights]
        n_price_features = self.n_assets * lookback_window
        n_technical_features = len(features.columns)
        n_weight_features = self.n_assets
        
        self.observation_dim = n_price_features + n_technical_features + n_weight_features
        
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
        
        # 2. Current technical features
        current_features = self.features.iloc[data_idx].values
        
        # 3. Previous weights
        prev_weights = self.weights.copy()
        
        # Concatenate all components
        state = np.concatenate([
            price_window,
            current_features,
            prev_weights
        ]).astype(np.float32)
        
        return state
    
    def _softmax_weights(self, logits: np.ndarray) -> np.ndarray:
        """
        Convert raw action logits to valid portfolio weights using softmax.
        
        Args:
            logits: Raw output from neural network.
            
        Returns:
            Valid portfolio weights summing to 1.
            
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
        
        return weights
    
    def _project_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Project weights onto feasible set: sum to 1, satisfy constraints.
        
        Args:
            weights: Raw weights.
            
        Returns:
            Valid weights satisfying all constraints.
        """
        # Clip weights
        if not self.allow_short:
            weights = np.maximum(weights, 0)
        
        # Normalize to sum to 1
        weight_sum = np.sum(np.abs(weights))
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones(self.n_assets) / self.n_assets
        
        # Apply leverage constraint
        if np.sum(np.abs(weights)) > self.max_leverage:
            weights = weights * self.max_leverage / np.sum(np.abs(weights))
        
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
        Calculate reward based on selected reward type with enhanced risk penalties.
        
        Args:
            portfolio_return: Net portfolio return for this step.
            turnover: Portfolio turnover.
            
        Returns:
            Reward value.
        """
        if self.reward_type == "log_return":
            # Simple log return
            reward = np.log(1 + portfolio_return + 1e-8)
            
        elif self.reward_type == "sharpe":
            # Sharpe-like reward: (return - rf) / volatility
            data_idx = self.lookback_window + self.current_step
            
            if len(self.portfolio_returns) >= self.volatility_window:
                recent_returns = self.portfolio_returns[-self.volatility_window:]
                volatility = np.std(recent_returns) + 1e-8
                reward = (portfolio_return - self.risk_free_rate) / volatility
            else:
                reward = portfolio_return - self.risk_free_rate
                
        elif self.reward_type == "risk_adjusted":
            # Risk-adjusted return: R_t - λ * σ²_t
            if len(self.portfolio_returns) >= self.volatility_window:
                recent_returns = self.portfolio_returns[-self.volatility_window:]
                variance = np.var(recent_returns)
                reward = portfolio_return - self.risk_penalty_lambda * variance
            else:
                reward = portfolio_return
        else:
            # Default to simple return
            reward = portfolio_return
        
        # Apply turnover penalty if specified
        if self.turnover_penalty > 0:
            reward -= self.turnover_penalty * turnover
        
        # Enhanced: Add progressive penalty for drawdowns (balanced approach)
        if len(self.portfolio_values) >= 2:
            current_value = self.portfolio_values[-1]
            peak_value = max(self.portfolio_values)
            if peak_value > 0:
                current_drawdown = (peak_value - current_value) / peak_value
                # Moderate penalty for drawdowns > 5% (target threshold)
                if current_drawdown > 0.05:
                    drawdown_penalty = 5.0 * (current_drawdown - 0.05)
                    reward -= drawdown_penalty
                # Stronger penalty for drawdowns > 10%
                if current_drawdown > 0.10:
                    reward -= 10.0 * (current_drawdown - 0.10)
        
        # Moderate penalty for negative returns (downside protection)
        if portfolio_return < 0:
            reward -= 0.5 * abs(portfolio_return)
        
        # Small bonus for positive returns (encourage gains)
        if portfolio_return > 0:
            reward += 0.2 * portfolio_return
        
        # Bonus for consistent positive performance
        if len(self.portfolio_returns) >= 10:
            recent_10 = self.portfolio_returns[-10:]
            positive_ratio = sum(1 for r in recent_10 if r > 0) / 10
            if positive_ratio >= 0.7:
                reward += 0.3 * positive_ratio
        
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
        
        # Calculate gross portfolio return: w_{t-1}^T * r_t
        gross_return = np.dot(self.weights, period_returns)
        
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
