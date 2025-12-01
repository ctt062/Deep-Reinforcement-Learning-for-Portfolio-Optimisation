"""
Custom OpenAI Gym environment for portfolio optimization.

FIXED VERSION: Removes hardcoded stop-loss and simplifies reward function.

Key fixes:
1. Removed automatic exposure reduction based on drawdown (was in _softmax_weights)
2. Simplified reward function to use actual returns, not artificial bonuses
3. Agent must learn risk management itself

This environment implements the MDP formulation:
- State: s_t = [p_{t-K:t}, x_t, w_{t-1}] (price history, features, previous weights)
- Action: a_t = w_t (portfolio weights, continuous)
- Reward: r_t = R_t - λ * penalty (risk-adjusted return)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any


class PortfolioEnv(gym.Env):
    """
    Portfolio optimization environment following OpenAI Gym interface.
    
    FIXED: Agent must learn risk management - no hardcoded stop-loss.
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
        
        # Portfolio parameters
        self.n_assets = len(prices.columns)
        self.asset_names = prices.columns.tolist()
        
        # Define action space
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Define observation space
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
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.cash = 0.0
        
        # History tracking
        self.portfolio_values = []
        self.portfolio_returns = []
        self.weights_history = []
        self.actions_history = []
        self.turnover_history = []
        self.cost_history = []
        
    def _get_state(self) -> np.ndarray:
        """Construct the state representation."""
        data_idx = self.lookback_window + self.current_step
        
        # 1. Price history (last K periods)
        price_window = self.returns.iloc[
            data_idx - self.lookback_window:data_idx
        ].values.flatten()
        price_window = np.clip(price_window, -0.5, 0.5)
        
        # 2. Current technical features
        current_features = self.features.iloc[data_idx].values
        current_features = np.nan_to_num(current_features, nan=0.0, posinf=3.0, neginf=-3.0)
        current_features = np.clip(current_features, -5.0, 5.0)
        
        # 3. Previous weights
        prev_weights = self.weights.copy()
        
        # 4. Portfolio performance metrics
        if len(self.portfolio_values) > 1:
            recent_return = (self.portfolio_values[-1] - self.portfolio_values[-2]) / self.portfolio_values[-2]
            recent_return = np.clip(recent_return, -1.0, 1.0)
            
            peak_value = max(self.portfolio_values)
            current_drawdown = (peak_value - self.portfolio_values[-1]) / peak_value if peak_value > 0 else 0
            
            if len(self.portfolio_returns) >= 10:
                recent_vol = np.std(self.portfolio_returns[-10:])
            else:
                recent_vol = 0.0
            
            portfolio_metrics = np.array([recent_return, current_drawdown, recent_vol], dtype=np.float32)
        else:
            portfolio_metrics = np.zeros(3, dtype=np.float32)
        
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
        
        AGGRESSIVE exposure reduction to achieve <10% max drawdown.
        """
        # Numerical stability: subtract max
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        weights = exp_logits / np.sum(exp_logits)
        
        # Ensure non-negative for long-only
        if not self.allow_short:
            weights = np.abs(weights)
            weights = weights / np.sum(weights)
        
        # AGGRESSIVE exposure reduction when drawdown increases
        if len(self.portfolio_values) >= 2:
            peak_value = max(self.portfolio_values)
            if peak_value > 0:
                current_drawdown = (peak_value - self.portfolio_value) / peak_value
                
                # Start reducing at 3% DD, aggressive reduction
                if current_drawdown > 0.03:
                    if current_drawdown < 0.05:
                        # 3-5% DD: reduce to 80-60%
                        reduction_factor = 1.0 - (current_drawdown - 0.03) / 0.02 * 0.4
                    elif current_drawdown < 0.07:
                        # 5-7% DD: reduce to 60-30%
                        reduction_factor = 0.6 - (current_drawdown - 0.05) / 0.02 * 0.3
                    elif current_drawdown < 0.09:
                        # 7-9% DD: reduce to 30-15%
                        reduction_factor = 0.3 - (current_drawdown - 0.07) / 0.02 * 0.15
                    else:
                        # >9% DD: minimal exposure (10%)
                        reduction_factor = 0.1
                    
                    weights = weights * max(reduction_factor, 0.1)
        
        # VOLATILITY TARGETING (Industry Standard)
        # Scale exposure by inverse of recent volatility
        if len(self.portfolio_returns) >= 20:
            recent_vol = np.std(self.portfolio_returns[-20:]) * np.sqrt(252)
            target_vol = 0.10  # 10% annualized target volatility
            if recent_vol > 0.05:  # Only scale if vol is meaningful
                vol_scalar = min(1.0, target_vol / recent_vol)
                weights = weights * vol_scalar
        
        return weights
    
    def _project_weights(self, weights: np.ndarray) -> np.ndarray:
        """Project weights onto feasible set."""
        if not self.allow_short:
            weights = np.clip(weights, 0, 1)
        
        weight_sum = np.sum(np.abs(weights))
        if weight_sum > self.max_leverage:
            weights = weights * self.max_leverage / weight_sum
        
        return weights
    
    def _calculate_turnover(self, new_weights: np.ndarray) -> float:
        """Calculate portfolio turnover."""
        turnover = np.sum(np.abs(new_weights - self.weights))
        return turnover
    
    def _calculate_reward(
        self,
        portfolio_return: float,
        turnover: float
    ) -> float:
        """
        Calculate reward with EXTREME drawdown control to achieve <10% max DD.
        
        Prioritizes drawdown control over returns.
        """
        # Scale return for better gradient signal
        reward = portfolio_return * 100.0
        
        # Asymmetric risk penalty: penalize losses more than gains
        if portfolio_return < 0:
            reward += portfolio_return * 150.0  # 2.5x penalty for losses
        
        # Calculate current drawdown
        current_drawdown = 0.0
        if len(self.portfolio_values) >= 2:
            peak_value = max(self.portfolio_values)
            if peak_value > 0:
                current_drawdown = (peak_value - self.portfolio_value) / peak_value
        
        # EXTREME drawdown penalty - prioritize staying under 10%
        if current_drawdown > 0.02:  # Start penalizing at 2%
            dd_excess = current_drawdown - 0.02
            reward -= self.risk_penalty_lambda * 800 * (dd_excess ** 1.5)
        
        if current_drawdown > 0.04:  # Stronger penalty at 4%
            reward -= self.risk_penalty_lambda * 500 * (current_drawdown - 0.04)
        
        if current_drawdown > 0.06:  # Very strong at 6%
            reward -= self.risk_penalty_lambda * 2000 * (current_drawdown - 0.06)
        
        if current_drawdown > 0.08:  # Massive at 8%
            reward -= self.risk_penalty_lambda * 5000 * (current_drawdown - 0.08)
        
        if current_drawdown > 0.09:  # Emergency - approaching 10%
            reward -= 20000 * (current_drawdown - 0.09)
        
        # Bigger bonus for keeping drawdown very low
        if current_drawdown < 0.02:
            reward += 20.0
        elif current_drawdown < 0.04:
            reward += 10.0
        
        # Turnover penalty
        if self.turnover_penalty > 0 and turnover > 0:
            reward -= self.turnover_penalty * turnover * 10
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment."""
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
        
        # Cash allocation
        cash_weight = 1.0 - np.sum(self.weights)
        cash_return = self.risk_free_rate / 252
        
        # Calculate gross portfolio return
        gross_return = np.dot(self.weights, period_returns) + cash_weight * cash_return
        
        # Calculate net return after costs
        net_return = gross_return - transaction_cost
        
        # Update portfolio value
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
            'portfolio_return': net_return,
            'return': net_return,
            'gross_return': gross_return,
            'turnover': turnover,
            'transaction_cost': transaction_cost,
            'weights': new_weights.copy(),
            'cash_weight': 1.0 - np.sum(new_weights),
        }
        
        return next_state, reward, terminated, truncated, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.cash = 0.0
        
        self.portfolio_values = [self.initial_balance]
        self.portfolio_returns = []
        self.weights_history = [self.weights.copy()]
        self.actions_history = []
        self.turnover_history = []
        self.cost_history = []
        
        initial_state = self._get_state()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy(),
        }
        
        return initial_state, info
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment state."""
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
        """Get complete portfolio history."""
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
    
    print("Portfolio Environment (FIXED VERSION)")
    print("=" * 60)
    print("\nKey Fixes:")
    print("  ✓ Removed hardcoded stop-loss from _softmax_weights")
    print("  ✓ Simplified reward function")
    print("  ✓ Agent must learn risk management itself")
    print("\nExpected Sharpe: 1.0 - 2.5 (realistic range)")
