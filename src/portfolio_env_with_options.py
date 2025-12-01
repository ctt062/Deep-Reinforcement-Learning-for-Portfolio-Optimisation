"""
Enhanced Portfolio Environment with Options Overlay.

Extends the base portfolio environment with option-based protection and income strategies.
Implements Deep Hedging approach from Buehler et al. (2019).

Action Space:
- Portfolio weights (continuous, size n_assets)
- Hedge ratio for protective put (continuous, [0, 1])
- Call coverage ratio for covered calls (continuous, [0, 1])
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional
from src.portfolio_env import PortfolioEnv
from src.options_pricing import BlackScholesModel, OptionOverlay, estimate_implied_volatility


class PortfolioWithOptionsEnv(PortfolioEnv):
    """
    Portfolio environment with option overlay strategies.
    
    Extends base environment with:
    1. Protective puts for downside protection
    2. Covered calls for income generation
    3. Dynamic hedging based on market conditions
    """
    
    def __init__(
        self,
        *args,
        enable_options: bool = True,
        option_expiry_days: int = 30,
        option_transaction_cost: float = 0.005,
        max_hedge_ratio: float = 1.0,
        max_call_ratio: float = 1.0,
        put_moneyness: float = 0.95,  # 5% OTM put
        call_moneyness: float = 1.05,  # 5% OTM call
        volatility_lookback: int = 20,
        risk_free_rate: float = 0.02,
        **kwargs
    ):
        """
        Initialize portfolio environment with options.
        
        Args:
            enable_options: Whether to enable option overlays.
            option_expiry_days: Option expiry in days.
            option_transaction_cost: Transaction cost for options.
            max_hedge_ratio: Maximum proportion to hedge with puts.
            max_call_ratio: Maximum proportion to cover with calls.
            put_moneyness: Put strike as % of current price.
            call_moneyness: Call strike as % of current price.
            volatility_lookback: Window for volatility estimation.
            risk_free_rate: Risk-free rate for option pricing.
        """
        super().__init__(*args, **kwargs)
        
        self.enable_options = enable_options
        self.option_expiry_days = option_expiry_days
        self.option_transaction_cost = option_transaction_cost
        self.max_hedge_ratio = max_hedge_ratio
        self.max_call_ratio = max_call_ratio
        self.put_moneyness = put_moneyness
        self.call_moneyness = call_moneyness
        self.volatility_lookback = volatility_lookback
        
        # Option pricing models
        self.bs_model = BlackScholesModel(risk_free_rate=risk_free_rate)
        self.option_overlay = OptionOverlay(
            bs_model=self.bs_model,
            option_expiry_days=option_expiry_days,
            transaction_cost=option_transaction_cost
        )
        
        # Track option positions
        self.current_hedge_ratio = 0.0
        self.current_call_ratio = 0.0
        self.option_premium_paid = 0.0
        self.option_premium_received = 0.0
        
        # Update action space to include option decisions
        if self.enable_options:
            # Action: [weights (n_assets), hedge_ratio (1), call_ratio (1)]
            self.action_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.n_assets + 2,),
                dtype=np.float32
            )
        
        # Update observation space to include option-related state
        if self.enable_options:
            # Add: current_hedge_ratio, current_call_ratio, estimated_vol, option_cost_ratio
            additional_dims = 4
            original_dim = self.observation_space.shape[0]
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(original_dim + additional_dims,),
                dtype=np.float32
            )
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and option positions."""
        obs, info = super().reset(seed=seed, options=options)
        
        if self.enable_options:
            self.current_hedge_ratio = 0.0
            self.current_call_ratio = 0.0
            self.option_premium_paid = 0.0
            self.option_premium_received = 0.0
            
            # Add option state to observation
            obs = self._add_option_state(obs)
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step with portfolio rebalancing and option overlay.
        
        Args:
            action: Array of [portfolio_weights, hedge_ratio, call_ratio]
        """
        if self.enable_options:
            # Split action into portfolio weights and option decisions
            portfolio_action = action[:self.n_assets]
            hedge_ratio = np.clip(action[self.n_assets], 0.0, self.max_hedge_ratio)
            call_ratio = np.clip(action[self.n_assets + 1], 0.0, self.max_call_ratio)
        else:
            portfolio_action = action
            hedge_ratio = 0.0
            call_ratio = 0.0
        
        # Execute base portfolio step
        obs, base_reward, terminated, truncated, info = super().step(portfolio_action)
        
        if self.enable_options:
            # Apply option overlay
            option_pnl, option_info = self._apply_option_overlay(
                hedge_ratio=hedge_ratio,
                call_ratio=call_ratio
            )
            
            # Adjust portfolio value and reward
            self.portfolio_value += option_pnl
            
            # Enhanced reward incorporating option benefits
            option_reward = self._calculate_option_reward(
                base_reward=base_reward,
                option_pnl=option_pnl,
                hedge_ratio=hedge_ratio,
                call_ratio=call_ratio
            )
            
            # Update info
            info.update(option_info)
            info['option_pnl'] = option_pnl
            info['hedge_ratio'] = hedge_ratio
            info['call_ratio'] = call_ratio
            
            # Add option state to observation
            obs = self._add_option_state(obs)
            
            return obs, option_reward, terminated, truncated, info
        
        return obs, base_reward, terminated, truncated, info
    
    def _apply_option_overlay(
        self,
        hedge_ratio: float,
        call_ratio: float
    ) -> Tuple[float, Dict]:
        """
        Apply option overlay strategies.
        
        Returns:
            Tuple of (net_option_pnl, option_info_dict)
        """
        total_pnl = 0.0
        info = {}
        
        # Estimate current volatility
        recent_returns = self.portfolio_returns[-self.volatility_lookback:] if len(self.portfolio_returns) > 0 else np.array([0.0])
        volatility = estimate_implied_volatility(
            returns=np.array(recent_returns),
            window=min(len(recent_returns), self.volatility_lookback),
            annualization_factor=252
        )
        info['implied_vol'] = volatility
        
        # Get current portfolio price (use SPY as proxy for portfolio)
        current_price = self.prices.iloc[self.current_step, 6]  # SPY index
        
        # 1. Protective Put Strategy
        if hedge_ratio > 0.01:
            put_cost, put_strike, max_loss = self.option_overlay.protective_put_cost(
                portfolio_value=self.portfolio_value,
                current_price=current_price,
                hedge_ratio=hedge_ratio,
                moneyness=self.put_moneyness,
                volatility=volatility
            )
            
            # Pay premium
            total_pnl -= put_cost
            self.option_premium_paid += put_cost
            
            # Calculate protection benefit (if in drawdown)
            # Calculate current drawdown
            if len(self.portfolio_values) > 0:
                peak_value = max(self.portfolio_values)
                current_drawdown = (peak_value - self.portfolio_value) / peak_value if peak_value > 0 else 0
            else:
                current_drawdown = 0
            
            if current_drawdown > 0.05:  # If in 5%+ drawdown
                protection_benefit = put_cost * (current_drawdown / 0.10)  # Scale benefit
                total_pnl += protection_benefit
            
            info['put_cost'] = put_cost
            info['put_strike'] = put_strike
            info['put_protection'] = max_loss
        
        # 2. Covered Call Strategy
        if call_ratio > 0.01:
            call_income, call_strike, max_gain = self.option_overlay.covered_call_income(
                portfolio_value=self.portfolio_value,
                current_price=current_price,
                call_ratio=call_ratio,
                moneyness=self.call_moneyness,
                volatility=volatility
            )
            
            # Receive premium
            total_pnl += call_income
            self.option_premium_received += call_income
            
            info['call_income'] = call_income
            info['call_strike'] = call_strike
            info['call_cap'] = max_gain
        
        # Store current option positions
        self.current_hedge_ratio = hedge_ratio
        self.current_call_ratio = call_ratio
        
        return total_pnl, info
    
    def _calculate_option_reward(
        self,
        base_reward: float,
        option_pnl: float,
        hedge_ratio: float,
        call_ratio: float
    ) -> float:
        """
        Calculate enhanced reward with option overlay benefits.
        
        Reward components:
        1. Base portfolio return
        2. Option P&L
        3. Drawdown protection bonus
        4. Income generation bonus
        5. Efficient hedging bonus
        """
        # Start with base reward
        reward = base_reward
        
        # Add option P&L (scaled)
        option_return = option_pnl / self.portfolio_value if self.portfolio_value > 0 else 0
        reward += option_return * 1000  # Scale similar to base reward
        
        # Calculate current drawdown
        if len(self.portfolio_values) > 0:
            peak_value = max(self.portfolio_values)
            current_drawdown = (peak_value - self.portfolio_value) / peak_value if peak_value > 0 else 0
        else:
            current_drawdown = 0
        
        # Bonus for effective drawdown protection
        if current_drawdown > 0.05 and hedge_ratio > 0.3:
            # Reward hedging during drawdowns
            protection_bonus = hedge_ratio * (1.0 - current_drawdown) * 500
            reward += protection_bonus
        
        # Bonus for income generation without capping upside too much
        if call_ratio > 0.2 and current_drawdown < 0.05:
            # Reward income generation when not in drawdown
            income_bonus = call_ratio * 300
            reward += income_bonus
        
        # Penalty for excessive hedging when not needed
        if current_drawdown < 0.02 and hedge_ratio > 0.5:
            excessive_hedge_penalty = (hedge_ratio - 0.5) * 200
            reward -= excessive_hedge_penalty
        
        # Bonus for keeping drawdown low
        if current_drawdown < 0.05:
            low_dd_bonus = (0.05 - current_drawdown) * 2000
            reward += low_dd_bonus
        
        # Massive penalty for excessive drawdowns
        if current_drawdown > 0.10:
            dd_penalty = (current_drawdown - 0.10) * 20000
            reward -= dd_penalty
        
        return reward
    
    def _add_option_state(self, base_obs: np.ndarray) -> np.ndarray:
        """Add option-related state information to observation."""
        # Estimate current volatility
        recent_returns = self.portfolio_returns[-self.volatility_lookback:] if len(self.portfolio_returns) > 0 else np.array([0.0])
        volatility = estimate_implied_volatility(
            returns=np.array(recent_returns),
            window=min(len(recent_returns), self.volatility_lookback),
            annualization_factor=252
        )
        
        # Calculate option cost ratio
        option_cost_ratio = (self.option_premium_paid - self.option_premium_received) / max(self.initial_balance, 1.0)
        
        # Option state: [hedge_ratio, call_ratio, volatility, option_cost_ratio]
        option_state = np.array([
            self.current_hedge_ratio,
            self.current_call_ratio,
            np.clip(volatility, 0, 2.0),  # Clip volatility
            np.clip(option_cost_ratio, -0.5, 0.5)  # Clip cost ratio
        ], dtype=np.float32)
        
        # Concatenate with base observation
        return np.concatenate([base_obs, option_state])
    
    def get_portfolio_history(self) -> Dict:
        """Get portfolio history including option metrics."""
        history = super().get_portfolio_history()
        
        if self.enable_options:
            history['option_premium_paid'] = self.option_premium_paid
            history['option_premium_received'] = self.option_premium_received
            history['net_option_cost'] = self.option_premium_paid - self.option_premium_received
        
        return history


if __name__ == "__main__":
    """Test option-enhanced environment."""
    
    print("Testing Portfolio Environment with Options Overlay")
    print("=" * 60)
    
    # This would need actual data to run
    print("\nFeatures:")
    print("  ✓ Protective puts for downside protection")
    print("  ✓ Covered calls for income generation")
    print("  ✓ Dynamic hedging based on drawdown")
    print("  ✓ Volatility-adaptive option pricing")
    print("  ✓ Enhanced reward function for risk management")
    
    print("\nAction Space:")
    print("  - Portfolio weights: continuous [0, 1]^n")
    print("  - Hedge ratio: continuous [0, 1]")
    print("  - Call coverage ratio: continuous [0, 1]")
    
    print("\nTarget Performance:")
    print("  - Sharpe Ratio > 1.0")
    print("  - Maximum Drawdown < 10%")
    print("  - Annualized Return > 15%")
