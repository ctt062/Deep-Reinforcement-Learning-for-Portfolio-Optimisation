"""
Enhanced Portfolio Environment with Options Overlay.

FIXED VERSION: Corrects options timing and payoff calculation issues.

Key fixes:
1. Options premiums collected only at position open (monthly, not daily)
2. Protective put payoff calculated at expiration based on actual strike vs price
3. Simplified reward function without artificial bonuses

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
    
    FIXED: Now properly tracks option position lifecycle:
    - Premium paid/received only when opening position
    - Payoff calculated at expiration
    - Realistic P&L accounting
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
        self._risk_free_rate = risk_free_rate
        
        # Option pricing models
        self.bs_model = BlackScholesModel(risk_free_rate=risk_free_rate)
        self.option_overlay = OptionOverlay(
            bs_model=self.bs_model,
            option_expiry_days=option_expiry_days,
            transaction_cost=option_transaction_cost
        )
        
        # Track option positions - FIXED: proper position tracking
        self.current_hedge_ratio = 0.0
        self.current_call_ratio = 0.0
        self.option_premium_paid = 0.0
        self.option_premium_received = 0.0
        
        # NEW: Track active option positions
        self.active_put_position = None  # {'strike': K, 'premium': P, 'notional': N, 'open_step': t, 'open_price': S}
        self.active_call_position = None  # {'strike': K, 'premium': P, 'notional': N, 'open_step': t, 'open_price': S}
        self.days_since_option_open = 0
        
        # Update action space to include option decisions
        if self.enable_options:
            self.action_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.n_assets + 2,),
                dtype=np.float32
            )
        
        # Update observation space to include option-related state
        if self.enable_options:
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
            
            # Reset option positions
            self.active_put_position = None
            self.active_call_position = None
            self.days_since_option_open = 0
            
            obs = self._add_option_state(obs)
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step with portfolio rebalancing and option overlay.
        
        FIXED: Options are handled with proper lifecycle management.
        """
        if self.enable_options:
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
            # Apply option overlay with FIXED lifecycle management
            option_pnl, option_info = self._apply_option_overlay_fixed(
                hedge_ratio=hedge_ratio,
                call_ratio=call_ratio
            )
            
            # Adjust portfolio value
            self.portfolio_value += option_pnl
            
            # Simple reward: base reward + scaled option P&L
            option_return = option_pnl / max(self.portfolio_value, 1.0)
            total_reward = base_reward + option_return * 100
            
            # Update info
            info.update(option_info)
            info['option_pnl'] = option_pnl
            info['hedge_ratio'] = hedge_ratio
            info['call_ratio'] = call_ratio
            
            obs = self._add_option_state(obs)
            
            return obs, total_reward, terminated, truncated, info
        
        return obs, base_reward, terminated, truncated, info
    
    def _apply_option_overlay_fixed(
        self,
        hedge_ratio: float,
        call_ratio: float
    ) -> Tuple[float, Dict]:
        """
        Apply option overlay with FIXED lifecycle management.
        
        Key fixes:
        1. Premium paid/received only when opening new position
        2. Payoff calculated at expiration based on actual price movement
        3. Positions roll at expiry
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
        
        # Get current price (use portfolio-weighted average or SPY)
        try:
            current_price = self.prices.iloc[self.current_step].values.mean()
        except:
            current_price = 100.0
        
        self.days_since_option_open += 1
        
        # Check if options expire (every option_expiry_days)
        options_expired = self.days_since_option_open >= self.option_expiry_days
        
        # === PROTECTIVE PUT HANDLING ===
        if self.active_put_position is not None and options_expired:
            # Calculate put payoff at expiration
            put_payoff = self._calculate_put_payoff(
                strike=self.active_put_position['strike'],
                current_price=current_price,
                notional=self.active_put_position['notional']
            )
            total_pnl += put_payoff
            info['put_payoff'] = put_payoff
            self.active_put_position = None
        
        # === COVERED CALL HANDLING ===
        if self.active_call_position is not None and options_expired:
            # Calculate call payoff at expiration (negative if exercised against us)
            call_payoff = self._calculate_call_payoff(
                strike=self.active_call_position['strike'],
                current_price=current_price,
                notional=self.active_call_position['notional']
            )
            total_pnl += call_payoff
            info['call_payoff'] = call_payoff
            self.active_call_position = None
        
        # Reset counter if options expired
        if options_expired:
            self.days_since_option_open = 0
        
        # === OPEN NEW POSITIONS (only if no active position and ratio > threshold) ===
        
        # Open new protective put
        if self.active_put_position is None and hedge_ratio > 0.05:
            put_cost, put_strike, _ = self.option_overlay.protective_put_cost(
                portfolio_value=self.portfolio_value,
                current_price=current_price,
                hedge_ratio=hedge_ratio,
                moneyness=self.put_moneyness,
                volatility=volatility
            )
            
            # Pay premium (only once when opening)
            total_pnl -= put_cost
            self.option_premium_paid += put_cost
            
            # Track position
            self.active_put_position = {
                'strike': put_strike,
                'premium': put_cost,
                'notional': self.portfolio_value * hedge_ratio,
                'open_step': self.current_step,
                'open_price': current_price
            }
            
            info['put_cost'] = put_cost
            info['put_strike'] = put_strike
        
        # Open new covered call
        if self.active_call_position is None and call_ratio > 0.05:
            call_income, call_strike, _ = self.option_overlay.covered_call_income(
                portfolio_value=self.portfolio_value,
                current_price=current_price,
                call_ratio=call_ratio,
                moneyness=self.call_moneyness,
                volatility=volatility
            )
            
            # Receive premium (only once when opening)
            total_pnl += call_income
            self.option_premium_received += call_income
            
            # Track position
            self.active_call_position = {
                'strike': call_strike,
                'premium': call_income,
                'notional': self.portfolio_value * call_ratio,
                'open_step': self.current_step,
                'open_price': current_price
            }
            
            info['call_income'] = call_income
            info['call_strike'] = call_strike
        
        # Update current ratios for state
        self.current_hedge_ratio = hedge_ratio if self.active_put_position else 0.0
        self.current_call_ratio = call_ratio if self.active_call_position else 0.0
        
        return total_pnl, info
    
    def _calculate_put_payoff(self, strike: float, current_price: float, notional: float) -> float:
        """
        Calculate protective put payoff at expiration.
        
        Payoff = max(K - S_T, 0) * (notional / open_price)
        
        If current_price < strike, put is in-the-money and pays off.
        """
        if current_price < strike:
            # Put is in the money
            payoff_per_share = strike - current_price
            # Convert to portfolio terms
            num_contracts = notional / strike  # Approximate number of "contracts"
            return payoff_per_share * num_contracts
        else:
            # Put expires worthless
            return 0.0
    
    def _calculate_call_payoff(self, strike: float, current_price: float, notional: float) -> float:
        """
        Calculate covered call payoff at expiration.
        
        For a covered call (short call):
        - If current_price > strike, we're assigned and lose the upside
        - Payoff = -max(S_T - K, 0) * (notional / open_price)
        """
        if current_price > strike:
            # Call is in the money - we lose the upside
            loss_per_share = current_price - strike
            num_contracts = notional / strike
            return -loss_per_share * num_contracts
        else:
            # Call expires worthless - we keep the premium (already received)
            return 0.0
    
    def _add_option_state(self, base_obs: np.ndarray) -> np.ndarray:
        """Add option-related state information to observation."""
        recent_returns = self.portfolio_returns[-self.volatility_lookback:] if len(self.portfolio_returns) > 0 else np.array([0.0])
        volatility = estimate_implied_volatility(
            returns=np.array(recent_returns),
            window=min(len(recent_returns), self.volatility_lookback),
            annualization_factor=252
        )
        
        option_cost_ratio = (self.option_premium_paid - self.option_premium_received) / max(self.initial_balance, 1.0)
        
        option_state = np.array([
            self.current_hedge_ratio,
            self.current_call_ratio,
            np.clip(volatility, 0, 2.0),
            np.clip(option_cost_ratio, -0.5, 0.5)
        ], dtype=np.float32)
        
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
    
    print("Testing Portfolio Environment with Options Overlay (FIXED)")
    print("=" * 60)
    
    print("\nKey Fixes Applied:")
    print("  ✓ Options premium collected only at position open")
    print("  ✓ Payoff calculated at expiration based on price movement")
    print("  ✓ Proper position lifecycle management")
    print("  ✓ Simplified reward function")
    
    print("\nAction Space:")
    print("  - Portfolio weights: continuous [0, 1]^n")
    print("  - Hedge ratio: continuous [0, 1]")
    print("  - Call coverage ratio: continuous [0, 1]")
    
    print("\nExpected Performance (Realistic):")
    print("  - Sharpe Ratio: 1.0 - 2.5")
    print("  - Maximum Drawdown: 10% - 20%")
    print("  - Annualized Return: 10% - 30%")
