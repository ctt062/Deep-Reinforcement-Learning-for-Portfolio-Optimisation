"""
Options pricing module for portfolio protection and income generation.

Implements Black-Scholes and other models for option valuation in the
portfolio optimization framework.

References:
- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
- Hull, J. C. (2017). Options, Futures, and Other Derivatives.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional


class BlackScholesModel:
    """
    Black-Scholes option pricing model.
    
    Assumptions:
    - Constant volatility and interest rate
    - No dividends (or continuous dividend yield)
    - Efficient markets
    - Log-normal price distribution
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        dividend_yield: float = 0.0
    ):
        """
        Initialize Black-Scholes model.
        
        Args:
            risk_free_rate: Annual risk-free rate.
            dividend_yield: Annual continuous dividend yield.
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
    
    def d1(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """Calculate d1 parameter."""
        return (np.log(S / K) + (self.risk_free_rate - self.dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    def d2(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """Calculate d2 parameter."""
        return self.d1(S, K, T, sigma) - sigma * np.sqrt(T)
    
    def call_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """
        Calculate European call option price.
        
        Args:
            S: Current asset price.
            K: Strike price.
            T: Time to expiration (years).
            sigma: Volatility (annual).
            
        Returns:
            Call option price.
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1 = self.d1(S, K, T, sigma)
        d2 = self.d2(S, K, T, sigma)
        
        call = (S * np.exp(-self.dividend_yield * T) * norm.cdf(d1) - 
                K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2))
        
        return max(call, 0)
    
    def put_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """
        Calculate European put option price.
        
        Args:
            S: Current asset price.
            K: Strike price.
            T: Time to expiration (years).
            sigma: Volatility (annual).
            
        Returns:
            Put option price.
        """
        if T <= 0:
            return max(K - S, 0)
        
        d1 = self.d1(S, K, T, sigma)
        d2 = self.d2(S, K, T, sigma)
        
        put = (K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2) - 
               S * np.exp(-self.dividend_yield * T) * norm.cdf(-d1))
        
        return max(put, 0)
    
    def call_delta(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """Calculate call option delta (sensitivity to price change)."""
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        d1 = self.d1(S, K, T, sigma)
        return np.exp(-self.dividend_yield * T) * norm.cdf(d1)
    
    def put_delta(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """Calculate put option delta."""
        if T <= 0:
            return -1.0 if S < K else 0.0
        
        d1 = self.d1(S, K, T, sigma)
        return -np.exp(-self.dividend_yield * T) * norm.cdf(-d1)
    
    def gamma(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """Calculate option gamma (rate of change of delta)."""
        if T <= 0:
            return 0.0
        
        d1 = self.d1(S, K, T, sigma)
        return (np.exp(-self.dividend_yield * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    
    def vega(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """Calculate option vega (sensitivity to volatility)."""
        if T <= 0:
            return 0.0
        
        d1 = self.d1(S, K, T, sigma)
        return S * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * np.sqrt(T) / 100
    
    def theta(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """Calculate option theta (time decay)."""
        if T <= 0:
            return 0.0
        
        d1 = self.d1(S, K, T, sigma)
        d2 = self.d2(S, K, T, sigma)
        
        if option_type.lower() == 'call':
            theta = (-(S * norm.pdf(d1) * sigma * np.exp(-self.dividend_yield * T)) / (2 * np.sqrt(T)) - 
                    self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2) + 
                    self.dividend_yield * S * np.exp(-self.dividend_yield * T) * norm.cdf(d1))
        else:  # put
            theta = (-(S * norm.pdf(d1) * sigma * np.exp(-self.dividend_yield * T)) / (2 * np.sqrt(T)) + 
                    self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2) - 
                    self.dividend_yield * S * np.exp(-self.dividend_yield * T) * norm.cdf(-d1))
        
        return theta / 365  # Daily theta


class OptionOverlay:
    """
    Option overlay strategies for portfolio protection and income generation.
    
    Strategies:
    1. Protective Put: Buy puts to limit downside
    2. Covered Call: Sell calls to generate income
    3. Collar: Buy put + sell call (zero cost or low cost protection)
    """
    
    def __init__(
        self,
        bs_model: Optional[BlackScholesModel] = None,
        option_expiry_days: int = 30,
        transaction_cost: float = 0.005,
    ):
        """
        Initialize option overlay.
        
        Args:
            bs_model: Black-Scholes model for pricing.
            option_expiry_days: Default option expiry in days.
            transaction_cost: Transaction cost for options (% of premium).
        """
        self.bs_model = bs_model if bs_model is not None else BlackScholesModel()
        self.option_expiry_days = option_expiry_days
        self.transaction_cost = transaction_cost
    
    def protective_put_cost(
        self,
        portfolio_value: float,
        current_price: float,
        hedge_ratio: float,
        moneyness: float = 0.95,
        volatility: float = 0.20
    ) -> Tuple[float, float, float]:
        """
        Calculate cost and protection of protective put.
        
        Args:
            portfolio_value: Current portfolio value.
            current_price: Current price of underlying.
            hedge_ratio: Proportion of portfolio to hedge (0-1).
            moneyness: Strike as % of current price (e.g., 0.95 = 5% OTM).
            volatility: Implied volatility.
            
        Returns:
            Tuple of (premium_paid, strike_price, max_loss).
        """
        T = self.option_expiry_days / 365.0
        strike = current_price * moneyness
        
        # Calculate put premium
        put_premium = self.bs_model.put_price(
            S=current_price,
            K=strike,
            T=T,
            sigma=volatility
        )
        
        # Cost for hedging
        notional = portfolio_value * hedge_ratio
        num_contracts = notional / current_price
        total_premium = put_premium * num_contracts * (1 + self.transaction_cost)
        
        # Maximum loss with protection
        max_loss = max(current_price - strike, 0) * num_contracts + total_premium
        
        return total_premium, strike, max_loss
    
    def covered_call_income(
        self,
        portfolio_value: float,
        current_price: float,
        call_ratio: float,
        moneyness: float = 1.05,
        volatility: float = 0.20
    ) -> Tuple[float, float, float]:
        """
        Calculate income from covered call.
        
        Args:
            portfolio_value: Current portfolio value.
            current_price: Current price of underlying.
            call_ratio: Proportion of holdings to cover (0-1).
            moneyness: Strike as % of current price (e.g., 1.05 = 5% OTM).
            volatility: Implied volatility.
            
        Returns:
            Tuple of (premium_received, strike_price, max_gain).
        """
        T = self.option_expiry_days / 365.0
        strike = current_price * moneyness
        
        # Calculate call premium
        call_premium = self.bs_model.call_price(
            S=current_price,
            K=strike,
            T=T,
            sigma=volatility
        )
        
        # Income from selling calls
        notional = portfolio_value * call_ratio
        num_contracts = notional / current_price
        total_premium = call_premium * num_contracts * (1 - self.transaction_cost)
        
        # Maximum gain (capped at strike)
        max_gain = (strike - current_price) * num_contracts + total_premium
        
        return total_premium, strike, max_gain
    
    def collar_strategy(
        self,
        portfolio_value: float,
        current_price: float,
        hedge_ratio: float,
        put_moneyness: float = 0.95,
        call_moneyness: float = 1.05,
        volatility: float = 0.20
    ) -> Tuple[float, float, float, float]:
        """
        Calculate cost/income of collar strategy (buy put + sell call).
        
        Args:
            portfolio_value: Current portfolio value.
            current_price: Current price of underlying.
            hedge_ratio: Proportion to collar (0-1).
            put_moneyness: Put strike as % of price.
            call_moneyness: Call strike as % of price.
            volatility: Implied volatility.
            
        Returns:
            Tuple of (net_cost, put_strike, call_strike, protected_range).
        """
        T = self.option_expiry_days / 365.0
        
        put_strike = current_price * put_moneyness
        call_strike = current_price * call_moneyness
        
        # Put premium (cost)
        put_premium = self.bs_model.put_price(
            S=current_price,
            K=put_strike,
            T=T,
            sigma=volatility
        )
        
        # Call premium (income)
        call_premium = self.bs_model.call_price(
            S=current_price,
            K=call_strike,
            T=T,
            sigma=volatility
        )
        
        # Net cost (can be negative if call premium > put premium)
        notional = portfolio_value * hedge_ratio
        num_contracts = notional / current_price
        
        net_cost = (put_premium - call_premium) * num_contracts
        net_cost *= (1 + self.transaction_cost) if net_cost > 0 else (1 - self.transaction_cost)
        
        # Protected range
        protected_range = call_strike - put_strike
        
        return net_cost, put_strike, call_strike, protected_range


def estimate_implied_volatility(
    returns: np.ndarray,
    window: int = 20,
    annualization_factor: int = 252
) -> float:
    """
    Estimate implied volatility from historical returns.
    
    Args:
        returns: Array of returns.
        window: Lookback window for volatility calculation.
        annualization_factor: Factor to annualize (252 for daily).
        
    Returns:
        Annualized volatility estimate.
    """
    if len(returns) < window:
        window = len(returns)
    
    recent_returns = returns[-window:]
    volatility = np.std(recent_returns) * np.sqrt(annualization_factor)
    
    # Bound volatility to reasonable range
    volatility = np.clip(volatility, 0.10, 2.0)
    
    return volatility


if __name__ == "__main__":
    """Example usage of option pricing."""
    
    print("Black-Scholes Option Pricing Examples")
    print("=" * 60)
    
    bs = BlackScholesModel(risk_free_rate=0.02, dividend_yield=0.0)
    
    S = 100  # Current price
    K = 100  # Strike price
    T = 30/365  # 30 days
    sigma = 0.25  # 25% volatility
    
    call = bs.call_price(S, K, T, sigma)
    put = bs.put_price(S, K, T, sigma)
    
    print(f"\nATM Option Prices (S={S}, K={K}, T={30} days):")
    print(f"  Call: ${call:.2f}")
    print(f"  Put:  ${put:.2f}")
    
    print(f"\nGreeks:")
    print(f"  Call Delta: {bs.call_delta(S, K, T, sigma):.4f}")
    print(f"  Put Delta:  {bs.put_delta(S, K, T, sigma):.4f}")
    print(f"  Gamma:      {bs.gamma(S, K, T, sigma):.4f}")
    print(f"  Vega:       {bs.vega(S, K, T, sigma):.4f}")
    
    print("\n" + "=" * 60)
    print("Option Overlay Strategy Examples")
    print("=" * 60)
    
    overlay = OptionOverlay()
    portfolio_value = 100000
    
    # Protective Put
    put_cost, put_strike, max_loss = overlay.protective_put_cost(
        portfolio_value=portfolio_value,
        current_price=S,
        hedge_ratio=0.5,
        moneyness=0.95,
        volatility=sigma
    )
    
    print(f"\nProtective Put (50% hedge, 5% OTM):")
    print(f"  Premium Cost: ${put_cost:.2f}")
    print(f"  Strike:       ${put_strike:.2f}")
    print(f"  Max Loss:     ${max_loss:.2f}")
    print(f"  Cost as %:    {put_cost/portfolio_value*100:.3f}%")
    
    # Covered Call
    call_income, call_strike, max_gain = overlay.covered_call_income(
        portfolio_value=portfolio_value,
        current_price=S,
        call_ratio=0.5,
        moneyness=1.05,
        volatility=sigma
    )
    
    print(f"\nCovered Call (50% coverage, 5% OTM):")
    print(f"  Premium Income: ${call_income:.2f}")
    print(f"  Strike:         ${call_strike:.2f}")
    print(f"  Max Gain:       ${max_gain:.2f}")
    print(f"  Income as %:    {call_income/portfolio_value*100:.3f}%")
    
    # Collar
    collar_cost, put_s, call_s, prot_range = overlay.collar_strategy(
        portfolio_value=portfolio_value,
        current_price=S,
        hedge_ratio=0.5,
        put_moneyness=0.95,
        call_moneyness=1.05,
        volatility=sigma
    )
    
    print(f"\nCollar Strategy (50% hedge, 5% OTM put/call):")
    print(f"  Net Cost:        ${collar_cost:.2f}")
    print(f"  Put Strike:      ${put_s:.2f}")
    print(f"  Call Strike:     ${call_s:.2f}")
    print(f"  Protected Range: ${prot_range:.2f}")
    print(f"  Net Cost as %:   {collar_cost/portfolio_value*100:.3f}%")
