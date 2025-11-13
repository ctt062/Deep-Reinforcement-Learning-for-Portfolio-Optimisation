# Deep Reinforcement Learning for Portfolio Optimization - Results Report

**Course**: IEDA4000F - Deep Learning for Decision Analytics  
**Institution**: The Hong Kong University of Science and Technology (HKUST)  
**Date**: November 2025

---

## Executive Summary

This report presents the results of applying Deep Reinforcement Learning (DRL) algorithms to portfolio optimization. We compare DRL agents (PPO, DDPG) against traditional benchmarks (Equal-Weight, Mean-Variance, Momentum) across multiple financial metrics.

### Key Findings

- **Best Strategy**: [To be filled after running experiments]
- **Best Sharpe Ratio**: [Value]
- **Best Annualized Return**: [Value]
- **DRL vs Benchmarks**: [Summary comparison]

---

## 1. Methodology

### 1.1 Problem Formulation

Portfolio optimization as Markov Decision Process (MDP):

**State Space**: $s_t = [p_{t-K:t}, x_t, w_{t-1}]$
- Price history window: $p_{t-K:t} \in \mathbb{R}^{N \times K}$
- Technical features: $x_t$ (SMA, EMA, Momentum, Volatility)
- Previous weights: $w_{t-1} \in \mathbb{R}^N$

**Action Space**: Portfolio weights $w_t \in \mathbb{R}^N$
- Constraints: $\sum_{i=1}^N w_{t,i} = 1$, $w_{t,i} \geq 0$
- Softmax: $w_t = \frac{\exp(z_t)}{\mathbf{1}^T \exp(z_t)}$

**Reward Function**: 
$$r_t = R_t - \lambda \hat{\sigma}^2_t$$

Where:
- Net return: $R_t = w_{t-1}^T r_t - c \cdot \text{Turn}_t$
- Turnover: $\text{Turn}_t = \sum_{i=1}^N |w_{t,i} - w_{t-1,i}|$
- Transaction cost: $c = 0.001$ (0.1%)

**Objective**: 
$$\max_{\pi_\theta} J(\pi_\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T \gamma^t r_t\right]$$

### 1.2 Dataset

- **Assets**: AAPL, NVDA, TSLA, MSFT, GOOGL, AMZN, SPY, GLD, BTC-USD, ETH-USD
- **Period**: 2015-01-01 to 2024-12-31
- **Frequency**: Daily
- **Train Split**: 70% (2015-2020)
- **Test Split**: 30% (2021-2024)
- **Features**: 
  - Price history (20-day lookback)
  - SMA (5, 10, 20 days)
  - EMA (5, 10, 20 days)
  - Momentum (5, 10, 20 days)
  - Volatility (20-day window)

### 1.3 DRL Algorithms

#### PPO (Proximal Policy Optimization)
- **Policy Network**: 2 layers [128, 128] with ReLU
- **Value Network**: 2 layers [128, 128] with ReLU
- **Learning Rate**: 0.0003
- **Batch Size**: 64
- **Clip Range**: 0.2
- **Training Steps**: 100,000

#### DDPG (Deep Deterministic Policy Gradient)
- **Actor Network**: 2 layers [128, 128] with ReLU
- **Critic Network**: 2 layers [128, 128] with ReLU
- **Learning Rate**: 0.001
- **Batch Size**: 100
- **Soft Update (τ)**: 0.005
- **Training Steps**: 100,000

### 1.4 Benchmark Strategies

1. **Equal-Weight**: $w_i = 1/N$ for all assets
2. **Mean-Variance**: Markowitz optimization with 60-day lookback
3. **Momentum**: Top-3 assets by 20-day momentum
4. **Buy-and-Hold**: Initial equal allocation, no rebalancing

---

## 2. Results

### 2.1 Performance Metrics Summary

| Strategy | Ann. Return | Ann. Vol. | Sharpe | Max DD | Avg Turnover | Final Value |
|----------|------------|-----------|--------|--------|--------------|-------------|
| PPO | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |
| DDPG | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |
| Equal-Weight | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |
| Mean-Variance | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |
| Momentum | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |
| Buy-and-Hold | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |

**Metrics Definitions**:
- **Ann. Return**: Annualized return = $\left[\prod_{t=1}^T (1 + R_t)\right]^{252/T} - 1$
- **Ann. Vol.**: Annualized volatility = $\text{std}(R_t) \sqrt{252}$
- **Sharpe**: Sharpe ratio = $(AR - r_f) / \sigma_{\text{ann}}$
- **Max DD**: Maximum drawdown = $\max_{t'<t} \frac{V_{t'} - V_t}{V_{t'}}$
- **Avg Turnover**: Average portfolio turnover

### 2.2 Cumulative Returns

[Insert cumulative returns plot]

**Observations**:
- [Analysis of return trajectories]
- [Comparison of strategies]
- [Market regime performance]

### 2.3 Risk-Return Profile

[Insert risk-return scatter plot]

**Observations**:
- [Efficient frontier analysis]
- [Risk-adjusted performance]

### 2.4 Drawdown Analysis

[Insert drawdown plots for top strategies]

**Observations**:
- [Maximum drawdown comparison]
- [Recovery periods]
- [Tail risk analysis]

### 2.5 Portfolio Allocation

[Insert allocation heatmaps/stacked area charts]

**Observations**:
- [Asset allocation patterns]
- [Diversification analysis]
- [Adaptive behavior]

### 2.6 Turnover Analysis

[Insert turnover comparison]

**Observations**:
- [Trading frequency]
- [Cost impact]
- [Stability analysis]

---

## 3. Analysis and Discussion

### 3.1 DRL Agent Performance

#### PPO Agent
- **Strengths**: [List strengths]
- **Weaknesses**: [List weaknesses]
- **Key Behaviors**: [Observed patterns]

#### DDPG Agent
- **Strengths**: [List strengths]
- **Weaknesses**: [List weaknesses]
- **Key Behaviors**: [Observed patterns]

### 3.2 Benchmark Comparison

- **vs Equal-Weight**: [Comparison]
- **vs Mean-Variance**: [Comparison]
- **vs Momentum**: [Comparison]

### 3.3 Transaction Cost Impact

Analysis of performance under different transaction cost scenarios:
- **c = 0.000** (no costs): [Results]
- **c = 0.001** (0.1%): [Results]
- **c = 0.010** (1.0%): [Results]

**Key Finding**: [Impact summary]

### 3.4 Market Regime Analysis

Performance breakdown by market conditions:

| Strategy | Bull Market | Bear Market | Volatile | Stable |
|----------|------------|-------------|----------|--------|
| PPO | [Value] | [Value] | [Value] | [Value] |
| DDPG | [Value] | [Value] | [Value] | [Value] |
| Equal-Weight | [Value] | [Value] | [Value] | [Value] |

**Observations**: [Analysis of regime-specific performance]

### 3.5 Statistical Significance

[Insert statistical tests if applicable]
- **t-tests**: [Results]
- **Confidence intervals**: [Results]

---

## 4. Sensitivity Analysis

### 4.1 Hyperparameter Sensitivity

Impact of key hyperparameters on PPO performance:
- **Learning Rate**: [Analysis]
- **Network Architecture**: [Analysis]
- **Risk Penalty (λ)**: [Analysis]

### 4.2 Lookback Window

Performance with different lookback windows:
- **K = 10**: [Results]
- **K = 20**: [Results]
- **K = 40**: [Results]

---

## 5. Limitations and Future Work

### 5.1 Limitations

1. **Data Limitations**:
   - Historical data only (survivorship bias)
   - Limited to liquid assets
   - Daily frequency (no intraday)

2. **Model Limitations**:
   - Long-only constraint (no shorting)
   - Simplified transaction cost model
   - No market impact modeling
   - No liquidity constraints

3. **Evaluation Limitations**:
   - Single test period
   - Limited market regimes
   - No out-of-distribution testing

### 5.2 Future Research Directions

1. **Model Enhancements**:
   - Ensemble methods (multiple agents)
   - Attention mechanisms for feature processing
   - Hierarchical RL for multi-timeframe decisions
   - Transfer learning across asset classes

2. **Risk Management**:
   - Incorporate CVaR constraints
   - Dynamic risk budgeting
   - Stress testing framework

3. **Practical Extensions**:
   - Real-time execution simulation
   - Multi-currency portfolios
   - Alternative data sources (sentiment, etc.)
   - Regulatory constraint handling

4. **Robustness**:
   - Adversarial testing
   - Non-stationary environment adaptation
   - Online learning and retraining strategies

---

## 6. Conclusions

### Key Takeaways

1. **DRL Viability**: [Summary of whether DRL is effective for portfolio optimization]

2. **Best Practices**:
   - [Practice 1]
   - [Practice 2]
   - [Practice 3]

3. **Practical Implications**:
   - [Implication 1]
   - [Implication 2]
   - [Implication 3]

### Final Remarks

[Concluding thoughts on the project and its contributions]

---

## 7. References

1. Jiang, Z., Xu, D., & Liang, J. (2017). A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem. *arXiv preprint arXiv:1706.10059*.

2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

3. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.

4. Lillicrap, T. P., et al. (2015). Continuous control with deep reinforcement learning. *arXiv preprint arXiv:1509.02971*.

5. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.

6. Moody, J., & Saffell, M. (2001). Learning to trade via direct reinforcement. *IEEE Transactions on Neural Networks*, 12(4), 875-889.

7. Deng, Y., et al. (2016). Deep Direct Reinforcement Learning for Financial Signal Representation and Trading. *IEEE Transactions on Neural Networks and Learning Systems*, 28(3), 653-664.

---

## Appendix

### A. Code Repository

Complete code available at: https://github.com/ctt062/Deep-Reinforcement-Learning-for-Portfolio-Optimisation

### B. Hyperparameter Settings

[Complete table of all hyperparameters used]

### C. Additional Plots

[Any supplementary visualizations]

### D. Asset Statistics

[Detailed statistics for individual assets]

---

**Disclaimer**: This is an academic research project for educational purposes only. The results presented should not be interpreted as financial advice. Real-world trading involves additional complexities, risks, and regulatory requirements not captured in this simulation.
