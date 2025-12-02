# Final Benchmark: Training & Results Documentation

## Overview

This document describes the final benchmark training with unified hyperparameters to ensure fair comparison between DDPG and PPO algorithms. Both agents are trained with identical learning rates, batch sizes, and risk penalty coefficients to isolate algorithmic differences.

## Training Configuration

### Time Period
- **Training**: 2010-01-01 to 2018-12-31 (8 years, 2,064 samples)
- **Testing**: 2019-01-02 to 2020-12-30 (2 years, 504 samples)
  - Includes December 2018 market volatility
  - Includes February-March 2020 COVID-19 crash

### Portfolio Composition
- **18 Assets** across multiple sectors:
  - **Technology** (5): AAPL, MSFT, GOOGL, NVDA, AMZN
  - **Healthcare** (3): JNJ, UNH, PFE
  - **Financials** (2): JPM, V
  - **Consumer** (2): WMT, COST
  - **Indices** (3): SPY, QQQ, IWM
  - **Bonds** (2): TLT, AGG
  - **Commodities** (1): GLD
- **Frequency**: Daily (1d)
- **Initial Balance**: $100,000

### Unified Hyperparameters

To ensure fair comparison, both agents use identical hyperparameters:

| Parameter | Value (Both Agents) |
|-----------|---------------------|
| Learning Rate | 5×10⁻⁵ |
| Batch Size | 128 |
| Discount Factor (γ) | 0.99 |
| Network Architecture | [512, 512, 256, 128] |
| Risk Penalty (λ) | 5.0 |
| Training Timesteps | 100,000 |

### DDPG-Specific Parameters
- Replay Buffer Size: 500,000
- Learning Starts: 10,000
- Soft Update Coefficient (τ): 0.01
- Action Noise: 0.15

### PPO-Specific Parameters
- Steps per Update: 2,048
- Number of Epochs: 10
- GAE Parameter (λ): 0.95
- Clip Range (ε): 0.2
- Entropy Coefficient: 0.01

### Risk Management Mechanisms

#### 1. Volatility Targeting
- Target: 10% annualized volatility
- Scales exposure inversely to realized volatility
- Formula: $\text{Exposure} = \min(1.0, \frac{\sigma_{\text{target}}}{\sigma_{\text{realized}}})$

#### 2. Progressive Position Reduction
- Starts reducing exposure at 3% drawdown
- Gradual deleveraging from 3% to 9% drawdown
- Minimum exposure: 10% at 9% drawdown

#### 3. Aggressive Drawdown Penalties
- Risk penalty coefficient: λ = 5.0
- Penalty starts at 2% drawdown
- Exponential scaling as drawdown increases
- Massive penalty above 8% drawdown

## Training Process

### Scripts

1. **`scripts/train_and_evaluate_final.py`**
   - Trains both DDPG and PPO sequentially
   - Uses explicit `train_end='2018-12-31'` for train/test split
   - Evaluates on 2019-2020 test period
   - Saves metrics, portfolio values, drawdowns, and weights in JSON format

2. **`scripts/evaluate_final_models.py`**
   - Evaluates pre-trained models
   - Generates comprehensive metrics
   - Saves weight history for analysis

3. **`scripts/visualize_benchmark_comparison.py`**
   - Generates 5 comparison charts:
     1. Sharpe Ratio Comparison
     2. All Metrics Comparison (6 subplots)
     3. Cumulative Portfolio Values
     4. Drawdown Over Time
     5. Metrics Summary Table

4. **`scripts/generate_additional_plots.py`**
   - Generates additional visualizations:
     - Asset correlation matrix
     - Training curves
     - Portfolio weight allocation analysis
     - Risk management analysis

### Configuration Files
- **`configs/config_final_benchmark.yaml`**
  - Complete configuration for benchmark
  - Includes `train_end: '2018-12-31'` parameter
  - Unified hyperparameters for both agents

### Output Locations
- **Models**: `models/`
  - `ddpg_options_final.zip`
  - `ppo_options_final.zip`

- **Results**: `results/`
  - `ddpg_options_final_metrics.json`
  - `ppo_options_final_metrics.json`
  - `ddpg_options_final_portfolio_values.json`
  - `ppo_options_final_portfolio_values.json`
  - `ddpg_options_final_drawdowns.json`
  - `ppo_options_final_drawdowns.json`
  - `ddpg_options_final_weights.json`
  - `ppo_options_final_weights.json`

- **Visualizations**: `visualizations/`
  - `sharpe_comparison.png`
  - `all_metrics_comparison.png`
  - `cumulative_portfolio_values.png`
  - `drawdown_over_time.png`
  - `metrics_summary_table.png`
  - `correlation_matrix.png`
  - `training_curves.png`
  - `weight_allocation.png`
  - `risk_analysis.png`

## Training Timeline

### Execution
```bash
# Start training (runs in background)
python scripts/train_and_evaluate_final.py

# Monitor progress
bash scripts/watch_training.sh

# Evaluate models
python scripts/evaluate_final_models.py

# Generate visualizations
python scripts/visualize_benchmark_comparison.py
python scripts/generate_additional_plots.py
```

### Estimated Duration
- **DDPG**: ~10-15 minutes (100K timesteps)
- **PPO**: ~10-15 minutes (100K timesteps)
- **Total**: ~20-30 minutes

## Final Results (2019-2020 Test Period)

### DDPG Results
- **Sharpe Ratio**: 1.78 ✅
- **Total Return**: 40.82% ✅
- **Annualized Return**: 21.50% ✅
- **Max Drawdown**: 9.02% ✅ (Met <10% target)
- **Volatility**: 10.96% ✅
- **Turnover**: 0.04% ✅
- **Final Portfolio Value**: $132,194
- **Total P&L**: $32,194

### PPO Results
- **Sharpe Ratio**: 1.84 ✅
- **Total Return**: 42.73% ✅
- **Annualized Return**: 22.43% ✅
- **Max Drawdown**: 9.05% ✅ (Met <10% target)
- **Volatility**: 11.09% ✅
- **Turnover**: 0.04% ✅
- **Final Portfolio Value**: $142,729
- **Total P&L**: $42,729

### Key Findings

1. **Comparable Performance**: Both agents achieve similar risk-adjusted returns under unified configuration
   - Sharpe: 1.78 (DDPG) vs 1.84 (PPO)
   - Max Drawdown: 9.02% (DDPG) vs 9.05% (PPO)

2. **Risk Management Success**: Both agents achieved the <10% maximum drawdown target
   - Volatility targeting maintained consistent ~11% annualized volatility
   - Progressive position reduction provided effective downside protection

3. **COVID-19 Resilience**: Both agents limited losses to ~8% while the market declined 33.9%
   - DDPG: -7.8% during COVID crash period
   - PPO: -8.1% during COVID crash period

4. **Algorithm Comparison**: 
   - PPO slightly outperforms in risk-adjusted returns (Sharpe: 1.84 vs 1.78)
   - Both algorithms demonstrate effective risk management
   - Unified hyperparameters ensure fair comparison

5. **Risk Management > Algorithm**: The key insight is that proper risk management design (volatility targeting, progressive position reduction, aggressive penalties) is more important than algorithm selection.

## Portfolio Allocation Analysis

### DDPG Allocation
- **Top 5 Holdings**: MSFT (7.4%), AAPL (7.4%), PFE (7.3%), AGG (7.3%), JNJ (7.3%)
- **Sector Allocation**: Technology (28.0%), Healthcare (17.6%), Index (13.2%), Bonds (10.2%)
- **Style**: More concentrated with higher healthcare allocation

### PPO Allocation
- **Top 5 Holdings**: V (5.8%), AAPL (5.2%), NVDA (5.2%), UNH (5.1%), SPY (5.1%)
- **Sector Allocation**: Technology (25.1%), Index (14.9%), Healthcare (14.9%), Financials (10.7%)
- **Style**: More diversified, closer to equal-weight

## Success Criteria

### Minimum Acceptable Performance
- ✅ Sharpe Ratio > 1.0 (Both achieved: 1.78, 1.84)
- ✅ Max Drawdown < 10% (Both achieved: 9.02%, 9.05%)
- ✅ Positive returns over 2-year period (Both achieved: 40.82%, 42.73%)

### Target Performance
- ✅ Sharpe Ratio > 1.0
- ✅ Max Drawdown < 10%
- ✅ Annualized Return > 15% (Both achieved: 21.50%, 22.43%)

### Stretch Goals
- ✅ Sharpe Ratio > 1.5 (Both achieved)
- ✅ Max Drawdown < 10% (Both achieved)
- ✅ Annualized Return > 20% (Both achieved)

## Notes

- **Why Unified Hyperparameters?** Ensures fair comparison by isolating algorithmic differences from hyperparameter effects
- **Why 2019-2020?** This period includes extreme market volatility (COVID-19) that tests model robustness
- **Risk Management Focus**: The framework emphasizes explicit risk management mechanisms over algorithm selection
- **Reproducibility**: All hyperparameters documented in `configs/config_final_benchmark.yaml`
- **Academic Report**: Full analysis available in `zz report/main.pdf`

## Next Steps

1. **Review Results**: Examine visualizations in `visualizations/` directory
2. **Read Report**: See `zz report/main.pdf` for comprehensive analysis
3. **Experiment**: Modify `configs/config_final_benchmark.yaml` to test different configurations
4. **Extend**: Add more assets, different risk management strategies, or other algorithms

---

**Repository**: [https://github.com/ctt062/Deep-Reinforcement-Learning-for-Portfolio-Optimisation](https://github.com/ctt062/Deep-Reinforcement-Learning-for-Portfolio-Optimisation)
