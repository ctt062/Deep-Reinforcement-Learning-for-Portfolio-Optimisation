# Deep Reinforcement Learning for Portfolio Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**IEDA4000F - Deep Learning for Decision Analytics**  
**The Hong Kong University of Science and Technology (HKUST)**

## Overview

This project implements a deep reinforcement learning (DRL) framework for portfolio optimization with advanced risk management mechanisms. We formulate portfolio management as a Markov Decision Process (MDP) and train intelligent agents to maximize risk-adjusted returns while maintaining strict drawdown control through volatility targeting and progressive position reduction.

**Final Result**: Both DDPG and PPO agents achieve comparable performance under unified hyperparameters, with **<10% maximum drawdown** during the 2019-2020 test period (including COVID-19 crash), demonstrating that proper risk management design is more important than algorithm selection.

### Key Features

- 🤖 **Multiple DRL Algorithms**: DDPG and PPO implementations with continuous action spaces
- 📊 **Custom Trading Environment**: Gymnasium-compatible environment with realistic constraints
- 💰 **Transaction Cost Modeling**: Explicit turnover and slippage modeling (0.1% per trade)
- 📈 **Rich Feature Engineering**: 252 features including SMA, EMA, Momentum, Volatility
- 🛡️ **Advanced Risk Management**: Volatility targeting, progressive position reduction, aggressive drawdown penalties
- 🎯 **Comprehensive Benchmarks**: Equal-weight, Mean-Variance, Momentum strategies
- 📉 **Financial Metrics**: Sharpe ratio, Maximum Drawdown, Volatility, Turnover, VaR, CVaR
- 🏆 **Outstanding Performance**: Both agents achieve <10% max drawdown target with Sharpe >1.7
- 🔬 **Academic Quality**: Clean, modular code with detailed docstrings and PEP 8 compliance

## Mathematical Formulation

### Markov Decision Process (MDP)

**State Space**: $s_t = [p_{t-K:t}, x_t, w_{t-1}]$
- Price history window: $p_{t-K:t} \in \mathbb{R}^{N \times K}$ (K=60 days)
- Technical features: $x_t$ (252 features: SMA, EMA, Momentum, Volatility)
- Previous weights: $w_{t-1} \in \mathbb{R}^N$

**Action Space**: Portfolio weights $w_t \in \mathbb{R}^N$
- Constraints: $\sum_{i=1}^N w_{t,i} \leq 1$, $w_{t,i} \geq 0$ (long-only)
- Softmax parameterization: $w_t = \frac{\exp(z_t)}{\mathbf{1}^T \exp(z_t)}$

**Reward Function**: Risk-adjusted return with aggressive drawdown penalties
$$r_t = R_t - \lambda \cdot \text{DD}_t^2 \cdot \mathbb{1}(\text{DD}_t > 0.02)$$

Where:
- Gross return: $R_t^{\text{gross}} = w_{t-1}^T r_t$
- Turnover: $\text{Turn}_t = \sum_{i=1}^N |w_{t,i} - w_{t-1,i}|$
- Transaction cost: $\text{cost}_t = c \cdot \text{Turn}_t$
- Net return: $R_t = R_t^{\text{gross}} - \text{cost}_t$
- Drawdown penalty: $\lambda = 5.0$ (aggressive risk control)

**Risk Management**:
- Volatility targeting: $\text{Exposure} = \min(1.0, \frac{\sigma_{\text{target}}}{\sigma_{\text{realized}}})$
- Progressive position reduction: Starts at 3% DD, reaches 10% exposure at 9% DD

**Objective**: Maximize cumulative discounted reward
$$J(\pi_\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T \gamma^t r_t\right]$$

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/ctt062/Deep-Reinforcement-Learning-for-Portfolio-Optimisation.git
cd Deep-Reinforcement-Learning-for-Portfolio-Optimisation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Project Structure

```
Deep-Reinforcement-Learning-for-Portfolio-Optimisation/
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_loader.py           # Data fetching and preprocessing
│   ├── portfolio_env.py         # Base portfolio environment
│   ├── portfolio_env_with_options.py  # Environment with options overlay
│   ├── agents.py                # DRL agent implementations
│   ├── benchmarks.py            # Benchmark strategies
│   ├── metrics.py               # Performance evaluation metrics
│   ├── options_pricing.py       # Black-Scholes options pricing
│   └── visualization.py         # Plotting utilities
│
├── configs/                      # Configuration files
│   ├── config.yaml              # Default configuration
│   └── config_final_benchmark.yaml  # Final benchmark configuration
│
├── scripts/                      # Executable scripts
│   ├── train_and_evaluate_final.py  # Training and evaluation pipeline
│   ├── evaluate_final_models.py     # Model evaluation
│   ├── visualize_benchmark_comparison.py  # Visualization generation
│   └── generate_additional_plots.py  # Additional visualizations
│
├── tests/                        # Unit tests
│   └── test_all.py              # Test suite
│
├── data/                         # Data storage (gitignored)
├── models/                       # Saved model checkpoints
├── results/                      # Evaluation results (JSON)
├── visualizations/              # Generated plots
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── README.md                     # This file
└── LICENSE                       # MIT License
```

## Usage

### 1. Training Final Benchmark Models

```bash
# Train both DDPG and PPO on 2010-2018, test on 2019-2020
python scripts/train_and_evaluate_final.py

# Or use the shell script
bash scripts/train_final_benchmark.sh

# Monitor training progress
bash scripts/watch_training.sh
```

### 2. Evaluation and Visualization

```bash
# Evaluate the final trained models
python scripts/evaluate_final_models.py

# Generate comparison visualizations
python scripts/visualize_benchmark_comparison.py

# Generate additional plots (correlation matrix, training curves, weight allocation)
python scripts/generate_additional_plots.py
```

The results will be saved to:
- Models: `models/` (ddpg_options_final.zip, ppo_options_final.zip)
- Results: `results/` (JSON files with metrics, portfolio values, drawdowns, weights)
- Visualizations: `visualizations/` (PNG charts)

### 3. Custom Configuration

Edit `configs/config_final_benchmark.yaml` to customize:
- Asset universe and data period
- Feature engineering parameters
- Network architecture
- Training hyperparameters (unified for fair comparison)
- Risk management parameters

## Datasets

The project uses **Yahoo Finance** data via the `yfinance` library:

- **Assets**: 18 diversified assets across multiple sectors  
  - **Technology** (5): AAPL, MSFT, GOOGL, NVDA, AMZN  
  - **Healthcare** (3): JNJ, UNH, PFE  
  - **Financials** (2): JPM, V  
  - **Consumer** (2): WMT, COST  
  - **Equity Indices** (3): SPY, QQQ, IWM  
  - **Bonds** (2): TLT, AGG  
  - **Commodities** (1): GLD
- **Period**: 2010-01-01 to 2020-12-31 (11 years)
- **Frequency**: Daily closing prices
- **Splits**:  
  - Training: 2010-01-01 to 2018-12-31 (8 years, 2,064 samples)  
  - Testing: 2019-01-02 to 2020-12-30 (2 years, 504 samples)  
  - **Test period includes COVID-19 crash** for robustness validation

Data is automatically downloaded and cached on first run. The dataset file is located at:
`data/prices_AAPL_MSFT_GOOGL_NVDA_AMZN_JNJ_UNH_PFE_JPM_V_WMT_COST_SPY_QQQ_IWM_TLT_AGG_GLD_2010-01-01_2020-12-31.csv`

## Algorithms Implemented

### 1. Deep Deterministic Policy Gradient (DDPG)
- **Continuous action space** for precise portfolio weights
- **Actor-critic architecture** with separate policy and value networks
- **Off-policy learning** with experience replay (500K buffer)
- **Deterministic policy** for consistent decision-making
- **Network**: Actor [512, 512, 256, 128], Critic [512, 512, 256, 128]
- **Unified Hyperparameters**: LR=5e-5, Batch=128, Gamma=0.99, Risk Penalty λ=5.0
- **Final Performance**: Sharpe 1.78, Return 40.82%, Max DD 9.02%

### 2. Proximal Policy Optimization (PPO)
- **Continuous action space** with stochastic policy
- **Clipped surrogate objective** for stable training
- **On-policy learning** with Generalized Advantage Estimation (GAE)
- **Network**: [512, 512, 256, 128]
- **Unified Hyperparameters**: LR=5e-5, Batch=128, Epochs=10, Gamma=0.99, Risk Penalty λ=5.0
- **Final Performance**: Sharpe 1.84, Return 42.73%, Max DD 9.05%

### Key Finding
With unified hyperparameters and proper risk management, both algorithms achieve comparable performance, demonstrating that **risk management design is more important than algorithm selection**.

## Benchmarks

1. **Equal-Weight**: $w_i = 1/N$ for all assets
2. **Mean-Variance Optimization**: Markowitz quadratic programming
3. **SPY Buy-and-Hold**: 100% allocation to S&P 500 ETF

## Performance Metrics

- **Annualized Return (AR)**: $\left[\prod_{t=1}^T (1 + R_t)\right]^{252/T} - 1$
- **Sharpe Ratio**: $\frac{AR - r_f}{\sigma_{\text{ann}}}$
- **Maximum Drawdown**: $\max_{t'<t} \frac{V_{t'} - V_t}{V_{t'}}$
- **Annualized Volatility**: $\text{std}(R_t) \sqrt{252}$
- **Average Turnover**: $\frac{1}{T}\sum_{t=1}^T \text{Turn}_t$
- **Value at Risk (VaR)**: 95th percentile of daily returns
- **Conditional VaR (CVaR)**: Expected loss beyond VaR threshold

## Results

See the full academic report in `zz report/main.pdf` for detailed analysis. Key visualizations available in `visualizations/`:
- Sharpe ratio comparison
- Cumulative portfolio values (2019-2020)
- Drawdown analysis over time
- Comprehensive metrics comparison
- Training curves
- Portfolio weight allocation analysis
- Asset correlation matrix
- Risk management analysis

### Final Benchmark Results (2019-2020 Test Period)

| Metric | DDPG | PPO | Target | Status |
|--------|------|-----|--------|--------|
| **Sharpe Ratio** | 1.78 | **1.84** | > 1.0 | ✅ Both |
| **Total Return** | 40.82% | **42.73%** | > 15% | ✅ Both |
| **Annualized Return** | 21.50% | **22.43%** | > 15% | ✅ Both |
| **Max Drawdown** | **9.02%** | 9.05% | < 10% | ✅ Both |
| **Volatility** | **10.96%** | 11.09% | - | Both Low |
| **Turnover** | 0.04% | **0.04%** | - | Both Low |
| **Final Portfolio** | $132,194 | **$142,729** | - | - |

**Key Finding**: Both agents achieve the <10% maximum drawdown target with comparable risk-adjusted returns, demonstrating the effectiveness of unified risk management mechanisms.

## Key Findings

1. **Comparable Algorithm Performance**: With unified hyperparameters, both DDPG and PPO achieve similar performance (Sharpe: 1.78 vs 1.84)
2. **Effective Risk Management**: Volatility targeting and progressive position reduction achieve <10% max drawdown target
3. **COVID-19 Resilience**: Both agents limited losses to ~8% while the market declined 33.9%
4. **Risk Management > Algorithm**: Proper risk management design is more important than algorithm selection
5. **RL Successfully Generalizes**: Trained on 2010-2018, successfully handled unprecedented 2019-2020 volatility
6. **Unified Configuration**: Fair comparison demonstrates that both off-policy (DDPG) and on-policy (PPO) approaches work well when properly configured

## Reproducibility

Set random seeds for reproducibility:

```python
import numpy as np
import torch
import random

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
```

Configuration files are provided in `configs/config_final_benchmark.yaml` with all hyperparameters documented.

## Ethical Considerations

⚠️ **Important Disclaimers:**
- This is an **academic research project** for educational purposes only
- **NOT financial advice** - do not use for real trading without proper validation
- Historical performance does not guarantee future results
- Real-world trading involves additional complexities not modeled here
- Always consult qualified financial advisors before making investment decisions

## References

1. Jiang, Z., Xu, D., & Liang, J. (2017). A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem. *arXiv preprint arXiv:1706.10059*.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
3. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.
4. Lillicrap, T. P., et al. (2015). Continuous control with deep reinforcement learning. *arXiv preprint arXiv:1509.02971*.
5. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.

## Contributing

This is an academic project. For improvements or bug fixes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Course**: IEDA4000F - Deep Learning for Decision Analytics
- **Institution**: The Hong Kong University of Science and Technology (HKUST)
- **Libraries**: stable-baselines3, OpenAI Gym, PyTorch, yfinance

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Disclaimer**: This project is for educational and research purposes only. The authors are not responsible for any financial losses incurred from using this code for real trading.
