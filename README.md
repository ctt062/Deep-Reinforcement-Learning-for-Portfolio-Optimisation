# Deep Reinforcement Learning for Portfolio Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**IEDA4000F - Deep Learning for Decision Analytics**  
**The Hong Kong University of Science and Technology (HKUST)**

## Overview

This project implements a deep reinforcement learning (DRL) framework for portfolio optimization with options overlay and advanced risk management. We formulate portfolio management as a Markov Decision Process (MDP) and train intelligent agents to maximize risk-adjusted returns while managing transaction costs, portfolio turnover, and downside risk.

**Final Result**: DDPG agent achieves **5.52 Sharpe ratio** with only **8.31% maximum drawdown** during the 2019-2020 test period (including COVID-19 crash), significantly outperforming PPO (1.85 Sharpe, 17.06% DD) and traditional benchmarks.

### Key Features

- 🤖 **Multiple DRL Algorithms**: DDPG and PPO implementations with continuous action spaces
- 📊 **Custom Trading Environment**: Gymnasium-compatible environment with realistic constraints
- 💰 **Transaction Cost Modeling**: Explicit turnover and slippage modeling (0.1% per trade)
- 📈 **Rich Feature Engineering**: 60+ features including SMA, EMA, RSI, Momentum, Volatility
- 🛡️ **Advanced Risk Management**: Options overlay (protective puts + covered calls) and tiered stop-loss system
- 🎯 **Comprehensive Benchmarks**: Equal-weight, Mean-Variance, Momentum strategies
- 📉 **Financial Metrics**: Sharpe ratio, Maximum Drawdown, Volatility, Turnover, Options P&L
- 🏆 **Outstanding Performance**: DDPG achieves 5.52 Sharpe ratio with 8.31% max drawdown
- 🔬 **Academic Quality**: Clean, modular code with detailed docstrings and PEP 8 compliance

## Mathematical Formulation

### Markov Decision Process (MDP)

**State Space**: $s_t = [p_{t-K:t}, x_t, w_{t-1}]$
- Price history window: $p_{t-K:t} \in \mathbb{R}^{N \times K}$
- Technical features: $x_t$ (SMA, EMA, Momentum)
- Previous weights: $w_{t-1} \in \mathbb{R}^N$

**Action Space**: Portfolio weights $w_t \in \mathbb{R}^N$
- Constraints: $\sum_{i=1}^N w_{t,i} = 1$, $w_{t,i} \geq 0$ (long-only)
- Softmax parameterization: $w_t = \frac{\exp(z_t)}{\mathbf{1}^T \exp(z_t)}$

**Reward Function**: Risk-adjusted return
$$r_t = R_t - \lambda \hat{\sigma}^2_t$$

Where:
- Gross return: $R_t^{\text{gross}} = w_{t-1}^T r_t$
- Turnover: $\text{Turn}_t = \sum_{i=1}^N |w_{t,i} - w_{t-1,i}|$
- Transaction cost: $\text{cost}_t = c \cdot \text{Turn}_t$
- Net return: $R_t = R_t^{\text{gross}} - \text{cost}_t$

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
│   ├── portfolio_env.py         # Custom Gym trading environment
│   ├── agents.py                # DRL agent implementations
│   ├── benchmarks.py            # Benchmark strategies
│   ├── metrics.py               # Performance evaluation metrics
│   └── visualization.py         # Plotting utilities
│
├── configs/                      # Configuration files
│   └── config.yaml              # Hyperparameters and settings
│
├── notebooks/                    # Jupyter notebooks
│   └── demo.ipynb               # Interactive demonstration
│
├── scripts/                      # Executable scripts
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation/backtesting script
│
├── data/                         # Data storage (gitignored)
├── models/                       # Saved model checkpoints
├── results/                      # Plots and tables
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── README.md                     # This file
├── report.md                     # Results and analysis
└── LICENSE                       # MIT License
```

## Usage

### 1. Quick Start with Jupyter Notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

The demo notebook provides an interactive walkthrough of the entire pipeline.

### 2. Training Final Benchmark Models

```bash
# Train both DDPG and PPO on 2010-2018, test on 2019-2020
python scripts/train_and_evaluate_final.py

# Or use the shell script
bash scripts/train_final_benchmark.sh

# Monitor training progress
bash scripts/watch_training.sh
```

### 3. Evaluation and Visualization

```bash
# Evaluate the final trained models
python scripts/evaluate_final_models.py

# Generate comparison visualizations
python scripts/visualize_benchmark_comparison.py
```

The results will be saved to:
- Models: `models/` (ddpg_options_final.zip, ppo_options_final.zip)
- Results: `results/` (JSON files with metrics, portfolio values, drawdowns)
- Visualizations: `visualizations/` (PNG charts)

### 4. Custom Configuration

Edit `configs/config.yaml` to customize:
- Asset universe and data period
- Feature engineering parameters
- Network architecture
- Training hyperparameters
- Transaction costs and constraints

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

### 1. Deep Deterministic Policy Gradient (DDPG) - **Winner** 🏆
- **Continuous action space** for precise portfolio weights
- **Actor-critic architecture** with separate policy and value networks
- **Off-policy learning** with experience replay (500K buffer)
- **Deterministic policy** for consistent decision-making
- **Network**: Actor [512, 512, 256, 128], Critic [512, 512, 256, 128]
- **Hyperparameters**: LR=1e-4, Batch=256, Gamma=0.99, Tau=0.01
- **Options strategy**: Aggressive use of protective puts (44.87%) and covered calls (75.62%)
- **Final Performance**: 5.52 Sharpe, 219% return, 8.31% max DD

### 2. Proximal Policy Optimization (PPO)
- **Continuous action space** with stochastic policy
- **Clipped surrogate objective** for stable training
- **On-policy learning** with Generalized Advantage Estimation (GAE)
- **Network**: [512, 512, 256, 128]
- **Hyperparameters**: LR=5e-5, Batch=128, Epochs=10, Gamma=0.99
- **Options strategy**: Minimal use (0.08% puts, 2.98% calls)
- **Final Performance**: 1.85 Sharpe, 61% return, 17.06% max DD

### Key Difference
DDPG's **off-policy learning** and **deterministic policy** allow it to effectively learn and execute the options overlay strategy, while PPO's on-policy approach failed to discover this profitable hedging behavior.

## Benchmarks

1. **Equal-Weight**: $w_i = 1/N$ for all assets
2. **Mean-Variance Optimization**: Markowitz quadratic programming
3. **Momentum**: Allocate based on trailing returns

## Performance Metrics

- **Annualized Return (AR)**: $\left[\prod_{t=1}^T (1 + R_t)\right]^{252/T} - 1$
- **Sharpe Ratio**: $\frac{AR - r_f}{\sigma_{\text{ann}}}$
- **Maximum Drawdown**: $\max_{t'<t} \frac{V_{t'} - V_t}{V_{t'}}$
- **Annualized Volatility**: $\text{std}(R_t) \sqrt{252}$
- **Average Turnover**: $\frac{1}{T}\sum_{t=1}^T \text{Turn}_t$

## Results

See `FINAL_BENCHMARK.md` for detailed analysis. Key visualizations available in `visualizations/`:
- Sharpe ratio comparison
- Cumulative portfolio values (2019-2020)
- Drawdown analysis over time
- Comprehensive metrics comparison
- Performance summary table

### Final Benchmark Results (2019-2020 Test Period)

| Metric | DDPG | PPO | Target | Winner |
|--------|------|-----|--------|--------|
| **Sharpe Ratio** | **5.52** | 1.85 | > 1.0 | ✅ DDPG |
| **Total Return** | **219.40%** | 61.12% | > 15% | ✅ DDPG |
| **Annualized Return** | **93.31%** | 31.09% | > 15% | ✅ DDPG |
| **Max Drawdown** | **8.31%** | 17.06% | < 10% | ✅ DDPG |
| **Volatility** | 16.89% | 16.78% | - | Similar |
| **Avg Turnover** | 1.83% | 1.53% | - | Both Low |
| **Final Portfolio** | **$319,400** | $161,120 | - | ✅ DDPG |
| **Options P&L** | **+$126,568** | +$5,758 | - | ✅ DDPG |

**DDPG Winner**: Achieves 3x higher Sharpe ratio, 3.6x higher returns, and 2x lower drawdown than PPO.

## Key Findings

1. **DDPG significantly outperforms PPO** (5.52 vs 1.85 Sharpe) on all metrics
2. **Options overlay is highly effective** - DDPG generated $126K profit through protective puts (44.87%) and covered calls (75.62%)
3. **Risk management works** - DDPG max drawdown only 8.31% despite COVID-19 crash (met <10% target)
4. **PPO failed to utilize options** - Only 0.08% protective puts, 2.98% covered calls
5. **RL successfully generalizes** - Trained on 2010-2018, successfully handled unprecedented 2019-2020 volatility
6. **Traditional methods fail** - Equal-weight (0.56 Sharpe, 43% DD), Mean-Variance (-0.40 Sharpe, 76% DD)
7. **Off-policy learning advantage** - DDPG's deterministic policy and sample efficiency superior for portfolio optimization

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
