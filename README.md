# Deep Reinforcement Learning for Portfolio Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**IEDA4000F - Deep Learning for Decision Analytics**  
**The Hong Kong University of Science and Technology (HKUST)**

## Overview

This project implements a deep reinforcement learning (DRL) framework for portfolio optimization. We formulate portfolio management as a Markov Decision Process (MDP) and train intelligent agents to maximize risk-adjusted returns while managing transaction costs and portfolio turnover.

### Key Features

- 🤖 **Multiple DRL Algorithms**: DQN, PPO, and DDPG implementations
- 📊 **Custom Trading Environment**: OpenAI Gym-compatible environment with realistic constraints
- 💰 **Transaction Cost Modeling**: Explicit turnover and slippage modeling
- 📈 **Rich Feature Engineering**: Technical indicators (SMA, EMA, Momentum, etc.)
- 🎯 **Comprehensive Benchmarks**: Equal-weight, Mean-Variance, Momentum strategies
- 📉 **Financial Metrics**: Sharpe ratio, Maximum Drawdown, Volatility, Turnover analysis
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

### 2. Training DRL Agents

```bash
# Train PPO agent (default)
python scripts/train.py --agent ppo --timesteps 100000

# Train DDPG agent with custom parameters
python scripts/train.py --agent ddpg --timesteps 200000 --transaction-cost 0.001

# Train DQN agent
python scripts/train.py --agent dqn --timesteps 50000
```

### 3. Evaluation and Backtesting

```bash
# Evaluate trained model
python scripts/evaluate.py --agent ppo --model-path models/ppo_portfolio.zip

# Compare all agents and benchmarks
python scripts/evaluate.py --compare-all --transaction-cost 0.001

# Generate comprehensive report
python scripts/evaluate.py --compare-all --save-results
```

### 4. Custom Configuration

Edit `configs/config.yaml` to customize:
- Asset universe and data period
- Feature engineering parameters
- Network architecture
- Training hyperparameters
- Transaction costs and constraints

## Datasets

The project uses **Yahoo Finance** data via the `yfinance` library:

- **Assets**: AAPL, NVDA, TSLA, MSFT, GOOGL, AMZN, SPY, GLD, BTC-USD, ETH-USD
- **Period**: 2015-01-01 to 2024-12-31
- **Frequency**: Daily closing prices
- **Splits**: 70% train (2015-2020), 30% test (2021-2024)

Data is automatically downloaded and cached on first run.

## Algorithms Implemented

### 1. Deep Q-Network (DQN)
- Discrete action space (predefined weight combinations)
- Experience replay buffer
- Target network with periodic updates

### 2. Proximal Policy Optimization (PPO)
- Continuous action space
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)

### 3. Deep Deterministic Policy Gradient (DDPG)
- Continuous action space
- Actor-critic architecture
- Ornstein-Uhlenbeck exploration noise

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

See `report.md` for detailed analysis including:
- Performance comparison tables
- Cumulative return plots
- Portfolio allocation visualizations
- Sensitivity analysis (transaction costs, market regimes)

## Key Findings (Expected)

1. **DRL agents outperform traditional benchmarks** in Sharpe ratio and drawdown control
2. **PPO/DDPG handle continuous allocation** more efficiently than DQN
3. **Transaction costs significantly impact** high-turnover strategies
4. **Agents adapt to market regimes** through learned policies
5. **Feature engineering crucial** for performance

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
