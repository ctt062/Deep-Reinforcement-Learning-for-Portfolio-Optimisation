# Quick Start Guide

## 🚀 Running the Demo

### Option 1: Quick Demo (What We Just Did)
```bash
# 1. Install dependencies
pip install -q numpy pandas matplotlib seaborn yfinance gymnasium stable-baselines3 torch cvxpy scipy tqdm pyyaml

# 2. Run the demo in Python
python3 -c "
import sys; sys.path.insert(0, 'src')
from data_loader import DataLoader
from portfolio_env import PortfolioEnv
from agents import create_ppo_agent, train_agent
# ... (see demo steps below)
"
```

### Option 2: Using Scripts
```bash
# 1. Train an agent
python scripts/train.py --agent ppo --timesteps 50000 --save-path models/ppo_demo

# 2. Evaluate the agent
python scripts/evaluate.py --model models/ppo_demo.zip --n-episodes 5

# 3. Run full experiments
python scripts/run_experiments.py --config configs/config.yaml
```

### Option 3: Using Jupyter Notebook
```bash
# Open the demo notebook
jupyter notebook notebooks/demo.ipynb
```

## 📊 Demo Steps (What We Ran)

### Step 1: Data Loading & Feature Engineering
```python
from data_loader import DataLoader

assets = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'GLD', 'BTC-USD', 'ETH-USD']

loader = DataLoader(assets=assets, start_date='2015-01-01', end_date='2024-12-31')
loader.download_data()
loader.build_features(sma_periods=[5, 10, 20], ema_periods=[5, 10, 20], momentum_periods=[5, 10, 20])

train_data, test_data = loader.train_test_split(train_ratio=0.7)
```

### Step 2: Environment Setup
```python
from portfolio_env import PortfolioEnv

env = PortfolioEnv(
    prices=train_data['prices'],
    returns=train_data['returns'],
    features=train_data['features'],
    initial_balance=10000.0,
    transaction_cost=0.001,
    lookback_window=20,
    reward_type='risk_adjusted',
    risk_penalty_lambda=0.5
)
```

### Step 3: Train PPO Agent
```python
from agents import create_ppo_agent, train_agent

agent = create_ppo_agent(
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    net_arch=[256, 256]
)

agent = train_agent(agent=agent, total_timesteps=50000, save_path='models/ppo_demo')
```

### Step 4: Evaluate & Benchmark
```python
from agents import evaluate_agent
from benchmarks import run_all_benchmarks
from stable_baselines3 import PPO

# Load and evaluate PPO
agent = PPO.load('models/ppo_demo')
ppo_results = evaluate_agent(agent=agent, env=test_env, n_episodes=1)

# Run benchmarks
benchmarks = run_all_benchmarks(returns=test_data['returns'], transaction_cost=0.001, initial_value=10000.0)

# Compare performance
from metrics import PerformanceMetrics
for name, returns in all_strategies.items():
    metrics = PerformanceMetrics(returns=returns.values, risk_free_rate=0.02)
    print(f"{name}: Return={metrics.total_return():.2%}, Sharpe={metrics.sharpe_ratio():.3f}")
```

## 📁 Project Structure

```
Deep-Reinforcement-Learning-for-Portfolio-Optimisation/
├── src/                        # Core modules
│   ├── data_loader.py         # Data acquisition & feature engineering
│   ├── portfolio_env.py       # Custom Gym environment
│   ├── agents.py              # DRL agent creation (PPO, DDPG, DQN)
│   ├── benchmarks.py          # Baseline strategies
│   ├── metrics.py             # Performance evaluation
│   └── visualization.py       # Plotting utilities
├── scripts/                    # Command-line scripts
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── run_experiments.py     # Full experiment runner
├── notebooks/
│   └── demo.ipynb             # Interactive demo notebook
├── configs/
│   └── config.yaml            # Configuration file
├── data/                       # Data storage
│   ├── train_data.pkl         # Training data
│   └── test_data.pkl          # Testing data
├── models/                     # Saved models
│   └── ppo_demo.zip           # Trained PPO agent
├── results/                    # Results & visualizations
│   ├── evaluation_results.pkl # Evaluation metrics
│   ├── performance_analysis.png
│   └── ppo_weights.png
├── tests/                      # Unit tests
│   └── test_all.py
├── requirements.txt
├── setup.py
├── README.md
├── DEMO_RESULTS.md            # Demo results summary
└── PROJECT_SUMMARY.md         # Project documentation
```

## 🔧 Key Components

### Data Loader
- Downloads historical data from Yahoo Finance
- Computes technical indicators (SMA, EMA, Momentum, Volatility)
- Normalizes features using z-score
- Handles train/test splitting

### Portfolio Environment
- Custom Gym environment for RL training
- Observation: 20-day lookback window + portfolio weights
- Action: 10 continuous values → softmax → portfolio weights
- Reward: Risk-adjusted returns with transaction costs
- Constraints: Long-only, leverage=1.0

### Agents
- **PPO**: Best for portfolio optimization (continuous actions)
- **DDPG**: Alternative continuous control algorithm
- **DQN**: For discrete action spaces (not recommended here)

### Benchmarks
- **Equal-Weight**: Simple 1/N allocation
- **Mean-Variance**: Markowitz optimization
- **Momentum**: Top-K momentum-based allocation
- **Buy-and-Hold**: Static allocation

### Metrics
- Total Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Annualized Volatility
- Calmar Ratio
- Value at Risk (VaR)

## 🎯 Performance Targets

### Training
- **Quick Demo**: 50K timesteps (~3 minutes)
- **Standard**: 100K timesteps (~6 minutes)
- **Production**: 500K timesteps (~30 minutes)

### Expected Results (Test Set 2022-2024)
- **Total Return**: 60-80%
- **Sharpe Ratio**: 0.4-0.6
- **Max Drawdown**: 40-50%

### Baselines to Beat
- Equal-Weight: 83% return, 0.56 Sharpe
- Momentum: 42% return, 0.22 Sharpe

## 🐛 Troubleshooting

### ImportError: No module named 'xxx'
```bash
pip install -r requirements.txt
```

### CVXPY not available
```bash
pip install cvxpy  # For Mean-Variance optimization
```

### GPU not detected
```bash
# PyTorch with CUDA (for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Yahoo Finance download fails
- Check internet connection
- Verify asset tickers are valid
- Try reducing date range
- Delete cached files in `data/` and retry

## 📚 Additional Resources

### Documentation
- `README.md` - Project overview
- `DEMO_RESULTS.md` - Demo results summary
- `PROJECT_SUMMARY.md` - Technical documentation
- `report.md` - Academic report

### Code Examples
- `notebooks/demo.ipynb` - Interactive walkthrough
- `scripts/train.py` - Training example
- `tests/test_all.py` - Unit tests

### References
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Yahoo Finance API](https://github.com/ranaroussi/yfinance)

## 🎓 Learning Path

1. ✅ **Run the demo** (what we just did)
2. **Understand the code**: Read through src/ modules
3. **Modify parameters**: Try different hyperparameters
4. **Add features**: Experiment with new technical indicators
5. **Try other algorithms**: Test DDPG, A2C, SAC
6. **Extend the project**: Add more assets, different reward functions, risk constraints

## 💡 Experiment Ideas

1. **More Training**: Train for 500K timesteps
2. **Different Assets**: Try sector ETFs or international stocks
3. **Reward Functions**: Implement Sharpe-based or Sortino-based rewards
4. **Risk Constraints**: Add VaR limits or leverage constraints
5. **Feature Engineering**: Add RSI, MACD, Bollinger Bands
6. **Ensemble Methods**: Combine multiple agents

---

**Happy Learning! 🚀📈**
