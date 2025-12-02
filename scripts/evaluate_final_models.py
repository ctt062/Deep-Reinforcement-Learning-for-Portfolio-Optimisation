#!/usr/bin/env python3
"""
Evaluate the trained DDPG and PPO models from the final benchmark.
This script only runs evaluation, not training.
"""

import yaml
import numpy as np
import json
import os
import sys
from pathlib import Path
from stable_baselines3 import DDPG, PPO

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.portfolio_env_with_options import PortfolioWithOptionsEnv

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_agent(agent, env, num_episodes=1):
    """Evaluate an agent on the environment"""
    results = {
        'episode_returns': [],
        'portfolio_values': [],
        'weights': [],
        'turnovers': [],
        'protective_puts': [],
        'covered_calls': [],
        'option_pnls': []
    }
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_returns = []
        episode_values = [env.portfolio_value]
        episode_weights = []
        episode_turnover = []
        episode_hedge = []
        episode_calls = []
        episode_option_pnl = []
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_returns.append(info.get('portfolio_return', 0))
            episode_values.append(env.portfolio_value)
            episode_weights.append(info.get('weights', np.zeros(env.n_assets)))
            episode_turnover.append(info.get('turnover', 0))
            episode_hedge.append(info.get('hedge_ratio', 0))
            episode_calls.append(info.get('call_ratio', 0))
            episode_option_pnl.append(info.get('option_pnl', 0))
        
        results['episode_returns'].append(episode_returns)
        results['portfolio_values'].append(episode_values)
        results['weights'].append(episode_weights)
        results['turnovers'].append(episode_turnover)
        results['protective_puts'].append(episode_hedge)
        results['covered_calls'].append(episode_calls)
        results['option_pnls'].append(episode_option_pnl)
    
    return results

def calculate_metrics(results):
    """Calculate performance metrics from results"""
    portfolio_values = results['portfolio_values'][0]
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Basic metrics
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    annualized_return = ((portfolio_values[-1] / portfolio_values[0]) ** (252 / len(portfolio_values)) - 1) * 100
    volatility = np.std(returns) * np.sqrt(252) * 100
    # FIXED: Sharpe ratio should subtract risk-free rate (2% = 2 in percentage terms)
    risk_free_rate = 2.0  # 2% annual
    sharpe_ratio = ((annualized_return - risk_free_rate) / volatility) if volatility > 0 else 0
    
    # Drawdown
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (cummax - portfolio_values) / cummax * 100
    max_drawdown = np.max(drawdown)
    
    # Turnover
    avg_turnover = np.mean(results['turnovers'][0])
    
    # Options
    avg_puts = np.mean(results['protective_puts'][0])
    avg_calls = np.mean(results['covered_calls'][0])
    total_option_pnl = np.sum(results['option_pnls'][0])
    
    metrics = {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'average_turnover': float(avg_turnover),
        'average_protective_puts': float(avg_puts),
        'average_covered_calls': float(avg_calls),
        'total_option_pnl': float(total_option_pnl),
        'final_portfolio_value': float(portfolio_values[-1]),
        'initial_portfolio_value': float(portfolio_values[0])
    }
    
    return metrics

def save_metrics_json(metrics, filename):
    """Save metrics to JSON file"""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)

def save_timeseries_json(results, prefix, dates=None, asset_names=None):
    """Save time series data to JSON files"""
    # Portfolio values
    data_to_save = {
        'values': [float(v) for v in results['portfolio_values'][0]]
    }
    if dates is not None:
        data_to_save['dates'] = [str(d) for d in dates]
    
    with open(f"{prefix}_portfolio_values.json", 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    # Drawdown
    portfolio_values = results['portfolio_values'][0]
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (cummax - portfolio_values) / cummax * 100
    
    data_to_save = {
        'drawdowns': [float(d) for d in drawdown]
    }
    if dates is not None:
        data_to_save['dates'] = [str(d) for d in dates]
    
    with open(f"{prefix}_drawdowns.json", 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    # Weights history
    weights_list = results['weights'][0]
    weights_data = {
        'weights': [[float(w) for w in weights] for weights in weights_list]
    }
    if dates is not None and len(dates) > 1:
        # Weights correspond to steps (one less than portfolio values)
        weights_data['dates'] = [str(d) for d in dates[1:len(weights_list)+1]]
    if asset_names is not None:
        weights_data['assets'] = asset_names
    
    with open(f"{prefix}_weights.json", 'w') as f:
        json.dump(weights_data, f, indent=2)

def main():
    print("=" * 70)
    print("  EVALUATING FINAL BENCHMARK MODELS")
    print("=" * 70)
    
    # Load configuration
    config_path = "configs/config_final_benchmark.yaml"
    config = load_config(config_path)
    
    # Create output directories
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    print("\n📊 Loading data...")
    data_config = config['data']
    
    loader = DataLoader(
        assets=data_config.get('assets', []),
        start_date=data_config.get('start_date'),
        end_date=data_config.get('end_date'),
        data_dir=data_config.get('data_dir', 'data'),
        frequency=data_config.get('frequency', '1d')
    )
    
    # Download/load data
    loader.download_data()
    
    # Get feature configuration
    feature_config = config.get('features', {})
    
    loader.build_features(
        sma_periods=feature_config.get('sma_periods', [5, 10, 20]),
        ema_periods=feature_config.get('ema_periods', [5, 10, 20]),
        momentum_periods=feature_config.get('momentum_periods', [5, 10, 20]),
        include_volatility=feature_config.get('include_volatility', True),
        volatility_window=feature_config.get('volatility_window', 20),
        normalize=feature_config.get('normalize', True),
        normalize_method=feature_config.get('normalize_method', 'zscore'),
        rolling_window=feature_config.get('rolling_window', 60)
    )
    
    # Split data using train_end
    train_end = data_config.get('train_end', None)
    if train_end:
        train_split, test_split = loader.train_test_split(train_end=train_end)
    else:
        train_split, test_split = loader.train_test_split(train_ratio=data_config.get('train_ratio', 0.7))
    
    print(f"✓ Test data: {len(test_split['features'])} samples")
    
    # Get environment configuration
    env_config = config.get('environment', {})
    options_config = config.get('options', {})
    
    # Create test environment
    test_env = PortfolioWithOptionsEnv(
        prices=test_split['prices'],
        returns=test_split['returns'],
        features=test_split['features'],
        initial_balance=env_config['initial_balance'],
        transaction_cost=env_config['transaction_cost'],
        lookback_window=config['features']['lookback_window'],
        reward_type=env_config['reward_type'],
        risk_penalty_lambda=env_config['risk_penalty_lambda'],
        volatility_window=env_config['volatility_window'],
        risk_free_rate=env_config['risk_free_rate'],
        allow_short=env_config['allow_short'],
        max_leverage=env_config['max_leverage'],
        turnover_penalty=env_config['turnover_penalty'],
        enable_options=options_config.get('enable_options', True),
        option_expiry_days=options_config.get('option_expiry_days', 30),
        option_transaction_cost=options_config.get('option_transaction_cost', 0.005),
        max_hedge_ratio=options_config.get('max_hedge_ratio', 1.0),
        max_call_ratio=options_config.get('max_call_ratio', 1.0),
        put_moneyness=options_config.get('put_moneyness', 0.95),
        call_moneyness=options_config.get('call_moneyness', 1.05),
        volatility_lookback=options_config.get('volatility_lookback', 20),
    )
    
    # Models to evaluate
    models = [
        ('ddpg', 'models/ddpg_options_final.zip', DDPG),
        ('ppo', 'models/ppo_options_final.zip', PPO)
    ]
    
    all_metrics = {}
    
    for agent_name, model_path, ModelClass in models:
        if not os.path.exists(model_path):
            print(f"\n⚠️  Model not found: {model_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"  Evaluating {agent_name.upper()}")
        print(f"{'='*70}")
        
        # Load model
        print(f"\n📦 Loading model from {model_path}...")
        agent = ModelClass.load(model_path)
        
        # Evaluate
        print(f"\n🧪 Evaluating on test set ({data_config.get('train_end', 'split')}-{data_config['end_date']})...")
        results = evaluate_agent(agent, test_env, num_episodes=1)
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        all_metrics[agent_name] = metrics
        
        # Save results
        print(f"\n💾 Saving results...")
        prefix = f"{results_dir}/{agent_name}_options_final"
        save_metrics_json(metrics, f"{prefix}_metrics.json")
        # Get dates from test environment - match the number of portfolio values
        # Portfolio values has one extra point (initial value before first step)
        num_values = len(results['portfolio_values'][0])
        dates = test_split['features'].index[:num_values].tolist()
        asset_names = data_config.get('assets', [])
        save_timeseries_json(results, prefix, dates=dates, asset_names=asset_names)
        
        print(f"\n✓ {agent_name.upper()} Results:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"  Total Return: {metrics['total_return']:.2f}%")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"  Volatility: {metrics['volatility']:.2f}%")
    
    # Print summary
    print(f"\n{'='*70}")
    print("  FINAL BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"\nModel Comparison ({data_config.get('train_end', 'split')}-{data_config['end_date']} Test Period):")
    print("-" * 70)
    print(f"{'Model':<15} {'Sharpe':>10} {'Return':>10} {'Max DD':>10} {'Volatility':>10}")
    print("-" * 70)
    
    for agent_name, metrics in all_metrics.items():
        print(f"{agent_name.upper():<15} "
              f"{metrics['sharpe_ratio']:>10.4f} "
              f"{metrics['total_return']:>9.2f}% "
              f"{metrics['max_drawdown']:>9.2f}% "
              f"{metrics['volatility']:>9.2f}%")
    
    print("-" * 70)
    print(f"\n{'='*70}")
    print("  ✅ Evaluation completed!")
    print(f"{'='*70}")
    print(f"\nNext step: Generate visualizations with:")
    print(f"  python scripts/visualize_benchmark_comparison.py")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
