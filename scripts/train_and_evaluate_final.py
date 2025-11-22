#!/usr/bin/env python3
"""
Train and evaluate all three models (DDPG, PPO, DQN) for final benchmark.
Train on 2010-2018, test on 2018-2020.
Save results in JSON format for visualization comparison.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.portfolio_env_with_options import PortfolioWithOptionsEnv
from src.agents import create_agent, train_agent, TrainingCallback
from src.metrics import PerformanceMetrics
import yaml
import numpy as np
import pandas as pd
import torch


def set_random_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_environment(config: dict, data: dict):
    """Create portfolio environment with options from config and data."""
    env_config = config['environment']
    options_config = config.get('options', {})
    
    env = PortfolioWithOptionsEnv(
        prices=data['prices'],
        returns=data['returns'],
        features=data['features'],
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
    
    return env


def evaluate_agent(agent, env, num_episodes: int = 1):
    """Evaluate trained agent and return results."""
    results = {
        'returns': [],
        'values': [],
        'weights': [],
        'turnover': [],
        'hedge_ratios': [],
        'call_ratios': [],
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
            
            done = done or truncated
        
        results['returns'].append(np.array(episode_returns))
        results['values'].append(np.array(episode_values))
        results['weights'].append(np.array(episode_weights))
        results['turnover'].append(np.array(episode_turnover))
        results['hedge_ratios'].append(np.array(episode_hedge))
        results['call_ratios'].append(np.array(episode_calls))
        results['option_pnls'].append(np.array(episode_option_pnl))
    
    # Average over episodes
    for key in results:
        results[key] = np.mean(np.array(results[key]), axis=0)
    
    return results


def save_metrics_json(metrics_dict, filepath):
    """Save metrics to JSON file."""
    # Convert numpy types to Python types
    json_metrics = {}
    for key, value in metrics_dict.items():
        if isinstance(value, np.ndarray):
            json_metrics[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            json_metrics[key] = float(value)
        else:
            json_metrics[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(json_metrics, f, indent=2)


def save_timeseries_json(dates, values, filepath):
    """Save time series data to JSON file."""
    data = {
        'dates': [str(d) for d in dates],
        'values': values.tolist() if isinstance(values, np.ndarray) else values
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def train_and_evaluate_agent(agent_name, config, train_data, test_data, output_dir):
    """Train and evaluate a single agent."""
    print("\n" + "="*70)
    print(f"  Training and Evaluating {agent_name.upper()}")
    print("="*70)
    
    # Set random seed
    seed = config.get('seed', 42)
    set_random_seed(seed)
    
    # Get agent-specific config
    agent_config = config['agents'][agent_name]
    
    # Create agent-specific config with environment overrides
    agent_env_config = config.copy()
    if 'risk_penalty_lambda' in agent_config:
        agent_env_config['environment']['risk_penalty_lambda'] = agent_config['risk_penalty_lambda']
    if 'turnover_penalty' in agent_config:
        agent_env_config['environment']['turnover_penalty'] = agent_config['turnover_penalty']
    
    # Create training environment
    print("\nCreating training environment...")
    train_env = create_environment(agent_env_config, train_data)
    print(f"Training environment: {train_env.n_assets} assets, {train_env.max_steps} steps")
    print(f"Risk penalty lambda: {train_env.risk_penalty_lambda}")
    print(f"Turnover penalty: {train_env.turnover_penalty}")
    
    # Create agent
    print(f"\nCreating {agent_name.upper()} agent...")
    
    # Filter out environment parameters from agent config
    agent_params = {k: v for k, v in agent_config.items() 
                   if k not in ['risk_penalty_lambda', 'max_position_size', 'turnover_penalty']}
    
    agent = create_agent(
        agent_type=agent_name,
        env=train_env,
        config=agent_params,
    )
    
    # Train agent
    print(f"\nTraining {agent_name.upper()}...")
    print(f"Total timesteps: {config['training']['total_timesteps']:,}")
    
    callback = TrainingCallback(verbose=1)
    agent = train_agent(
        agent=agent,
        total_timesteps=config['training']['total_timesteps'],
        callback=callback
    )
    
    # Save model
    model_path = os.path.join(output_dir, f"{agent_name}_options_final.zip")
    agent.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Evaluate on test set
    print(f"\nEvaluating on test set (2018-2020)...")
    test_env = create_environment(agent_env_config, test_data)
    
    results = evaluate_agent(agent, test_env, num_episodes=1)
    
    # Calculate metrics
    print("\nCalculating performance metrics...")
    metrics = PerformanceMetrics(
        returns=results['returns'],
        values=results['values'],
        weights=results['weights'],
        turnover=results['turnover'],
        risk_free_rate=config['metrics']['risk_free_rate'],
        periods_per_year=config['metrics']['annualized_factor']
    )
    
    metrics_dict = metrics.get_all_metrics()
    
    # Convert metric names to lowercase with underscores
    clean_metrics = {
        'sharpe_ratio': metrics_dict['Sharpe Ratio'],
        'total_return': metrics_dict['Total Return'],
        'annualized_return': metrics_dict['Annualized Return'],
        'max_drawdown': metrics_dict['Max Drawdown'],
        'volatility': metrics_dict['Annualized Volatility'],
        'sortino_ratio': metrics_dict['Sortino Ratio'],
        'calmar_ratio': metrics_dict['Calmar Ratio'],
        'turnover': metrics_dict['Average Turnover'] / 100.0,
        'win_rate': metrics_dict['Hit Ratio'],
        'profit_factor': metrics_dict.get('Profit Factor', 0),
        'var_95': metrics_dict['VaR (95%)'],
        'cvar_95': metrics_dict['CVaR (95%)']
    }
    
    # Print results
    print("\n" + "="*70)
    print(f"  {agent_name.upper()} Results (2018-2020 Test Period)")
    print("="*70)
    print(f"Sharpe Ratio:        {clean_metrics['sharpe_ratio']:8.4f}")
    print(f"Total Return:        {clean_metrics['total_return']*100:8.2f}%")
    print(f"Annualized Return:   {clean_metrics['annualized_return']*100:8.2f}%")
    print(f"Max Drawdown:        {clean_metrics['max_drawdown']*100:8.2f}%")
    print(f"Volatility:          {clean_metrics['volatility']*100:8.2f}%")
    print(f"Sortino Ratio:       {clean_metrics['sortino_ratio']:8.4f}")
    print("="*70)
    
    # Check targets
    print("\nTarget Achievement:")
    print(f"  Sharpe > 1.0:    {'✓' if clean_metrics['sharpe_ratio'] > 1.0 else '✗'}")
    print(f"  DD < 10%:        {'✓' if clean_metrics['max_drawdown'] < 0.10 else '✗'}")
    print(f"  Return > 15%:    {'✓' if clean_metrics['annualized_return'] > 0.15 else '✗'}")
    
    # Save results to JSON
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Save metrics
    metrics_file = results_dir / f"{agent_name}_options_final_metrics.json"
    save_metrics_json(clean_metrics, metrics_file)
    print(f"\n✓ Metrics saved to: {metrics_file}")
    
    # Save portfolio values with dates
    dates = test_data['prices'].index
    values_file = results_dir / f"{agent_name}_options_final_portfolio_values.json"
    save_timeseries_json(dates, results['values'], values_file)
    print(f"✓ Portfolio values saved to: {values_file}")
    
    # Calculate and save drawdowns
    cummax = np.maximum.accumulate(results['values'])
    drawdowns = (results['values'] - cummax) / cummax
    dd_file = results_dir / f"{agent_name}_options_final_drawdowns.json"
    save_timeseries_json(dates, drawdowns, dd_file)
    print(f"✓ Drawdowns saved to: {dd_file}")
    
    return clean_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate all models for final benchmark"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_final_benchmark.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--agents',
        type=str,
        nargs='+',
        default=['ddpg', 'ppo'],
        help='Agents to train (DQN not supported for continuous actions)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  Final Benchmark Training & Evaluation")
    print("="*70)
    print("\nTraining Period: 2010-2018 (8 years)")
    print("Test Period: 2018-2020 (2 years)")
    print(f"Models: {', '.join([a.upper() for a in args.agents])}")
    print("="*70)
    
    # Load config
    config = load_config(args.config)
    
    # Load and prepare data
    print("\nLoading data...")
    data_config = config['data']
    loader = DataLoader(
        assets=data_config['assets'],
        start_date=data_config['start_date'],
        end_date=data_config['end_date'],
        data_dir=config['paths']['data_dir'],
        frequency=data_config['frequency']
    )
    
    prices = loader.download_data()
    
    print("\nBuilding features...")
    feature_config = config['features']
    features = loader.build_features(
        sma_periods=feature_config['sma_periods'],
        ema_periods=feature_config['ema_periods'],
        momentum_periods=feature_config['momentum_periods'],
        normalize=feature_config['normalize_prices'],
        normalize_method=feature_config['normalize_method'],
        rolling_window=feature_config.get('rolling_window', 60)
    )
    
    print("\nSplitting data (train: 2010-2018, test: 2018-2020)...")
    train_end = data_config.get('train_end', None)
    if train_end:
        train_data, test_data = loader.train_test_split(train_end=train_end)
    else:
        train_data, test_data = loader.train_test_split(
            train_ratio=data_config['train_ratio']
        )
    
    print(f"Train samples: {len(train_data['features'])}")
    print(f"Test samples: {len(test_data['features'])}")
    
    # Create output directory
    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)
    
    # Train and evaluate each agent
    all_results = {}
    for agent_name in args.agents:
        try:
            results = train_and_evaluate_agent(
                agent_name, config, train_data, test_data, output_dir
            )
            all_results[agent_name] = results
        except Exception as e:
            print(f"\n⚠️  Error training {agent_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("  FINAL BENCHMARK SUMMARY")
    print("="*70)
    print("\nModel Comparison (2018-2020 Test Period):")
    print("-" * 70)
    print(f"{'Model':<10} {'Sharpe':>10} {'Return':>10} {'Max DD':>10} {'Volatility':>12}")
    print("-" * 70)
    
    for agent_name, metrics in all_results.items():
        print(f"{agent_name.upper():<10} "
              f"{metrics['sharpe_ratio']:>10.4f} "
              f"{metrics['annualized_return']*100:>9.2f}% "
              f"{metrics['max_drawdown']*100:>9.2f}% "
              f"{metrics['volatility']*100:>11.2f}%")
    
    print("-" * 70)
    
    # Find best model
    if all_results:
        best_agent = max(all_results.items(), key=lambda x: x[1]['sharpe_ratio'])
        print(f"\n🏆 Best Model: {best_agent[0].upper()} "
              f"(Sharpe: {best_agent[1]['sharpe_ratio']:.4f})")
    
    print("\n" + "="*70)
    print("  ✅ All training and evaluation completed!")
    print("="*70)
    print("\nNext step: Generate visualizations with:")
    print("  python scripts/visualize_benchmark_comparison.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
