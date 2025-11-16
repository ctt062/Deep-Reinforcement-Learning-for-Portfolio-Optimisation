"""
Evaluation script for DRL portfolio optimization agents.

Usage:
    python scripts/evaluate.py --agent ppo --model-path models/ppo_final.zip
    python scripts/evaluate.py --compare-all --transaction-cost 0.001
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.portfolio_env import PortfolioEnv
from src.agents import create_agent
from src.benchmarks import run_all_benchmarks
from src.metrics import PerformanceMetrics, compare_strategies
from src.visualization import (
    plot_cumulative_returns,
    plot_portfolio_values,
    plot_weights_stacked,
    plot_drawdown,
    plot_return_distribution,
    plot_metrics_comparison,
    plot_turnover_analysis
)
from src.discrete_wrapper import DiscretePortfolioWrapper

from stable_baselines3 import PPO, DDPG, DQN


def set_random_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_environment(config: dict, data: dict):
    """Create portfolio environment from config and data."""
    env_config = config['environment']
    
    env = PortfolioEnv(
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
    )
    
    return env


def evaluate_drl_agent(agent, env):
    """Evaluate a DRL agent and return results."""
    obs, info = env.reset()
    done = False
    
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    # Get episode history
    history = env.get_portfolio_history()
    
    return {
        'returns': history['returns'],
        'values': history['values'],
        'weights': history['weights'],
        'turnover': history['turnover'],
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate DRL agent for portfolio optimization")
    parser.add_argument(
        '--agent',
        type=str,
        default=None,
        choices=['dqn', 'ppo', 'ddpg'],
        help='Type of DRL agent to evaluate'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model file (.zip)'
    )
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Compare all available agents and benchmarks'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--transaction-cost',
        type=float,
        default=None,
        help='Transaction cost (overrides config)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save results and plots'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.compare_all and (args.agent is None or args.model_path is None):
        parser.error("Either --compare-all or both --agent and --model-path must be specified")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.transaction_cost is not None:
        config['environment']['transaction_cost'] = args.transaction_cost
    
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Set random seed
    seed = config.get('seed', 42)
    set_random_seed(seed)
    print(f"Random seed set to: {seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*60)
    print("Portfolio Optimization Evaluation")
    print("="*60)
    
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
    
    # Download data
    prices = loader.download_data()
    
    # Build features
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
    
    # Train-test split
    print("\nSplitting data...")
    train_data, test_data = loader.train_test_split(
        train_ratio=data_config['train_ratio']
    )
    
    print(f"Test period: {test_data['prices'].index[0]} to {test_data['prices'].index[-1]}")
    print(f"Test samples: {len(test_data['prices'])}")
    
    # Dictionary to store all results
    all_results = {}
    
    # Evaluate DRL agent if specified
    if args.model_path is not None and args.agent is not None:
        print(f"\n{'='*60}")
        print(f"Evaluating {args.agent.upper()} Agent")
        print("="*60)
        
        # Create test environment
        env = create_environment(config, test_data)
        
        # Wrap with discrete action space for DQN
        if args.agent == 'dqn':
            print("Wrapping environment with discrete action space for DQN...")
            n_discrete_actions = config['agents'].get('dqn', {}).get('n_discrete_actions', 100)
            env = DiscretePortfolioWrapper(env, n_discrete_actions=n_discrete_actions, strategy="mixed")
        
        # Load trained agent
        print(f"Loading model from {args.model_path}...")
        agent_map = {'ppo': PPO, 'ddpg': DDPG, 'dqn': DQN}
        AgentClass = agent_map[args.agent]
        agent = AgentClass.load(args.model_path, env=env)
        
        print(f"Model loaded successfully")
        
        # Evaluate
        print("Running evaluation...")
        drl_results = evaluate_drl_agent(agent, env)
        all_results[f'{args.agent.upper()}'] = drl_results
        
        print(f"Evaluation completed")
        print(f"Final portfolio value: ${drl_results['values'][-1]:,.2f}")
    
    # Run benchmarks
    print(f"\n{'='*60}")
    print("Running Benchmark Strategies")
    print("="*60)
    
    benchmark_results = run_all_benchmarks(
        returns=test_data['returns'],
        transaction_cost=config['environment']['transaction_cost'],
        initial_value=config['environment']['initial_balance'],
        mv_lookback=config['benchmarks'].get('mv_lookback', 60),
        momentum_lookback=config['benchmarks'].get('momentum_lookback', 20),
        momentum_top_k=config['benchmarks'].get('momentum_top_k', None),
    )
    
    # Add benchmarks to results
    all_results.update(benchmark_results)
    
    # Calculate and display metrics
    print(f"\n{'='*60}")
    print("Performance Metrics")
    print("="*60)
    
    metrics_df = compare_strategies(
        all_results,
        risk_free_rate=config['environment']['risk_free_rate'],
        periods_per_year=config['metrics']['annualized_factor']
    )
    
    print("\n" + metrics_df.to_string())
    
    # Detailed metrics for each strategy
    print(f"\n{'='*60}")
    print("Detailed Metrics by Strategy")
    print("="*60)
    
    for name, data in all_results.items():
        print(f"\n{name}:")
        metrics = PerformanceMetrics(
            returns=data['returns'],
            values=data['values'],
            weights=data.get('weights'),
            turnover=data.get('turnover'),
            risk_free_rate=config['environment']['risk_free_rate'],
            periods_per_year=config['metrics']['annualized_factor']
        )
        metrics.print_metrics(name)
    
    # Save results if requested
    if args.save_results:
        print(f"\n{'='*60}")
        print("Saving Results")
        print("="*60)
        
        # Save metrics to CSV
        metrics_path = os.path.join(args.output_dir, f'metrics_{timestamp}.csv')
        metrics_df.to_csv(metrics_path)
        print(f"Metrics saved to: {metrics_path}")
        
        # Create visualizations
        print("\nGenerating visualizations...")
        
        # Extract data for plotting
        returns_dict = {name: data['returns'] for name, data in all_results.items()}
        values_dict = {name: data['values'] for name, data in all_results.items()}
        turnover_dict = {name: data['turnover'] for name, data in all_results.items() 
                        if 'turnover' in data}
        
        # 1. Cumulative returns
        fig_path = os.path.join(args.output_dir, f'cumulative_returns_{timestamp}.png')
        plot_cumulative_returns(returns_dict, save_path=fig_path)
        
        # 2. Portfolio values
        fig_path = os.path.join(args.output_dir, f'portfolio_values_{timestamp}.png')
        plot_portfolio_values(values_dict, save_path=fig_path)
        
        # 3. Return distribution
        fig_path = os.path.join(args.output_dir, f'return_distribution_{timestamp}.png')
        plot_return_distribution(returns_dict, save_path=fig_path)
        
        # 4. Metrics comparison
        fig_path = os.path.join(args.output_dir, f'metrics_comparison_{timestamp}.png')
        plot_metrics_comparison(metrics_df, save_path=fig_path)
        
        # 5. Turnover analysis
        if turnover_dict:
            fig_path = os.path.join(args.output_dir, f'turnover_analysis_{timestamp}.png')
            plot_turnover_analysis(turnover_dict, save_path=fig_path)
        
        # 6. Drawdown for best strategy
        best_strategy = metrics_df['Sharpe Ratio'].idxmax()
        best_values = all_results[best_strategy]['values']
        fig_path = os.path.join(args.output_dir, f'drawdown_{best_strategy}_{timestamp}.png')
        plot_drawdown(best_values, title=f"Drawdown Analysis - {best_strategy}", save_path=fig_path)
        
        # 7. Weights visualization for DRL agent
        if args.model_path is not None and 'weights' in drl_results:
            fig_path = os.path.join(args.output_dir, f'weights_{args.agent}_{timestamp}.png')
            plot_weights_stacked(
                drl_results['weights'],
                test_data['prices'].columns.tolist(),
                title=f"Portfolio Allocation - {args.agent.upper()}",
                save_path=fig_path
            )
        
        print(f"\nAll visualizations saved to: {args.output_dir}")
        
        # Save raw results
        results_path = os.path.join(args.output_dir, f'results_{timestamp}.npz')
        np.savez(
            results_path,
            **{f"{name}_returns": data['returns'] for name, data in all_results.items()},
            **{f"{name}_values": data['values'] for name, data in all_results.items()},
        )
        print(f"Raw results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("Evaluation completed successfully!")
    print("="*60)
    
    # Summary
    print("\nSummary:")
    print(f"Best Strategy (Sharpe): {metrics_df['Sharpe Ratio'].idxmax()}")
    print(f"Best Sharpe Ratio: {metrics_df['Sharpe Ratio'].max():.4f}")
    print(f"Best Strategy (Return): {metrics_df['Annualized Return'].idxmax()}")
    print(f"Best Annualized Return: {metrics_df['Annualized Return'].max():.2%}")
    print(f"Lowest Drawdown: {metrics_df['Max Drawdown'].min():.2%} "
          f"({metrics_df['Max Drawdown'].idxmin()})")


if __name__ == "__main__":
    main()
