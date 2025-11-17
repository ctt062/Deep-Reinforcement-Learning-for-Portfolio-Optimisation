"""
Evaluation script for portfolio optimization with option overlays.

Evaluates DRL agents trained on portfolio environment with options
and compares against baseline models without options.
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
from src.portfolio_env_with_options import PortfolioWithOptionsEnv
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

from stable_baselines3 import PPO, DDPG


def set_random_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_config(config_path: str = "configs/config_with_options.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        # Fall back to default config
        config_path = "configs/config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure options config exists
    if 'options' not in config:
        config['options'] = {
            'enable_options': True,
            'option_expiry_days': 30,
            'option_transaction_cost': 0.005,
            'max_hedge_ratio': 1.0,
            'max_call_ratio': 1.0,
            'put_moneyness': 0.95,
            'call_moneyness': 1.05,
            'volatility_lookback': 20,
        }
    
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
        # Options-specific parameters
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


def evaluate_drl_agent(agent, env):
    """Evaluate a DRL agent and return results."""
    obs, info = env.reset()
    done = False
    
    # Track option usage
    hedge_ratios = []
    call_ratios = []
    option_pnls = []
    
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Track option metrics
        if 'hedge_ratio' in info:
            hedge_ratios.append(info['hedge_ratio'])
        if 'call_ratio' in info:
            call_ratios.append(info['call_ratio'])
        if 'option_pnl' in info:
            option_pnls.append(info['option_pnl'])
    
    # Get episode history
    history = env.get_portfolio_history()
    
    # Add option metrics
    history['hedge_ratios'] = np.array(hedge_ratios) if hedge_ratios else np.array([])
    history['call_ratios'] = np.array(call_ratios) if call_ratios else np.array([])
    history['option_pnls'] = np.array(option_pnls) if option_pnls else np.array([])
    
    return history


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate DRL agent with options overlay"
    )
    parser.add_argument(
        '--agent',
        type=str,
        required=True,
        choices=['ppo', 'ddpg'],
        help='Type of DRL agent'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_with_options.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save evaluation results and visualizations'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results_with_options',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Set random seed
    seed = config.get('seed', 42)
    set_random_seed(seed)
    print(f"Random seed set to: {seed}")
    
    print("\n" + "="*60)
    print("Portfolio Optimization with Options Overlay - Evaluation")
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
    
    # Evaluate DRL agent with options
    print(f"\n{'='*60}")
    print(f"Evaluating {args.agent.upper()} Agent with Options")
    print("="*60)
    
    # Create test environment
    env = create_environment(config, test_data)
    
    # Load trained agent
    print(f"Loading model from {args.model_path}...")
    agent_map = {'ppo': PPO, 'ddpg': DDPG}
    AgentClass = agent_map[args.agent]
    agent = AgentClass.load(args.model_path, env=env)
    print("Model loaded successfully")
    
    # Evaluate
    print("Running evaluation...")
    results = evaluate_drl_agent(agent, env)
    
    print(f"Evaluation completed")
    print(f"Final portfolio value: ${results['values'][-1]:,.2f}")
    
    if 'option_premium_paid' in results:
        print(f"Total option premium paid: ${results['option_premium_paid']:,.2f}")
        print(f"Total option premium received: ${results['option_premium_received']:,.2f}")
        print(f"Net option cost: ${results['net_option_cost']:,.2f}")
    
    # Calculate performance metrics
    print("\n" + "="*60)
    print("Performance Metrics with Options")
    print("="*60)
    
    metrics = PerformanceMetrics(
        returns=results['returns'],
        values=results['values'],
        weights=results['weights'],
        turnover=results['turnover'],
        risk_free_rate=config['metrics']['risk_free_rate'],
        periods_per_year=config['metrics']['annualized_factor']
    )
    
    metrics_dict = metrics.get_all_metrics()
    
    # Print formatted metrics
    print(f"\n{'='*60}")
    print(f" Performance Metrics: {args.agent.upper()} with Options")
    print(f"{'='*60}")
    print(f"Total Return............................{metrics_dict['Total Return']*100:10.2f}%")
    print(f"Annualized Return.......................{metrics_dict['Annualized Return']*100:10.2f}%")
    print(f"Annualized Volatility...................{metrics_dict['Annualized Volatility']*100:10.2f}%")
    print(f"Sharpe Ratio............................{metrics_dict['Sharpe Ratio']:10.4f}")
    print(f"Sortino Ratio...........................{metrics_dict['Sortino Ratio']:10.4f}")
    print(f"Max Drawdown............................{metrics_dict['Max Drawdown']*100:10.2f}%")
    print(f"Calmar Ratio............................{metrics_dict['Calmar Ratio']:10.4f}")
    print(f"VaR (95%)...............................{metrics_dict['VaR (95%)']*100:10.2f}%")
    print(f"CVaR (95%)..............................{metrics_dict['CVaR (95%)']*100:10.2f}%")
    print(f"Hit Ratio...............................{metrics_dict['Hit Ratio']:10.4f}")
    print(f"Average Turnover........................{metrics_dict['Average Turnover']:10.2f}%")
    print(f"{'='*60}\n")
    
    # Check if targets achieved
    print("Target Achievement:")
    print(f"  Sharpe Ratio > 1.0:        {'✓ YES' if metrics_dict['Sharpe Ratio'] > 1.0 else '✗ NO'} ({metrics_dict['Sharpe Ratio']:.4f})")
    print(f"  Max Drawdown < 10%:        {'✓ YES' if metrics_dict['Max Drawdown'] < 0.10 else '✗ NO'} ({metrics_dict['Max Drawdown']*100:.2f}%)")
    print(f"  Annualized Return > 15%:   {'✓ YES' if metrics_dict['Annualized Return'] > 0.15 else '✗ NO'} ({metrics_dict['Annualized Return']*100:.2f}%)")
    
    # Save results if requested
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*60}")
        print("Saving Results")
        print("="*60)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics_dict])
        metrics_df.insert(0, 'strategy', f'{args.agent.upper()}_with_options')
        metrics_path = os.path.join(args.output_dir, f'metrics_{timestamp}.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to: {metrics_path}")
        
        # Save visualizations
        print("\nGenerating visualizations...")
        
        # Plot cumulative returns
        fig = plot_cumulative_returns({f'{args.agent.upper()}_Options': results['returns']})
        fig_path = os.path.join(args.output_dir, f'cumulative_returns_{timestamp}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
        
        # Plot portfolio values
        fig = plot_portfolio_values({f'{args.agent.upper()}_Options': results['values']})
        fig_path = os.path.join(args.output_dir, f'portfolio_values_{timestamp}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
        
        # Plot drawdown
        fig = plot_drawdown(results['returns'], title=f"{args.agent.upper()} with Options - Drawdown")
        fig_path = os.path.join(args.output_dir, f'drawdown_{timestamp}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
        
        # Save raw results
        results_path = os.path.join(args.output_dir, f'results_{timestamp}.npz')
        np.savez(
            results_path,
            returns=results['returns'],
            values=results['values'],
            weights=results['weights'],
            turnover=results['turnover'],
            hedge_ratios=results.get('hedge_ratios', np.array([])),
            call_ratios=results.get('call_ratios', np.array([])),
            option_pnls=results.get('option_pnls', np.array([]))
        )
        print(f"Raw results saved to: {results_path}")
        
        print(f"\nAll results saved to: {args.output_dir}")
    
    print("\n" + "="*60)
    print("Evaluation completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
