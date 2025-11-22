"""
Training script for portfolio optimization with option overlays.

Trains DRL agents on portfolio environment enhanced with protective puts
and covered calls for superior risk-adjusted returns.

Target: Sharpe > 1.0, Max Drawdown < 10%
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.portfolio_env_with_options import PortfolioWithOptionsEnv
from src.agents import create_agent, train_agent, TrainingCallback


def set_random_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_config(config_path: str = "configs/config_with_options.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_environment(config: dict, train_data: dict):
    """Create portfolio environment with options from config and data."""
    env_config = config['environment']
    options_config = config.get('options', {})
    
    env = PortfolioWithOptionsEnv(
        prices=train_data['prices'],
        returns=train_data['returns'],
        features=train_data['features'],
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


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train DRL agent for portfolio optimization with options"
    )
    parser.add_argument(
        '--agent',
        type=str,
        default='ppo',
        choices=['ppo', 'ddpg'],
        help='Type of DRL agent (PPO or DDPG recommended for continuous actions)'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Total training timesteps (overrides config)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_with_options.yaml',
        help='Path to configuration file'
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
        default='models_with_options',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    
    # Check if options config exists, otherwise use default
    if not os.path.exists(args.config):
        print(f"Options config not found at {args.config}")
        print("Using default config and enabling options...")
        config = load_config('configs/config.yaml')
        # Add options configuration
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
    else:
        config = load_config(args.config)
    
    # Override config with command line arguments
    if args.seed is not None:
        config['seed'] = args.seed
    
    if args.timesteps is not None:
        config['training']['total_timesteps'] = args.timesteps
    
    # Set random seed
    seed = config.get('seed', 42)
    set_random_seed(seed)
    print(f"Random seed set to: {seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.agent}_options_{timestamp}"
    model_path = os.path.join(args.output_dir, model_name)
    
    print("\n" + "="*60)
    print(f"Training {args.agent.upper()} Agent with Options Overlay")
    print("="*60)
    print("Target Performance:")
    print("  - Sharpe Ratio > 1.0")
    print("  - Maximum Drawdown < 10%")
    print("  - Annualized Return > 15%")
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
    train_end = data_config.get('train_end', None)
    if train_end:
        train_data, test_data = loader.train_test_split(train_end=train_end)
    else:
        train_data, test_data = loader.train_test_split(
            train_ratio=data_config['train_ratio']
        )
    
    # Create environment with options
    print("\nCreating environment with options overlay...")
    env = create_environment(config, train_data)
    print(f"Environment created: {env.n_assets} assets, {env.max_steps} max steps")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Options enabled: {env.enable_options}")
    if env.enable_options:
        print(f"  - Protective puts: up to {env.max_hedge_ratio*100:.0f}% hedge")
        print(f"  - Covered calls: up to {env.max_call_ratio*100:.0f}% coverage")
        print(f"  - Put strike: {env.put_moneyness*100:.0f}% of spot")
        print(f"  - Call strike: {env.call_moneyness*100:.0f}% of spot")
    
    # Create agent
    print(f"\nCreating {args.agent.upper()} agent...")
    agent_config = config['agents'][args.agent]
    agent = create_agent(
        agent_type=args.agent,
        env=env,
        config=agent_config,
        device=args.device,
        seed=seed,
    )
    
    print(f"Agent created: {agent.__class__.__name__}")
    print(f"Policy: {agent.policy.__class__.__name__}")
    
    # Create callback
    training_config = config['training']
    callback = TrainingCallback(
        verbose=training_config.get('verbose', 1),
        log_freq=training_config.get('log_interval', 10) * 100,
        save_freq=training_config.get('save_freq', 10000),
        save_path=model_path,
    )
    
    # Train agent
    print("\n" + "="*60)
    print("Starting training with option overlay...")
    print("="*60)
    print(f"Total timesteps: {training_config['total_timesteps']}")
    print(f"Model save path: {model_path}")
    print("="*60 + "\n")
    
    trained_agent = train_agent(
        agent=agent,
        total_timesteps=training_config['total_timesteps'],
        callback=callback,
        log_interval=training_config.get('log_interval', 10),
    )
    
    # Save final model
    final_model_path = f"{model_path}_final.zip"
    trained_agent.save(final_model_path)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Final model saved to: {final_model_path}")
    
    # Quick evaluation on training set
    print("\nQuick evaluation on training set...")
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    total_hedge_cost = 0
    total_call_income = 0
    
    while not done and steps < min(100, env.max_steps):
        action, _ = trained_agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        
        if 'option_pnl' in info:
            if info.get('put_cost', 0) > 0:
                total_hedge_cost += info['put_cost']
            if info.get('call_income', 0) > 0:
                total_call_income += info['call_income']
    
    print(f"Steps: {steps}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final portfolio value: ${info['portfolio_value']:,.2f}")
    if env.enable_options:
        print(f"Total hedge cost: ${total_hedge_cost:,.2f}")
        print(f"Total call income: ${total_call_income:,.2f}")
        print(f"Net option P&L: ${total_call_income - total_hedge_cost:,.2f}")
    
    # Save configuration
    config_save_path = f"{model_path}_config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to: {config_save_path}")
    
    print("\nTraining script completed successfully!")
    print("\nNext steps:")
    print(f"  1. Evaluate: python scripts/evaluate_with_options.py --agent {args.agent} --model-path {final_model_path}")
    print(f"  2. Compare with baseline models in results/")


if __name__ == "__main__":
    main()
