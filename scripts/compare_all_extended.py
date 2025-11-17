"""
Compare all trained models (PPO, DDPG, DQN) with and without options
on an extended test period.

Evaluates on 2019-2024 test period (50/50 split) for more comprehensive testing.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.portfolio_env import PortfolioEnv
from src.portfolio_env_with_options import PortfolioWithOptionsEnv
from src.discrete_wrapper import DiscretePortfolioWrapper
from src.metrics import PerformanceMetrics
from stable_baselines3 import PPO, DDPG, DQN


def load_config(config_path: str = "configs/config_extended_test.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(agent, env, model_name):
    """Evaluate a model and return results."""
    obs, info = env.reset()
    done = False
    
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    history = env.get_portfolio_history()
    
    # Calculate metrics
    metrics = PerformanceMetrics(
        returns=history['returns'],
        values=history['values'],
        weights=history['weights'],
        turnover=history['turnover'],
        risk_free_rate=0.02,
        periods_per_year=252
    )
    
    metrics_dict = metrics.get_all_metrics()
    metrics_dict['Model'] = model_name
    
    return metrics_dict, history


def main():
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON - EXTENDED TEST PERIOD")
    print("="*80)
    print("\nComparing:")
    print("  - PPO, DDPG, DQN (baseline)")
    print("  - PPO, DDPG with options overlay")
    print("\nTest Period: 2020-2024 (Extended from 2022-2024)")
    print("="*80)
    
    # Load configuration with extended test period
    config = load_config()
    
    # Set random seed
    seed = config.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
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
    
    # Build features
    print("Building features...")
    feature_config = config['features']
    features = loader.build_features(
        sma_periods=feature_config['sma_periods'],
        ema_periods=feature_config['ema_periods'],
        momentum_periods=feature_config['momentum_periods'],
        normalize=feature_config['normalize_prices'],
        normalize_method=feature_config['normalize_method'],
        rolling_window=feature_config.get('rolling_window', 60)
    )
    
    # Train-test split with extended test period
    print("Splitting data...")
    train_data, test_data = loader.train_test_split(
        train_ratio=data_config['train_ratio']
    )
    
    print(f"\nExtended Test Period: {test_data['prices'].index[0]} to {test_data['prices'].index[-1]}")
    print(f"Test samples: {len(test_data['prices'])} days")
    
    # Dictionary to store all results
    all_results = {}
    
    # Models to evaluate
    models_to_eval = [
        ('PPO', 'models/ppo_20251116_121651_final.zip', False, False),
        ('DDPG', 'models/ddpg_20251116_174546_final.zip', False, False),
        ('DQN', 'models/dqn_20251116_194054_final.zip', False, True),
        ('PPO + Options', 'models_with_options/ppo_options_20251117_171216_final.zip', True, False),
    ]
    
    # Check for DDPG with options
    ddpg_options_models = [f for f in os.listdir('models_with_options') if f.startswith('ddpg_options') and f.endswith('_final.zip')]
    if ddpg_options_models:
        models_to_eval.append(('DDPG + Options', f'models_with_options/{ddpg_options_models[0]}', True, False))
    
    print(f"\n{'='*80}")
    print("Evaluating Models...")
    print("="*80)
    
    for model_name, model_path, use_options, is_dqn in models_to_eval:
        if not os.path.exists(model_path):
            print(f"\n⚠️  Skipping {model_name}: Model file not found")
            continue
        
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print("="*80)
        
        try:
            # Create appropriate environment
            env_config = config['environment']
            
            if use_options:
                options_config = config.get('options', {})
                env = PortfolioWithOptionsEnv(
                    prices=test_data['prices'],
                    returns=test_data['returns'],
                    features=test_data['features'],
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
            else:
                env = PortfolioEnv(
                    prices=test_data['prices'],
                    returns=test_data['returns'],
                    features=test_data['features'],
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
            
            # Wrap DQN environment
            if is_dqn:
                n_discrete_actions = config['agents'].get('dqn', {}).get('n_discrete_actions', 100)
                env = DiscretePortfolioWrapper(env, n_discrete_actions=n_discrete_actions, strategy="mixed")
            
            # Load model
            if 'ppo' in model_name.lower():
                agent = PPO.load(model_path, env=env)
            elif 'ddpg' in model_name.lower():
                agent = DDPG.load(model_path, env=env)
            elif 'dqn' in model_name.lower():
                agent = DQN.load(model_path, env=env)
            
            # Evaluate
            metrics_dict, history = evaluate_model(agent, env, model_name)
            all_results[model_name] = metrics_dict
            
            # Print results
            print(f"\nFinal Portfolio Value: ${history['values'][-1]:,.2f}")
            print(f"Total Return: {metrics_dict['Total Return']*100:.2f}%")
            print(f"Sharpe Ratio: {metrics_dict['Sharpe Ratio']:.4f}")
            print(f"Max Drawdown: {metrics_dict['Max Drawdown']*100:.2f}%")
            print(f"Annualized Return: {metrics_dict['Annualized Return']*100:.2f}%")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create comparison DataFrame
    print(f"\n{'='*80}")
    print("FINAL COMPARISON - EXTENDED TEST PERIOD")
    print("="*80)
    
    if all_results:
        df = pd.DataFrame(all_results).T
        
        # Reorder columns
        col_order = ['Total Return', 'Annualized Return', 'Annualized Volatility', 
                     'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio',
                     'VaR (95%)', 'CVaR (95%)', 'Average Turnover']
        df = df[col_order]
        
        # Format for display
        print("\n" + df.to_string())
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "results_extended_test"
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, f'comparison_{timestamp}.csv')
        df.to_csv(csv_path)
        print(f"\n✅ Results saved to: {csv_path}")
        
        # Highlight best performers
        print(f"\n{'='*80}")
        print("BEST PERFORMERS")
        print("="*80)
        print(f"Highest Sharpe Ratio: {df['Sharpe Ratio'].idxmax()} ({df['Sharpe Ratio'].max():.4f})")
        print(f"Lowest Max Drawdown: {df['Max Drawdown'].idxmin()} ({df['Max Drawdown'].min()*100:.2f}%)")
        print(f"Highest Return: {df['Annualized Return'].idxmax()} ({df['Annualized Return'].max()*100:.2f}%)")
        
        # Check targets
        print(f"\n{'='*80}")
        print("TARGET ACHIEVEMENT (Sharpe > 1.0, DD < 10%)")
        print("="*80)
        for model in df.index:
            sharpe = df.loc[model, 'Sharpe Ratio']
            dd = df.loc[model, 'Max Drawdown']
            sharpe_check = "✓" if sharpe > 1.0 else "✗"
            dd_check = "✓" if dd < 0.10 else "✗"
            print(f"{model:20s} Sharpe: {sharpe_check} ({sharpe:.4f})  DD: {dd_check} ({dd*100:.2f}%)")
    
    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
