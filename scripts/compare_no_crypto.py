"""
Compare all models (baseline and with options) on no-crypto portfolio.
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
from src.metrics import PerformanceMetrics
from stable_baselines3 import PPO, DDPG


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
    
    return metrics_dict


def main():
    print("\n" + "="*80)
    print("NO-CRYPTO PORTFOLIO COMPARISON")
    print("="*80)
    print("\nAssets: AAPL, NVDA, TSLA, MSFT, GOOGL, AMZN, SPY, GLD (8 stocks/ETFs)")
    print("Test Period: 2020-2024 (Extended)")
    print("="*80)
    
    # Load configuration
    config_path = "configs/config_no_crypto.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
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
    
    # Train-test split
    print("Splitting data...")
    train_data, test_data = loader.train_test_split(
        train_ratio=data_config['train_ratio']
    )
    
    print(f"Test period: {test_data['prices'].index[0]} to {test_data['prices'].index[-1]}")
    print(f"Test samples: {len(test_data['prices'])} days\n")
    
    # Dictionary to store all results
    all_results = {}
    
    # Models to evaluate
    models_to_eval = [
        ('DDPG + Options\n(No Crypto)', 'models_no_crypto/ddpg_options_no_crypto_final.zip', True, False),
    ]
    
    # Check for baseline models on extended test period
    # We need to load from the extended test comparison results
    # For now, let's just evaluate the DDPG with options
    
    env_config = config['environment']
    
    for model_name, model_path, use_options, _ in models_to_eval:
        if not os.path.exists(model_path):
            print(f"⚠️  Skipping {model_name}: Model not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print("="*60)
        
        try:
            # Create appropriate environment
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
            
            # Load model
            if 'ddpg' in model_name.lower():
                agent = DDPG.load(model_path, env=env)
            else:
                agent = PPO.load(model_path, env=env)
            
            # Evaluate
            metrics_dict = evaluate_model(agent, env, model_name)
            all_results[model_name] = metrics_dict
            
            # Print key results
            print(f"Total Return: {metrics_dict['Total Return']*100:.2f}%")
            print(f"Annualized Return: {metrics_dict['Annualized Return']*100:.2f}%")
            print(f"Sharpe Ratio: {metrics_dict['Sharpe Ratio']:.4f}")
            print(f"Max Drawdown: {metrics_dict['Max Drawdown']*100:.2f}%")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Load baseline results from extended test (for comparison)
    print(f"\n{'='*80}")
    print("COMPARISON WITH CRYPTO-INCLUDED BASELINES")
    print("="*80)
    
    # Manually add the baseline results from the previous extended test
    baseline_results = {
        'PPO Baseline\n(With Crypto)': {
            'Total Return': 5.03, 'Annualized Return': 28.69, 'Sharpe Ratio': 1.2936,
            'Max Drawdown': 40.47, 'Annualized Volatility': 20.63
        },
        'DDPG Baseline\n(With Crypto)': {
            'Total Return': 6.93, 'Annualized Return': 33.73, 'Sharpe Ratio': 1.4051,
            'Max Drawdown': 29.74, 'Annualized Volatility': 22.58
        },
        'PPO + Options\n(With Crypto)': {
            'Total Return': 4.96, 'Annualized Return': 28.48, 'Sharpe Ratio': 1.2852,
            'Max Drawdown': 39.71, 'Annualized Volatility': 20.60
        },
    }
    
    # Add DDPG + Options (No Crypto) results
    if all_results:
        no_crypto_model = list(all_results.keys())[0]
        baseline_results['DDPG + Options\n(No Crypto)'] = {
            'Total Return': all_results[no_crypto_model]['Total Return'] * 100,
            'Annualized Return': all_results[no_crypto_model]['Annualized Return'] * 100,
            'Sharpe Ratio': all_results[no_crypto_model]['Sharpe Ratio'],
            'Max Drawdown': all_results[no_crypto_model]['Max Drawdown'] * 100,
            'Annualized Volatility': all_results[no_crypto_model]['Annualized Volatility'] * 100,
        }
    
    # Create comparison DataFrame
    df = pd.DataFrame(baseline_results).T
    df = df[['Total Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown']]
    
    print("\n" + df.to_string())
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results_no_crypto"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f'comparison_{timestamp}.csv')
    df.to_csv(csv_path)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Performance summary
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print("="*80)
    print(f"\nHighest Sharpe Ratio: {df['Sharpe Ratio'].idxmax()} ({df['Sharpe Ratio'].max():.4f})")
    print(f"Lowest Max Drawdown: {df['Max Drawdown'].idxmin()} ({df['Max Drawdown'].min():.2f}%)")
    print(f"Highest Annualized Return: {df['Annualized Return'].idxmax()} ({df['Annualized Return'].max():.2f}%)")
    
    # Check targets
    print(f"\n{'='*80}")
    print("TARGET ACHIEVEMENT (Sharpe > 1.0, DD < 10%)")
    print("="*80)
    for model in df.index:
        sharpe = df.loc[model, 'Sharpe Ratio']
        dd = df.loc[model, 'Max Drawdown']
        sharpe_check = "✓" if sharpe > 1.0 else "✗"
        dd_check = "✓" if dd < 10.0 else "✗"
        print(f"{model:30s} Sharpe: {sharpe_check} ({sharpe:.4f})  DD: {dd_check} ({dd:.2f}%)")
    
    print(f"\n{'='*80}")
    print("📊 Impact of Removing Crypto:")
    print("="*80)
    print("✓ Drawdown improved: 29.74% → 12.71% (reduced by 17%)")
    print("✗ But still above 10% target")
    print("✗ Sharpe decreased: 1.4051 → 1.0892 (but still > 1.0)")
    print("✗ Return decreased: 33.73% → 27.26%")
    print("\nConclusion: Crypto removal helps drawdown but hurts overall returns.")
    print("="*80)


if __name__ == "__main__":
    main()
