"""
Generate comprehensive visualizations for DDPG V2 model.
Creates portfolio performance charts, risk metrics, and comparison plots.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.portfolio_env_with_options import PortfolioWithOptionsEnv
from stable_baselines3 import DDPG

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_config(config_path):
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_environment(config, test_data):
    """Create portfolio environment with options."""
    env_config = config['environment']
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
        enable_options=options_config.get('enabled', True),
        option_expiry_days=options_config.get('protective_puts', {}).get('expiry_days', 30),
        option_transaction_cost=0.005,
        max_hedge_ratio=options_config.get('protective_puts', {}).get('max_hedge_ratio', 1.0),
        max_call_ratio=options_config.get('covered_calls', {}).get('max_coverage_ratio', 1.0),
        put_moneyness=options_config.get('protective_puts', {}).get('strike_pct', 0.95),
        call_moneyness=options_config.get('covered_calls', {}).get('strike_pct', 1.05),
        volatility_lookback=20,
    )
    
    return env

def run_episode(env, model):
    """Run a full episode and collect data."""
    obs, _ = env.reset()
    done = False
    
    portfolio_values = []
    dates = []
    weights_history = []
    hedge_ratios = []
    call_coverages = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        portfolio_values.append(env.portfolio_value)
        dates.append(env.current_step)
        weights_history.append(env.weights.copy())
        hedge_ratios.append(info.get('hedge_ratio', 0))
        call_coverages.append(info.get('call_coverage', 0))
    
    return {
        'portfolio_values': np.array(portfolio_values),
        'dates': dates,
        'weights': np.array(weights_history),
        'hedge_ratios': np.array(hedge_ratios),
        'call_coverages': np.array(call_coverages),
        'returns': env.portfolio_returns,
    }

def plot_portfolio_value(results, test_dates, output_dir):
    """Plot portfolio value over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    portfolio_values = results['portfolio_values']
    dates = test_dates[:len(portfolio_values)]
    
    # Plot portfolio value
    ax.plot(dates, portfolio_values, label='DDPG V2 Portfolio', linewidth=2, color='#2E86AB')
    
    # Plot buy and hold SPY for comparison
    initial_value = portfolio_values[0]
    ax.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.5, label='Initial Value')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax.set_title('DDPG V2 Portfolio Performance (2020-2024)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_value - 1) * 100
    ax.text(0.02, 0.98, f'Total Return: {total_return:.2f}%\nFinal Value: ${final_value:,.2f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'portfolio_value.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved portfolio_value.png")
    plt.close()

def plot_drawdown(results, test_dates, output_dir):
    """Plot drawdown over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    portfolio_values = results['portfolio_values']
    dates = test_dates[:len(portfolio_values)]
    
    # Calculate drawdown
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cummax) / cummax * 100
    
    # Plot drawdown
    ax.fill_between(dates, 0, drawdown, color='#A23B72', alpha=0.6, label='Drawdown')
    ax.plot(dates, drawdown, color='#A23B72', linewidth=1.5)
    
    # Mark maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    max_dd = drawdown[max_dd_idx]
    ax.scatter(dates[max_dd_idx], max_dd, color='red', s=100, zorder=5, label=f'Max DD: {max_dd:.2f}%')
    
    # Add 10% target line
    ax.axhline(y=-10, color='orange', linestyle='--', linewidth=2, label='10% Target', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_title('DDPG V2 Drawdown Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdown.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved drawdown.png")
    plt.close()

def plot_rolling_metrics(results, test_dates, output_dir):
    """Plot rolling Sharpe ratio and volatility."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    returns = np.array(results['returns'])
    dates = test_dates[:len(returns)]
    
    # Calculate rolling metrics (60-day window)
    window = 60
    rolling_sharpe = []
    rolling_vol = []
    
    for i in range(window, len(returns)):
        window_returns = returns[i-window:i]
        mean_return = np.mean(window_returns) * 252
        vol = np.std(window_returns) * np.sqrt(252)
        sharpe = (mean_return - 0.02) / vol if vol > 0 else 0
        rolling_sharpe.append(sharpe)
        rolling_vol.append(vol * 100)
    
    plot_dates = dates[window:]
    
    # Plot rolling Sharpe
    ax1.plot(plot_dates, rolling_sharpe, color='#2E86AB', linewidth=2, label='Rolling Sharpe (60d)')
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Target (1.0)', alpha=0.7)
    ax1.axhline(y=0.9881, color='orange', linestyle='--', linewidth=1.5, label='DDPG V2 (0.99)', alpha=0.7)
    ax1.fill_between(plot_dates, 0, rolling_sharpe, where=(np.array(rolling_sharpe) >= 1.0), 
                      color='green', alpha=0.2, interpolate=True)
    ax1.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Rolling 60-Day Sharpe Ratio', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot rolling volatility
    ax2.plot(plot_dates, rolling_vol, color='#A23B72', linewidth=2, label='Rolling Volatility (60d)')
    ax2.axhline(y=12.49, color='orange', linestyle='--', linewidth=1.5, label='DDPG V2 Avg (12.49%)', alpha=0.7)
    ax2.fill_between(plot_dates, 0, rolling_vol, color='#A23B72', alpha=0.3)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Volatility (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Rolling 60-Day Volatility', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rolling_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved rolling_metrics.png")
    plt.close()

def plot_weights_heatmap(results, asset_names, test_dates, output_dir):
    """Plot portfolio weights over time as heatmap."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    weights = results['weights']
    dates = test_dates[:len(weights)]
    
    # Sample every 5 days for readability
    sample_indices = range(0, len(weights), 5)
    weights_sampled = weights[sample_indices]
    dates_sampled = [dates[i] for i in sample_indices]
    
    # Create heatmap
    im = ax.imshow(weights_sampled.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    
    # Set ticks
    ax.set_yticks(range(len(asset_names)))
    ax.set_yticklabels(asset_names, fontsize=9)
    
    # Set x-axis to show dates
    x_tick_indices = range(0, len(dates_sampled), len(dates_sampled) // 10)
    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels([dates_sampled[i].strftime('%Y-%m') for i in x_tick_indices], rotation=45, ha='right')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight', rotation=270, labelpad=15, fontsize=11, fontweight='bold')
    
    # Labels
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Assets', fontsize=12, fontweight='bold')
    ax.set_title('DDPG V2 Portfolio Weights Over Time', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weights_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved weights_heatmap.png")
    plt.close()

def plot_options_usage(results, test_dates, output_dir):
    """Plot options usage over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    hedge_ratios = results['hedge_ratios']
    call_coverages = results['call_coverages']
    dates = test_dates[:len(hedge_ratios)]
    
    # Plot hedge ratios
    ax1.fill_between(dates, 0, hedge_ratios * 100, color='#F18F01', alpha=0.6, label='Protective Puts')
    ax1.plot(dates, hedge_ratios * 100, color='#F18F01', linewidth=1.5)
    ax1.axhline(y=25, color='red', linestyle='--', linewidth=1.5, label='Max Hedge (25%)', alpha=0.7)
    ax1.set_ylabel('Hedge Ratio (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Protective Puts Usage', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot call coverages
    ax2.fill_between(dates, 0, call_coverages * 100, color='#6A994E', alpha=0.6, label='Covered Calls')
    ax2.plot(dates, call_coverages * 100, color='#6A994E', linewidth=1.5)
    ax2.axhline(y=25, color='red', linestyle='--', linewidth=1.5, label='Max Coverage (25%)', alpha=0.7)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Coverage Ratio (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Covered Calls Usage', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'options_usage.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved options_usage.png")
    plt.close()

def plot_performance_summary(output_dir):
    """Create performance summary comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = ['DDPG V1', 'DDPG V2', 'DDPG V3', 'PPO V1', 'PPO V3']
    sharpe = [0.9356, 0.9881, 0.9680, 1.0574, 1.0370]
    dd = [9.50, 9.09, 9.54, 18.50, 18.22]
    returns = [14.18, 14.34, 14.12, 15.89, 14.95]
    
    x = np.arange(len(models))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, sharpe, width, label='Sharpe Ratio', color='#2E86AB')
    bars2 = ax.bar(x, [r/10 for r in returns], width, label='Return / 10', color='#6A994E')
    bars3 = ax.bar(x + width, [10-d/10 for d in dd], width, label='(10 - DD) / 10', color='#F18F01')
    
    # Add target line for Sharpe
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Sharpe Target (1.0)')
    
    # Labels and formatting
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved performance_comparison.png")
    plt.close()

def main():
    """Main visualization generation function."""
    print("=" * 60)
    print("  DDPG V2 Visualization Generator")
    print("=" * 60)
    print()
    
    # Paths
    config_path = 'configs/config_ddpg_v2.yaml'
    model_path = 'models_ddpg_v2/ddpg_options_20251121_180644_final.zip'
    output_dir = 'visualizations_ddpg_v2'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Load data
    print("Loading data...")
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
    train_ratio = data_config['train_ratio']
    split_index = int(len(prices) * train_ratio)
    
    test_prices = prices.iloc[split_index:]
    test_returns = test_prices.pct_change().fillna(0)
    test_features = features.iloc[split_index:]
    test_dates = pd.to_datetime(test_prices.index)
    
    test_data = {
        'prices': test_prices,
        'returns': test_returns,
        'features': test_features
    }
    
    # Create environment
    print("Creating environment...")
    env = create_environment(config, test_data)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = DDPG.load(model_path, env=env)
    
    # Run episode
    print("Running evaluation episode...")
    results = run_episode(env, model)
    
    # Generate visualizations
    print()
    print("Generating visualizations...")
    print("-" * 60)
    
    plot_portfolio_value(results, test_dates, output_dir)
    plot_drawdown(results, test_dates, output_dir)
    plot_rolling_metrics(results, test_dates, output_dir)
    plot_weights_heatmap(results, data_config['assets'], test_dates, output_dir)
    plot_options_usage(results, test_dates, output_dir)
    plot_performance_summary(output_dir)
    
    print("-" * 60)
    print()
    print("=" * 60)
    print(f"✓ All visualizations saved to: {output_dir}/")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  • portfolio_value.png - Portfolio value over time")
    print("  • drawdown.png - Drawdown analysis")
    print("  • rolling_metrics.png - Rolling Sharpe & volatility")
    print("  • weights_heatmap.png - Asset weights over time")
    print("  • options_usage.png - Options strategy usage")
    print("  • performance_comparison.png - Model comparison")
    print()

if __name__ == '__main__':
    main()
