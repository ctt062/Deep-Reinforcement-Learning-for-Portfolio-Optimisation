"""
Utility script for running portfolio optimization experiments.

This script provides helper functions for batch experiments.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.portfolio_env import PortfolioEnv
from src.agents import create_agent, train_agent
from src.benchmarks import run_all_benchmarks
from src.metrics import compare_strategies
from src.visualization import *


def run_experiment(
    agent_type: str = 'ppo',
    assets: List[str] = None,
    train_timesteps: int = 50000,
    transaction_costs: List[float] = [0.001],
    save_results: bool = True,
    output_dir: str = '../results'
):
    """
    Run a complete experiment with specified parameters.
    
    Args:
        agent_type: Type of DRL agent ('ppo', 'ddpg', 'dqn').
        assets: List of asset tickers.
        train_timesteps: Number of training timesteps.
        transaction_costs: List of transaction costs to test.
        save_results: Whether to save results.
        output_dir: Directory to save results.
    """
    if assets is None:
        assets = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'SPY', 'GLD']
    
    print(f"\n{'='*80}")
    print(f"Running Experiment: {agent_type.upper()} on {len(assets)} assets")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading data...")
    loader = DataLoader(
        assets=assets,
        start_date='2015-01-01',
        end_date='2024-12-31',
        data_dir='../data'
    )
    
    prices = loader.download_data()
    features = loader.build_features()
    train_data, test_data = loader.train_test_split(train_ratio=0.7)
    
    results = {}
    
    for tc in transaction_costs:
        print(f"\n{'='*80}")
        print(f"Transaction Cost: {tc*100}%")
        print(f"{'='*80}\n")
        
        # Create training environment
        train_env = PortfolioEnv(
            prices=train_data['prices'],
            returns=train_data['returns'],
            features=train_data['features'],
            transaction_cost=tc,
            lookback_window=20,
        )
        
        # Train agent
        print(f"Training {agent_type.upper()} agent...")
        agent = create_agent(agent_type, train_env)
        agent.learn(total_timesteps=train_timesteps, log_interval=10)
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_env = PortfolioEnv(
            prices=test_data['prices'],
            returns=test_data['returns'],
            features=test_data['features'],
            transaction_cost=tc,
            lookback_window=20,
        )
        
        obs, info = test_env.reset()
        done = False
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
        
        history = test_env.get_portfolio_history()
        
        # Run benchmarks
        print("Running benchmarks...")
        benchmark_results = run_all_benchmarks(
            returns=test_data['returns'],
            transaction_cost=tc,
        )
        
        # Combine results
        all_results = {
            f'{agent_type.upper()}': {
                'returns': history['returns'],
                'values': history['values'],
                'weights': history['weights'],
                'turnover': history['turnover'],
            },
            **benchmark_results
        }
        
        # Calculate metrics
        metrics_df = compare_strategies(all_results)
        
        results[f'tc_{tc}'] = {
            'metrics': metrics_df,
            'results': all_results
        }
        
        print(f"\nResults for transaction cost {tc*100}%:")
        print(metrics_df[['Annualized Return', 'Sharpe Ratio', 'Max Drawdown']])
    
    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary
        summary_path = os.path.join(output_dir, f'experiment_{agent_type}_{timestamp}.csv')
        
        summary_data = []
        for tc_key, data in results.items():
            metrics = data['metrics']
            for strategy in metrics.index:
                row = {'Transaction_Cost': tc_key, 'Strategy': strategy}
                row.update(metrics.loc[strategy].to_dict())
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
    
    return results


def compare_agents(
    agent_types: List[str] = ['ppo', 'ddpg'],
    assets: List[str] = None,
    train_timesteps: int = 50000,
    transaction_cost: float = 0.001,
    output_dir: str = '../results'
):
    """
    Compare multiple DRL agents.
    
    Args:
        agent_types: List of agent types to compare.
        assets: List of asset tickers.
        train_timesteps: Number of training timesteps per agent.
        transaction_cost: Transaction cost to use.
        output_dir: Directory to save results.
    """
    if assets is None:
        assets = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'SPY', 'GLD']
    
    print(f"\n{'='*80}")
    print(f"Comparing Agents: {', '.join([a.upper() for a in agent_types])}")
    print(f"{'='*80}\n")
    
    # Load data (shared across agents)
    loader = DataLoader(assets=assets, start_date='2015-01-01', end_date='2024-12-31')
    prices = loader.download_data()
    features = loader.build_features()
    train_data, test_data = loader.train_test_split()
    
    all_results = {}
    
    for agent_type in agent_types:
        print(f"\n{'='*80}")
        print(f"Training {agent_type.upper()}")
        print(f"{'='*80}\n")
        
        # Train
        train_env = PortfolioEnv(
            prices=train_data['prices'],
            returns=train_data['returns'],
            features=train_data['features'],
            transaction_cost=transaction_cost,
        )
        
        agent = create_agent(agent_type, train_env)
        agent.learn(total_timesteps=train_timesteps, log_interval=10)
        
        # Evaluate
        test_env = PortfolioEnv(
            prices=test_data['prices'],
            returns=test_data['returns'],
            features=test_data['features'],
            transaction_cost=transaction_cost,
        )
        
        obs, info = test_env.reset()
        done = False
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
        
        history = test_env.get_portfolio_history()
        
        all_results[agent_type.upper()] = {
            'returns': history['returns'],
            'values': history['values'],
            'weights': history['weights'],
            'turnover': history['turnover'],
        }
    
    # Add benchmarks
    print("\nRunning benchmarks...")
    benchmark_results = run_all_benchmarks(
        returns=test_data['returns'],
        transaction_cost=transaction_cost,
    )
    all_results.update(benchmark_results)
    
    # Compare
    metrics_df = compare_strategies(all_results)
    
    print(f"\n{'='*80}")
    print("Comparison Results")
    print(f"{'='*80}\n")
    print(metrics_df)
    
    # Visualize
    returns_dict = {name: data['returns'] for name, data in all_results.items()}
    values_dict = {name: data['values'] for name, data in all_results.items()}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_cumulative_returns(
        returns_dict,
        save_path=os.path.join(output_dir, f'agent_comparison_{timestamp}.png')
    )
    
    plot_metrics_comparison(
        metrics_df,
        save_path=os.path.join(output_dir, f'metrics_comparison_{timestamp}.png')
    )
    
    metrics_df.to_csv(os.path.join(output_dir, f'metrics_{timestamp}.csv'))
    
    print(f"\nResults saved to: {output_dir}")
    
    return metrics_df, all_results


if __name__ == "__main__":
    """Run sample experiments."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run portfolio optimization experiments")
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'compare'])
    parser.add_argument('--agent', type=str, default='ppo', help='Agent type for single mode')
    parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        print(f"Running single experiment with {args.agent.upper()}")
        results = run_experiment(
            agent_type=args.agent,
            train_timesteps=args.timesteps,
            transaction_costs=[0.001],
            save_results=True
        )
    
    elif args.mode == 'compare':
        print("Comparing multiple agents")
        metrics, results = compare_agents(
            agent_types=['ppo', 'ddpg'],
            train_timesteps=args.timesteps
        )
    
    print("\nExperiment completed!")
