#!/usr/bin/env python3
"""
Generate comprehensive comparison visualizations for Final Benchmark
Comparing DDPG, PPO, DQN on 2018-2020 test period
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.size'] = 10

# Directories
results_dir = Path("results_final_benchmark")
output_dir = Path("visualizations_final_benchmark")
output_dir.mkdir(exist_ok=True)

def load_metrics(agent_name):
    """Load metrics for an agent"""
    metrics_file = results_dir / f"{agent_name}_options_final_metrics.json"
    if not metrics_file.exists():
        print(f"⚠️  Warning: {metrics_file} not found")
        return None
    
    with open(metrics_file, 'r') as f:
        return json.load(f)

def load_portfolio_values(agent_name):
    """Load portfolio values over time"""
    values_file = results_dir / f"{agent_name}_options_final_portfolio_values.json"
    if not values_file.exists():
        print(f"⚠️  Warning: {values_file} not found")
        return None
    
    with open(values_file, 'r') as f:
        data = json.load(f)
        return np.array(data['dates']), np.array(data['values'])

def load_drawdowns(agent_name):
    """Load drawdown data over time"""
    dd_file = results_dir / f"{agent_name}_options_final_drawdowns.json"
    if not dd_file.exists():
        print(f"⚠️  Warning: {dd_file} not found")
        return None
    
    with open(dd_file, 'r') as f:
        data = json.load(f)
        return np.array(data['dates']), np.array(data['drawdowns'])

def plot_sharpe_comparison(metrics_dict):
    """Plot 1: Sharpe Ratio Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    agents = list(metrics_dict.keys())
    sharpes = [metrics_dict[agent]['sharpe_ratio'] for agent in agents]
    
    colors = ['#2ecc71', '#3498db']  # Green for DDPG, Blue for PPO
    bars = ax.bar(agents, sharpes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add target line
    ax.axhline(y=1.0, color='gold', linestyle='--', linewidth=2, label='Target: 1.0', zorder=0)
    
    # Add value labels on bars
    for bar, sharpe in zip(bars, sharpes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{sharpe:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Sharpe Ratio Comparison (2018-2020 Test Period)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sharpe_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: sharpe_comparison.png")
    plt.close()

def plot_all_metrics_comparison(metrics_dict):
    """Plot 2: All Metrics Comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Comprehensive Metrics Comparison (2018-2020 Test Period)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    agents = list(metrics_dict.keys())
    colors = ['#2ecc71', '#3498db']  # Green for DDPG, Blue for PPO
    
    metrics = [
        ('sharpe_ratio', 'Sharpe Ratio', 1.0, 'Target: 1.0'),
        ('total_return', 'Total Return (%)', 15.0, 'Target: 15%'),
        ('max_drawdown', 'Max Drawdown (%)', 10.0, 'Target: <10%'),
        ('volatility', 'Volatility (%)', None, None),
        ('average_turnover', 'Portfolio Turnover', None, None),
        ('annualized_return', 'Annualized Return (%)', None, None)
    ]
    
    for idx, (metric_key, metric_label, target, target_label) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        values = [metrics_dict[agent][metric_key] for agent in agents]
        
        # No need to convert - already in percentage format from evaluation script
        
        bars = ax.bar(agents, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add target line if exists
        if target:
            ax.axhline(y=target, color='gold', linestyle='--', linewidth=2, 
                      label=target_label, zorder=0)
            ax.legend(fontsize=9)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.02,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: all_metrics_comparison.png")
    plt.close()

def plot_cumulative_values(values_dict):
    """Plot 3: Cumulative Portfolio Values Over Time"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = {'DDPG': '#2ecc71', 'PPO': '#3498db'}
    linestyles = {'DDPG': '-', 'PPO': '--'}
    
    for agent in values_dict.keys():
        dates, values = values_dict[agent]
        
        # Normalize to start at $100,000
        normalized_values = values / values[0] * 100000
        
        ax.plot(range(len(dates)), normalized_values, 
                label=f'{agent}', 
                color=colors[agent], 
                linestyle=linestyles[agent],
                linewidth=2.5, 
                alpha=0.9)
    
    # Add baseline
    ax.axhline(y=100000, color='gray', linestyle=':', linewidth=2, 
              label='Initial: $100K', alpha=0.6, zorder=0)
    
    # Formatting
    ax.set_xlabel('Trading Days (2018-2020)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Portfolio Value Comparison (2018-2020 Test Period)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_portfolio_values.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: cumulative_portfolio_values.png")
    plt.close()

def plot_drawdowns(drawdown_dict):
    """Plot 4: Drawdown Over Time"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = {'DDPG': '#2ecc71', 'PPO': '#3498db'}
    linestyles = {'DDPG': '-', 'PPO': '--'}
    
    for agent in drawdown_dict.keys():
        dates, drawdowns = drawdown_dict[agent]
        
        # Drawdowns are already in percentage format from the JSON
        drawdowns_pct = drawdowns
        
        ax.plot(range(len(dates)), -drawdowns_pct,  # Negative for downward direction
                label=f'{agent}', 
                color=colors[agent], 
                linestyle=linestyles[agent],
                linewidth=2.5, 
                alpha=0.9)
    
    # Add target line
    ax.axhline(y=-10, color='gold', linestyle='--', linewidth=2, 
              label='Target: -10%', zorder=0)
    
    # Formatting
    ax.set_xlabel('Trading Days (2018-2020)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_title('Drawdown Over Time (2018-2020 Test Period)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Fill areas
    for agent in drawdown_dict.keys():
        dates, drawdowns = drawdown_dict[agent]
        drawdowns_pct = -drawdowns  # Already in percentage
        ax.fill_between(range(len(dates)), drawdowns_pct, 0, 
                        color=colors[agent], alpha=0.1)
    
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'drawdown_over_time.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: drawdown_over_time.png")
    plt.close()

def plot_metrics_table(metrics_dict):
    """Plot 5: Summary Metrics Table"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    agents = list(metrics_dict.keys())
    
    # Prepare data
    table_data = []
    row_labels = [
        'Sharpe Ratio',
        'Total Return (%)',
        'Annualized Return (%)',
        'Max Drawdown (%)',
        'Volatility (%)',
        'Average Turnover',
        'Final Portfolio Value',
        'Total Option P&L',
        'Avg Protective Puts',
        'Avg Covered Calls'
    ]
    
    metric_keys = [
        'sharpe_ratio',
        'total_return',
        'annualized_return',
        'max_drawdown',
        'volatility',
        'average_turnover',
        'final_portfolio_value',
        'total_option_pnl',
        'average_protective_puts',
        'average_covered_calls'
    ]
    
    for label, key in zip(row_labels, metric_keys):
        row = [label]
        for agent in agents:
            value = metrics_dict[agent].get(key, 0)
            
            # Format based on metric type
            # Values are already in percentage format from evaluation script
            if 'return' in key or 'drawdown' in key or 'volatility' in key:
                row.append(f'{value:.2f}%')
            elif 'protective_puts' in key or 'covered_calls' in key or 'turnover' in key:
                # Options ratios and turnover - show as percentage with 2 decimals (multiply by 100)
                row.append(f'{value*100:.2f}%')
            elif 'ratio' in key:
                row.append(f'{value:.4f}')
            elif 'value' in key or 'pnl' in key:
                # Format as currency
                row.append(f'${value:,.0f}')
            else:
                row.append(f'{value:.2f}')
        
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Metric'] + agents,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(len(agents) + 1):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Color metric names
    for i in range(1, len(table_data) + 1):
        cell = table[(i, 0)]
        cell.set_facecolor('#ecf0f1')
        cell.set_text_props(weight='bold')
    
    # Color cells based on agent
    colors = {'DDPG': '#d5f4e6', 'PPO': '#d6eaf8'}
    for i in range(1, len(table_data) + 1):
        for j, agent in enumerate(agents, start=1):
            cell = table[(i, j)]
            cell.set_facecolor(colors[agent])
    
    plt.title('Final Benchmark: Comprehensive Metrics Comparison (2018-2020)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: metrics_summary_table.png")
    plt.close()

def main():
    print("\n" + "="*60)
    print("  Final Benchmark Visualization Generator")
    print("="*60 + "\n")
    
    # Load all data
    agents = ['DDPG', 'PPO']
    
    print("Loading metrics...")
    metrics_dict = {}
    for agent in agents:
        metrics = load_metrics(agent.lower())
        if metrics:
            metrics_dict[agent] = metrics
            print(f"  ✓ Loaded {agent} metrics")
    
    if len(metrics_dict) == 0:
        print("\n⚠️  No metrics found! Make sure training has completed.")
        print("   Expected files: results_final_benchmark/*_metrics.json")
        return
    
    print("\nLoading portfolio values...")
    values_dict = {}
    for agent in agents:
        values_data = load_portfolio_values(agent.lower())
        if values_data:
            values_dict[agent] = values_data
            print(f"  ✓ Loaded {agent} portfolio values")
    
    print("\nLoading drawdown data...")
    drawdown_dict = {}
    for agent in agents:
        dd_data = load_drawdowns(agent.lower())
        if dd_data:
            drawdown_dict[agent] = dd_data
            print(f"  ✓ Loaded {agent} drawdown data")
    
    print("\n" + "="*60)
    print("  Generating Visualizations")
    print("="*60 + "\n")
    
    # Generate plots
    if metrics_dict:
        plot_sharpe_comparison(metrics_dict)
        plot_all_metrics_comparison(metrics_dict)
        plot_metrics_table(metrics_dict)
    
    if values_dict:
        plot_cumulative_values(values_dict)
    
    if drawdown_dict:
        plot_drawdowns(drawdown_dict)
    
    print("\n" + "="*60)
    print("  ✅ All visualizations generated successfully!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  1. sharpe_comparison.png")
    print("  2. all_metrics_comparison.png")
    print("  3. cumulative_portfolio_values.png")
    print("  4. drawdown_over_time.png")
    print("  5. metrics_summary_table.png")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
