"""
Visualization utilities for portfolio optimization results.

Provides functions to plot:
- Cumulative returns
- Portfolio allocations over time
- Performance comparison
- Drawdown analysis
- Return distributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_cumulative_returns(
    strategies: Dict[str, np.ndarray],
    title: str = "Cumulative Returns Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot cumulative returns for multiple strategies.
    
    Args:
        strategies: Dictionary mapping strategy names to return arrays.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure (None to display).
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, returns in strategies.items():
        cumulative = np.cumprod(1 + returns)
        ax.plot(cumulative, label=name, linewidth=2)
    
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_portfolio_values(
    strategies: Dict[str, np.ndarray],
    title: str = "Portfolio Value Over Time",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot portfolio values for multiple strategies.
    
    Args:
        strategies: Dictionary mapping strategy names to value arrays.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, values in strategies.items():
        ax.plot(values, label=name, linewidth=2)
    
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_weights_heatmap(
    weights: np.ndarray,
    asset_names: List[str],
    title: str = "Portfolio Allocation Over Time",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot heatmap of portfolio weights over time.
    
    Args:
        weights: Array of shape (n_periods, n_assets).
        asset_names: List of asset names.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Transpose for better visualization
    im = ax.imshow(weights.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Assets")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(asset_names)))
    ax.set_yticklabels(asset_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_weights_stacked(
    weights: np.ndarray,
    asset_names: List[str],
    title: str = "Portfolio Allocation Over Time",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot stacked area chart of portfolio weights.
    
    Args:
        weights: Array of shape (n_periods, n_assets).
        asset_names: List of asset names.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create stacked area plot
    ax.stackplot(
        range(len(weights)),
        weights.T,
        labels=asset_names,
        alpha=0.8
    )
    
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Weight")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_drawdown(
    values: np.ndarray,
    title: str = "Drawdown Analysis",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot drawdown analysis.
    
    Args:
        values: Portfolio values over time.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Calculate running maximum and drawdown
    cummax = np.maximum.accumulate(values)
    drawdown = (cummax - values) / cummax
    
    # Plot portfolio value and running max
    ax1.plot(values, label='Portfolio Value', linewidth=2)
    ax1.plot(cummax, label='Running Maximum', linestyle='--', alpha=0.7)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best', frameon=True)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot drawdown
    ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color='red')
    ax2.plot(drawdown, color='darkred', linewidth=1)
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax2.grid(True, alpha=0.3)
    
    # Mark maximum drawdown
    max_dd_idx = np.argmax(drawdown)
    max_dd = drawdown[max_dd_idx]
    ax2.plot(max_dd_idx, max_dd, 'ro', markersize=8)
    ax2.annotate(
        f'Max DD: {max_dd:.2%}',
        xy=(max_dd_idx, max_dd),
        xytext=(10, -20),
        textcoords='offset points',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        arrowprops=dict(arrowstyle='->', color='red')
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_return_distribution(
    strategies: Dict[str, np.ndarray],
    title: str = "Return Distribution Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot return distributions for multiple strategies.
    
    Args:
        strategies: Dictionary mapping strategy names to return arrays.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    for name, returns in strategies.items():
        axes[0].hist(returns, bins=50, alpha=0.6, label=name, density=True)
    
    axes[0].set_xlabel("Returns")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Return Histograms", fontweight='bold')
    axes[0].legend(loc='best', frameon=True)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_for_box = [returns for returns in strategies.values()]
    labels_for_box = list(strategies.keys())
    
    bp = axes[1].boxplot(
        data_for_box,
        labels=labels_for_box,
        patch_artist=True,
        showmeans=True
    )
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[1].set_ylabel("Returns")
    axes[1].set_title("Return Box Plots", fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    title: str = "Performance Metrics Comparison",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot comparison of performance metrics.
    
    Args:
        metrics_df: DataFrame with metrics (strategies as rows, metrics as columns).
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
        
    Returns:
        Matplotlib figure.
    """
    # Select key metrics to plot
    key_metrics = [
        'Annualized Return',
        'Annualized Volatility',
        'Sharpe Ratio',
        'Max Drawdown',
        'Average Turnover'
    ]
    
    # Filter available metrics
    available_metrics = [m for m in key_metrics if m in metrics_df.columns]
    
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        values = metrics_df[metric]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(values)))
        
        bars = ax.bar(range(len(values)), values, color=colors, edgecolor='black')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(values.index, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(metric, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Format y-axis for percentage metrics
        if 'Return' in metric or 'Volatility' in metric or \
           'Drawdown' in metric or 'Turnover' in metric:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_turnover_analysis(
    strategies: Dict[str, np.ndarray],
    title: str = "Turnover Analysis",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot turnover analysis for multiple strategies.
    
    Args:
        strategies: Dictionary mapping strategy names to turnover arrays.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Time series plot
    for name, turnover in strategies.items():
        ax1.plot(turnover, label=name, alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Turnover")
    ax1.set_title("Turnover Over Time", fontweight='bold')
    ax1.legend(loc='best', frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Average turnover comparison
    avg_turnover = {name: np.mean(to) for name, to in strategies.items()}
    names = list(avg_turnover.keys())
    values = list(avg_turnover.values())
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax2.bar(range(len(names)), values, color=colors, edgecolor='black')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel("Average Turnover")
    ax2.set_title("Average Turnover Comparison", fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom'
        )
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    """Example usage of visualization utilities."""
    
    # Generate sample data
    np.random.seed(42)
    n_periods = 252
    
    # Sample strategies
    strategies_returns = {
        'Strategy A': np.random.normal(0.0005, 0.01, n_periods),
        'Strategy B': np.random.normal(0.0008, 0.015, n_periods),
        'Strategy C': np.random.normal(0.0003, 0.008, n_periods),
    }
    
    strategies_values = {
        name: np.cumprod(1 + returns) * 100000
        for name, returns in strategies_returns.items()
    }
    
    # Plot cumulative returns
    print("Plotting cumulative returns...")
    plot_cumulative_returns(strategies_returns)
    
    # Plot portfolio values
    print("Plotting portfolio values...")
    plot_portfolio_values(strategies_values)
    
    # Plot return distribution
    print("Plotting return distributions...")
    plot_return_distribution(strategies_returns)
    
    plt.show()
    print("Visualization examples complete!")
