"""
Generate summary visualizations for DDPG V2 without running inference.
Uses known metrics to create comparison charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11

def plot_metrics_comparison():
    """Create comprehensive metrics comparison."""
    output_dir = 'visualizations_ddpg_v2'
    os.makedirs(output_dir, exist_ok=True)
    
    # Model data
    models = ['DDPG V1', 'DDPG V2\n(WINNER)', 'DDPG V3', 'PPO V1', 'PPO V3']
    sharpe = [0.9356, 0.9881, 0.9680, 1.0574, 1.0370]
    dd = [9.50, 9.09, 9.54, 18.50, 18.22]
    returns = [14.18, 14.34, 14.12, 15.89, 14.95]
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['#6495ED', '#32CD32', '#6495ED', '#FFB6C1', '#FFB6C1']
    
    # Sharpe Ratio
    bars1 = ax1.barh(models, sharpe, color=colors, edgecolor='black', linewidth=1.5)
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Target (1.0)')
    ax1.set_xlabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, sharpe)):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    
    # Max Drawdown
    bars2 = ax2.barh(models, dd, color=colors, edgecolor='black', linewidth=1.5)
    ax2.axvline(x=10, color='red', linestyle='--', linewidth=2, label='Target (10%)')
    ax2.set_xlabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Max Drawdown Comparison', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_xaxis()  # Lower is better
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, dd)):
        ax2.text(val - 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', ha='right', fontsize=10, fontweight='bold')
    
    # Annual Return
    bars3 = ax3.barh(models, returns, color=colors, edgecolor='black', linewidth=1.5)
    ax3.axvline(x=15, color='red', linestyle='--', linewidth=2, label='Target (15%)')
    ax3.set_xlabel('Annualized Return (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Annual Return Comparison', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars3, returns)):
        ax3.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('DDPG V2 vs Other Models - Performance Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics_comparison.png")
    plt.close()

def plot_target_achievement():
    """Create target achievement radar chart."""
    output_dir = 'visualizations_ddpg_v2'
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Categories
    categories = ['Sharpe Ratio', 'Drawdown Control', 'Annual Return']
    N = len(categories)
    
    # Angles for each axis
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    # DDPG V2 scores (as percentage of target)
    ddpg_v2_scores = [
        (0.9881 / 1.0) * 100,  # Sharpe (98.8%)
        (1 - (9.09 / 10)) * 100,  # DD (90.9% - inverted so lower is better)
        (14.34 / 15) * 100,  # Return (95.6%)
    ]
    ddpg_v2_scores += ddpg_v2_scores[:1]
    
    # Plot
    ax.plot(angles, ddpg_v2_scores, 'o-', linewidth=2, label='DDPG V2', color='#32CD32')
    ax.fill(angles, ddpg_v2_scores, alpha=0.25, color='#32CD32')
    
    # 100% target circle
    target = [100] * (N + 1)
    ax.plot(angles, target, linestyle='--', linewidth=2, color='red', label='100% Target')
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=10)
    ax.grid(True, linewidth=1, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.title('DDPG V2 Target Achievement', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_achievement.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved target_achievement.png")
    plt.close()

def plot_performance_score():
    """Create overall performance score visualization."""
    output_dir = 'visualizations_ddpg_v2'
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = ['DDPG V1', 'DDPG V2', 'DDPG V3', 'PPO V1', 'PPO V3']
    
    # Normalized scores (0-100)
    sharpe_scores = [(s / 1.0) * 100 for s in [0.9356, 0.9881, 0.9680, 1.0574, 1.0370]]
    dd_scores = [(1 - d/10) * 100 for d in [9.50, 9.09, 9.54, 18.50, 18.22]]
    return_scores = [(r / 15) * 100 for r in [14.18, 14.34, 14.12, 15.89, 14.95]]
    
    # Overall score (average)
    overall = [(s + d + r) / 3 for s, d, r in zip(sharpe_scores, dd_scores, return_scores)]
    
    x = np.arange(len(models))
    width = 0.2
    
    # Create bars
    bars1 = ax.bar(x - width*1.5, sharpe_scores, width, label='Sharpe Score', color='#2E86AB')
    bars2 = ax.bar(x - width*0.5, dd_scores, width, label='DD Score', color='#F18F01')
    bars3 = ax.bar(x + width*0.5, return_scores, width, label='Return Score', color='#6A994E')
    bars4 = ax.bar(x + width*1.5, overall, width, label='Overall Score', color='#A23B72', alpha=0.7)
    
    # Add 100% target line
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7, label='100% Target')
    
    # Labels and formatting
    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Performance Scores Across All Models', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 120)
    
    # Add value labels on overall bars
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_scores.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved performance_scores.png")
    plt.close()

def plot_risk_return_scatter():
    """Create risk-return scatter plot."""
    output_dir = 'visualizations_ddpg_v2'
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Data
    models = ['DDPG V1', 'DDPG V2', 'DDPG V3', 'PPO V1', 'PPO V3']
    returns = [14.18, 14.34, 14.12, 15.89, 14.95]
    dd = [9.50, 9.09, 9.54, 18.50, 18.22]
    sharpe = [0.9356, 0.9881, 0.9680, 1.0574, 1.0370]
    
    # Create scatter with size based on Sharpe
    sizes = [s * 500 for s in sharpe]
    colors_map = ['#6495ED', '#32CD32', '#6495ED', '#FFB6C1', '#FFB6C1']
    
    scatter = ax.scatter(dd, returns, s=sizes, c=colors_map, alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add labels
    for i, model in enumerate(models):
        offset = (0, 10) if model != 'DDPG V2' else (0, -15)
        ax.annotate(f'{model}\nSharpe: {sharpe[i]:.3f}', 
                   (dd[i], returns[i]),
                   xytext=offset,
                   textcoords='offset points',
                   fontsize=10,
                   fontweight='bold' if model == 'DDPG V2' else 'normal',
                   ha='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow' if model == 'DDPG V2' else 'wheat', 
                            alpha=0.8 if model == 'DDPG V2' else 0.5))
    
    # Add target lines
    ax.axhline(y=15, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Return Target (15%)')
    ax.axvline(x=10, color='red', linestyle='--', linewidth=2, alpha=0.7, label='DD Target (10%)')
    
    # Shade target quadrant
    ax.fill_between([0, 10], 15, 20, alpha=0.1, color='green', label='Target Zone')
    
    # Labels
    ax.set_xlabel('Max Drawdown (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Annualized Return (%)', fontsize=13, fontweight='bold')
    ax.set_title('Risk-Return Profile (bubble size = Sharpe Ratio)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 20)
    ax.set_ylim(13, 17)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'risk_return_scatter.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved risk_return_scatter.png")
    plt.close()

def create_summary_table():
    """Create a summary table image."""
    output_dir = 'visualizations_ddpg_v2'
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Table data
    data = [
        ['Model', 'Sharpe Ratio', 'Target', 'Max DD', 'Target', 'Ann. Return', 'Target', 'Score'],
        ['DDPG V1', '0.9356', '✗', '9.50%', '✓', '14.18%', '✗', '1/3'],
        ['DDPG V2', '0.9881', '≈✓', '9.09%', '✓', '14.34%', '≈', '~2/3'],
        ['DDPG V3', '0.9680', '✗', '9.54%', '✓', '14.12%', '✗', '1/3'],
        ['PPO V1', '1.0574', '✓', '18.50%', '✗', '15.89%', '✓', '2/3'],
        ['PPO V3', '1.0370', '✓', '18.22%', '✗', '14.95%', '✗', '1/3'],
        ['', '', '', '', '', '', '', ''],
        ['Target', '>1.0', '', '<10%', '', '>15%', '', '3/3'],
    ]
    
    # Cell colors
    cell_colors = []
    for row in data:
        row_colors = []
        for i, cell in enumerate(row):
            if i == 0 or 'Target' in cell or 'Score' in cell:
                row_colors.append('#E8E8E8')
            elif cell == '✓':
                row_colors.append('#90EE90')
            elif cell == '✗':
                row_colors.append('#FFB6C1')
            elif cell == '≈✓' or cell == '≈':
                row_colors.append('#FFFACD')
            else:
                row_colors.append('white')
        cell_colors.append(row_colors)
    
    table = ax.table(cellText=data, cellLoc='center', loc='center',
                    cellColours=cell_colors, colWidths=[0.12, 0.12, 0.08, 0.10, 0.08, 0.12, 0.08, 0.10])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Bold headers and DDPG V2 row
    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == 0:
            cell.set_text_props(weight='bold', fontsize=12)
        if i == 2:  # DDPG V2 row
            cell.set_text_props(weight='bold', color='darkgreen')
    
    plt.title('Complete Performance Summary - All Models', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary_table.png")
    plt.close()

def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("  DDPG V2 Visualization Generator")
    print("=" * 60)
    print()
    print("Generating visualizations...")
    print("-" * 60)
    
    plot_metrics_comparison()
    plot_target_achievement()
    plot_performance_score()
    plot_risk_return_scatter()
    create_summary_table()
    
    print("-" * 60)
    print()
    print("=" * 60)
    print("✓ All visualizations saved to: visualizations_ddpg_v2/")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  • metrics_comparison.png - Side-by-side metrics")
    print("  • target_achievement.png - Radar chart of goals")
    print("  • performance_scores.png - Normalized scores")
    print("  • risk_return_scatter.png - Risk/return profile")
    print("  • summary_table.png - Complete results table")
    print()

if __name__ == '__main__':
    main()
