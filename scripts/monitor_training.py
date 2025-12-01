#!/usr/bin/env python3
"""
Training Monitor Script
Shows real-time training progress with estimated time remaining.

Usage:
    python scripts/monitor_training.py
    
Or watch the log file directly:
    tail -f logs/training_progress.log
"""

import os
import sys
import time
import re
import argparse
from datetime import datetime

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_time(seconds):
    """Format seconds into human readable time."""
    if seconds < 0:
        return "calculating..."
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

def parse_log_file(log_path):
    """Parse training log file and extract progress info."""
    if not os.path.exists(log_path):
        return None
    
    info = {
        'current_agent': None,
        'timesteps': 0,
        'total_timesteps': 100000,
        'episodes': 0,
        'mean_reward': 0,
        'best_reward': float('-inf'),
        'completed_agents': [],
        'metrics': {},
        'all_steps': []
    }
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Detect agent training start
            if 'Training and Evaluating' in line:
                # Extract agent name (e.g., "Training and Evaluating DDPG")
                match = re.search(r'Training and Evaluating\s+(\w+)', line)
                if match:
                    info['current_agent'] = match.group(1)
                    info['timesteps'] = 0
                    info['all_steps'] = []
            
            # Parse "Total timesteps: 100,000"
            if 'Total timesteps:' in line:
                match = re.search(r'Total timesteps:\s*([\d,]+)', line)
                if match:
                    info['total_timesteps'] = int(match.group(1).replace(',', ''))
            
            # Parse "Steps: 3000 | Episodes: 997 | Mean Reward: -228.2470"
            if line.startswith('Steps:'):
                match = re.search(r'Steps:\s*(\d+)\s*\|\s*Episodes:\s*(\d+)\s*\|\s*Mean Reward:\s*([-\d.]+)', line)
                if match:
                    steps = int(match.group(1))
                    episodes = int(match.group(2))
                    reward = float(match.group(3))
                    
                    info['timesteps'] = steps
                    info['episodes'] = episodes
                    info['mean_reward'] = reward
                    info['all_steps'].append((steps, reward))
                    
                    if reward > info['best_reward']:
                        info['best_reward'] = reward
            
            # Detect completed training
            if 'Model saved to' in line:
                if info['current_agent']:
                    if info['current_agent'] not in info['completed_agents']:
                        info['completed_agents'].append(info['current_agent'])
            
            # Parse final metrics
            if 'Sharpe Ratio:' in line and 'Target' not in line:
                match = re.search(r'Sharpe Ratio:\s*([-\d.]+)', line)
                if match:
                    info['metrics']['sharpe'] = float(match.group(1))
            
            if 'Total Return:' in line:
                match = re.search(r'Total Return:\s*([-\d.]+)%', line)
                if match:
                    info['metrics']['return'] = float(match.group(1))
            
            if 'Max Drawdown:' in line:
                match = re.search(r'Max Drawdown:\s*([-\d.]+)%', line)
                if match:
                    info['metrics']['drawdown'] = float(match.group(1))
            
            if 'Volatility:' in line:
                match = re.search(r'Volatility:\s*([-\d.]+)%', line)
                if match:
                    info['metrics']['volatility'] = float(match.group(1))
    
    except Exception as e:
        print(f"Error parsing log: {e}")
    
    return info

def display_progress(info, start_time):
    """Display formatted progress information."""
    clear_screen()
    
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("=" * 70)
    print("  🚀 DRL Portfolio Training Monitor")
    print("=" * 70)
    print(f"{Colors.ENDC}")
    
    # Current time and duration
    elapsed = time.time() - start_time
    print(f"\n{Colors.CYAN}⏱️  Elapsed Time: {format_time(elapsed)}{Colors.ENDC}")
    print(f"{Colors.CYAN}📅 Started: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    
    if info is None:
        print(f"\n{Colors.YELLOW}⏳ Waiting for training to start...{Colors.ENDC}")
        print(f"\nRun training with:")
        print(f"  {Colors.GREEN}python scripts/train_and_evaluate_final.py --agents ddpg ppo{Colors.ENDC}")
        return
    
    # Current agent
    agent = info.get('current_agent', 'Unknown')
    print(f"\n{Colors.BOLD}🤖 Current Agent: {Colors.BLUE}{agent.upper() if agent else 'Loading...'}{Colors.ENDC}")
    
    # Progress bar
    timesteps = info.get('timesteps', 0)
    total = info.get('total_timesteps', 100000)
    progress = timesteps / total if total > 0 else 0
    bar_width = 50
    filled = int(bar_width * progress)
    bar = '█' * filled + '░' * (bar_width - filled)
    
    print(f"\n{Colors.BOLD}📊 Training Progress:{Colors.ENDC}")
    print(f"   [{Colors.GREEN}{bar}{Colors.ENDC}] {progress*100:.1f}%")
    print(f"   Timesteps: {timesteps:,} / {total:,}")
    
    # Estimated time remaining
    if progress > 0.01 and elapsed > 10:
        steps_per_sec = timesteps / elapsed
        remaining_steps = total - timesteps
        remaining_time = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
        print(f"   {Colors.YELLOW}⏱️  ETA: {format_time(remaining_time)}{Colors.ENDC}")
        print(f"   Speed: {steps_per_sec:.1f} steps/sec")
    
    # Episodes
    episodes = info.get('episodes', 0)
    print(f"   Episodes: {episodes:,}")
    
    # Rewards
    mean_reward = info.get('mean_reward', 0)
    best_reward = info.get('best_reward', float('-inf'))
    print(f"\n{Colors.BOLD}💰 Rewards:{Colors.ENDC}")
    print(f"   Current Mean: {Colors.GREEN if mean_reward > -100 else Colors.YELLOW}{mean_reward:.2f}{Colors.ENDC}")
    if best_reward != float('-inf'):
        print(f"   Best Seen: {Colors.GREEN}{best_reward:.2f}{Colors.ENDC}")
    
    # Reward trend
    all_steps = info.get('all_steps', [])
    if len(all_steps) >= 3:
        recent = [s[1] for s in all_steps[-5:]]
        older = [s[1] for s in all_steps[-10:-5]] if len(all_steps) >= 10 else [s[1] for s in all_steps[:len(all_steps)//2]]
        if older:
            trend = sum(recent)/len(recent) - sum(older)/len(older)
            trend_symbol = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
            trend_color = Colors.GREEN if trend > 0 else Colors.RED if trend < 0 else Colors.YELLOW
            print(f"   Trend: {trend_symbol} {trend_color}{trend:+.2f}{Colors.ENDC}")
    
    # Completed agents
    completed = info.get('completed_agents', [])
    if completed:
        print(f"\n{Colors.BOLD}✅ Completed Agents:{Colors.ENDC}")
        for agent_name in completed:
            print(f"   {Colors.GREEN}✓ {agent_name.upper()}{Colors.ENDC}")
    
    # Latest metrics (after evaluation)
    metrics = info.get('metrics', {})
    if metrics:
        print(f"\n{Colors.BOLD}📈 Latest Test Results:{Colors.ENDC}")
        if 'sharpe' in metrics:
            color = Colors.GREEN if metrics['sharpe'] > 1.0 else Colors.YELLOW
            print(f"   Sharpe Ratio: {color}{metrics['sharpe']:.4f}{Colors.ENDC}")
        if 'return' in metrics:
            color = Colors.GREEN if metrics['return'] > 15 else Colors.YELLOW
            print(f"   Total Return: {color}{metrics['return']:.2f}%{Colors.ENDC}")
        if 'drawdown' in metrics:
            color = Colors.GREEN if metrics['drawdown'] < 15 else Colors.RED
            print(f"   Max Drawdown: {color}{metrics['drawdown']:.2f}%{Colors.ENDC}")
        if 'volatility' in metrics:
            print(f"   Volatility: {metrics['volatility']:.2f}%")
    
    # Instructions
    print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
    print(f"{Colors.YELLOW}Press Ctrl+C to exit monitor (training continues in background){Colors.ENDC}")
    print(f"{Colors.CYAN}Log file: logs/training_progress.log{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(description='Monitor DRL training progress')
    parser.add_argument('--log', type=str, default='logs/training_progress.log',
                       help='Path to training log file')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='Update interval in seconds')
    args = parser.parse_args()
    
    # Get log file modification time as start time estimate
    log_path = args.log
    if os.path.exists(log_path):
        start_time = os.path.getmtime(log_path)
    else:
        start_time = time.time()
    
    print(f"Monitoring training progress...")
    print(f"Log file: {log_path}")
    print(f"Update interval: {args.interval}s")
    print("\nStarting monitor in 2 seconds...")
    time.sleep(2)
    
    try:
        while True:
            info = parse_log_file(log_path)
            display_progress(info, start_time)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Monitor stopped. Training may still be running.{Colors.ENDC}")
        print(f"To check: {Colors.CYAN}ps aux | grep train_and_evaluate{Colors.ENDC}")
        print(f"To view logs: {Colors.CYAN}tail -f {log_path}{Colors.ENDC}")


if __name__ == '__main__':
    main()
