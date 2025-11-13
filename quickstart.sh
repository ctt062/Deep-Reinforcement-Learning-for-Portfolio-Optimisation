#!/bin/bash

# Quick Start Script for DRL Portfolio Optimization
# This script sets up the environment and runs a quick demo

echo "================================================"
echo "DRL Portfolio Optimization - Quick Start"
echo "================================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

echo "✓ Python 3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data models results logs
echo "✓ Directories created"

# Run a quick test
echo ""
echo "================================================"
echo "Running Quick Test"
echo "================================================"
echo ""

# Test data loading
echo "Testing data loader..."
python3 -c "
import sys
sys.path.insert(0, 'src')
from data_loader import DataLoader
print('✓ Data loader works!')
"

# Test environment
echo "Testing environment..."
python3 -c "
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'src')
from portfolio_env import PortfolioEnv
# Create dummy data
dates = pd.date_range('2020-01-01', periods=100)
prices = pd.DataFrame(np.random.randn(100, 3).cumsum(axis=0) + 100, 
                      index=dates, columns=['A', 'B', 'C'])
returns = prices.pct_change().fillna(0)
features = returns.copy()
# Create environment
env = PortfolioEnv(prices, returns, features)
print('✓ Environment works!')
"

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Train an agent:"
echo "   python scripts/train.py --agent ppo --timesteps 50000"
echo ""
echo "2. Evaluate performance:"
echo "   python scripts/evaluate.py --compare-all --save-results"
echo ""
echo "3. Explore the demo notebook:"
echo "   jupyter notebook notebooks/demo.ipynb"
echo ""
echo "For more information, see README.md"
echo ""
