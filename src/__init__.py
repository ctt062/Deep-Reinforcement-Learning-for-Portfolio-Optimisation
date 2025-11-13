"""Package initialization for DRL Portfolio Optimization."""

__version__ = "1.0.0"
__author__ = "IEDA4000F Project"
__license__ = "MIT"

# Import main components for easier access
from src.data_loader import DataLoader
from src.portfolio_env import PortfolioEnv
from src.agents import create_agent
from src.benchmarks import (
    EqualWeightStrategy,
    MeanVarianceStrategy,
    MomentumStrategy,
)
from src.metrics import PerformanceMetrics

__all__ = [
    "DataLoader",
    "PortfolioEnv",
    "create_agent",
    "EqualWeightStrategy",
    "MeanVarianceStrategy",
    "MomentumStrategy",
    "PerformanceMetrics",
]
