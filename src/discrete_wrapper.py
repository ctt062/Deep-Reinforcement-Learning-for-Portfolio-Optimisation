"""
Discrete action space wrapper for portfolio optimization.

Converts continuous portfolio weights to discrete actions for DQN.
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Any, Dict


class DiscretePortfolioWrapper(gym.Wrapper):
    """
    Wrapper to convert continuous portfolio action space to discrete.
    
    Discretization Strategy:
    - Pre-define a set of portfolio allocation patterns
    - Each discrete action maps to a specific weight configuration
    - Includes common strategies: equal weight, concentrated, defensive, etc.
    """
    
    def __init__(
        self,
        env: gym.Env,
        n_discrete_actions: int = 50,
        strategy: str = "grid"
    ):
        """
        Initialize discrete wrapper.
        
        Args:
            env: Continuous action portfolio environment.
            n_discrete_actions: Number of discrete actions to create.
            strategy: Discretization strategy ('grid', 'random', or 'mixed').
        """
        super().__init__(env)
        
        self.n_assets = env.action_space.shape[0]
        self.n_discrete_actions = n_discrete_actions
        self.strategy = strategy
        
        # Create discrete action space
        self.action_space = gym.spaces.Discrete(n_discrete_actions)
        
        # Generate portfolio weight templates
        self.weight_templates = self._generate_weight_templates()
        
        print(f"Created discrete action space with {n_discrete_actions} actions")
        print(f"Strategy: {strategy}")
        print(f"Sample actions (first 5):")
        for i in range(min(5, n_discrete_actions)):
            print(f"  Action {i}: {self.weight_templates[i]}")
    
    def _generate_weight_templates(self) -> np.ndarray:
        """
        Generate portfolio weight templates for discrete actions.
        
        Returns:
            Array of shape (n_discrete_actions, n_assets) with weight templates.
        """
        templates = []
        
        if self.strategy == "grid":
            templates = self._generate_grid_templates()
        elif self.strategy == "random":
            templates = self._generate_random_templates()
        elif self.strategy == "mixed":
            templates = self._generate_mixed_templates()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Ensure all weights sum to 1 and are non-negative
        templates = np.array(templates)
        templates = np.abs(templates)
        templates = templates / templates.sum(axis=1, keepdims=True)
        
        return templates
    
    def _generate_grid_templates(self) -> list:
        """Generate templates using grid-based discretization."""
        templates = []
        
        # Strategy 1: Equal weight
        equal_weight = np.ones(self.n_assets) / self.n_assets
        templates.append(equal_weight)
        
        # Strategy 2: All cash (first asset assumed to be cash/safe asset)
        if self.n_assets > 1:
            all_cash = np.zeros(self.n_assets)
            all_cash[0] = 1.0
            templates.append(all_cash)
        
        # Strategy 3: Concentrated positions (each asset individually)
        for i in range(min(self.n_assets, 10)):
            concentrated = np.zeros(self.n_assets)
            concentrated[i] = 1.0
            templates.append(concentrated)
        
        # Strategy 4: Two-asset combinations (grid)
        if self.n_assets >= 2:
            for i in range(min(self.n_assets, 5)):
                for j in range(i + 1, min(self.n_assets, 5)):
                    for weight in [0.3, 0.5, 0.7]:
                        two_asset = np.zeros(self.n_assets)
                        two_asset[i] = weight
                        two_asset[j] = 1.0 - weight
                        if len(templates) < self.n_discrete_actions - 5:
                            templates.append(two_asset)
        
        # Strategy 5: Random valid portfolios to fill remaining slots
        while len(templates) < self.n_discrete_actions:
            weights = np.random.dirichlet(np.ones(self.n_assets))
            templates.append(weights)
        
        return templates[:self.n_discrete_actions]
    
    def _generate_random_templates(self) -> list:
        """Generate templates using random sampling."""
        templates = []
        
        # Add equal weight first
        equal_weight = np.ones(self.n_assets) / self.n_assets
        templates.append(equal_weight)
        
        # Generate random portfolios using Dirichlet distribution
        for _ in range(self.n_discrete_actions - 1):
            # Use Dirichlet with varying concentration
            alpha = np.random.uniform(0.5, 2.0, self.n_assets)
            weights = np.random.dirichlet(alpha)
            templates.append(weights)
        
        return templates
    
    def _generate_mixed_templates(self) -> list:
        """Generate templates using mixed strategies."""
        templates = []
        
        # 1. Equal weight
        equal_weight = np.ones(self.n_assets) / self.n_assets
        templates.append(equal_weight)
        
        # 2. Single asset portfolios
        for i in range(self.n_assets):
            single = np.zeros(self.n_assets)
            single[i] = 1.0
            templates.append(single)
        
        # 3. Pairs with different weightings
        n_pairs = min(10, (self.n_discrete_actions - len(templates)) // 2)
        for _ in range(n_pairs):
            i, j = np.random.choice(self.n_assets, 2, replace=False)
            weight = np.random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
            pair = np.zeros(self.n_assets)
            pair[i] = weight
            pair[j] = 1.0 - weight
            templates.append(pair)
        
        # 4. Fill rest with random portfolios
        while len(templates) < self.n_discrete_actions:
            # Random concentration parameter
            alpha = np.random.uniform(0.3, 3.0)
            weights = np.random.dirichlet(np.ones(self.n_assets) * alpha)
            templates.append(weights)
        
        return templates[:self.n_discrete_actions]
    
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Take a step using discrete action.
        
        Args:
            action: Discrete action index.
            
        Returns:
            Observation, reward, terminated, truncated, info.
        """
        # Convert discrete action to continuous weights
        continuous_action = self.weight_templates[action]
        
        # Take step in underlying environment
        return self.env.step(continuous_action)
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset environment."""
        return self.env.reset(**kwargs)
    
    def __getattr__(self, name):
        """Forward attribute access to wrapped environment."""
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)


class AdaptiveDiscreteWrapper(gym.Wrapper):
    """
    Adaptive discrete wrapper that learns good portfolio templates over time.
    
    This is a more sophisticated version that can adapt the action space
    based on observed performance.
    """
    
    def __init__(
        self,
        env: gym.Env,
        n_discrete_actions: int = 50,
        adaptation_freq: int = 10000,
        keep_top_k: int = 10
    ):
        """
        Initialize adaptive discrete wrapper.
        
        Args:
            env: Continuous action portfolio environment.
            n_discrete_actions: Number of discrete actions.
            adaptation_freq: How often to adapt templates (in steps).
            keep_top_k: Number of best performing templates to keep.
        """
        super().__init__(env)
        
        self.n_assets = env.action_space.shape[0]
        self.n_discrete_actions = n_discrete_actions
        self.adaptation_freq = adaptation_freq
        self.keep_top_k = keep_top_k
        
        # Create discrete action space
        self.action_space = gym.spaces.Discrete(n_discrete_actions)
        
        # Initialize templates
        self.weight_templates = self._initialize_templates()
        
        # Track performance
        self.action_counts = np.zeros(n_discrete_actions)
        self.action_rewards = np.zeros(n_discrete_actions)
        self.step_count = 0
        
        print(f"Created adaptive discrete wrapper with {n_discrete_actions} actions")
    
    def _initialize_templates(self) -> np.ndarray:
        """Initialize weight templates."""
        templates = []
        
        # Equal weight
        equal_weight = np.ones(self.n_assets) / self.n_assets
        templates.append(equal_weight)
        
        # Single assets
        for i in range(self.n_assets):
            single = np.zeros(self.n_assets)
            single[i] = 1.0
            templates.append(single)
        
        # Random portfolios
        while len(templates) < self.n_discrete_actions:
            weights = np.random.dirichlet(np.ones(self.n_assets))
            templates.append(weights)
        
        return np.array(templates)
    
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        """Take a step and track performance."""
        continuous_action = self.weight_templates[action]
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        
        # Track performance
        self.action_counts[action] += 1
        self.action_rewards[action] += reward
        self.step_count += 1
        
        # Adapt templates periodically
        if self.step_count % self.adaptation_freq == 0:
            self._adapt_templates()
        
        return obs, reward, terminated, truncated, info
    
    def _adapt_templates(self):
        """Adapt weight templates based on performance."""
        # Calculate average reward per action
        avg_rewards = np.zeros(self.n_discrete_actions)
        for i in range(self.n_discrete_actions):
            if self.action_counts[i] > 0:
                avg_rewards[i] = self.action_rewards[i] / self.action_counts[i]
            else:
                avg_rewards[i] = -np.inf
        
        # Keep top performing templates
        top_k_indices = np.argsort(avg_rewards)[-self.keep_top_k:]
        top_templates = self.weight_templates[top_k_indices]
        
        # Generate new templates (variations of top performers)
        new_templates = [top_templates[i] for i in range(len(top_templates))]
        
        while len(new_templates) < self.n_discrete_actions:
            # Pick random top template and add noise
            base = top_templates[np.random.randint(len(top_templates))]
            noise = np.random.normal(0, 0.1, self.n_assets)
            new_template = base + noise
            new_template = np.abs(new_template)
            new_template = new_template / new_template.sum()
            new_templates.append(new_template)
        
        # Update templates
        self.weight_templates = np.array(new_templates[:self.n_discrete_actions])
        
        # Reset tracking
        self.action_counts = np.zeros(self.n_discrete_actions)
        self.action_rewards = np.zeros(self.n_discrete_actions)
        
        print(f"Adapted templates at step {self.step_count}")
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset environment."""
        return self.env.reset(**kwargs)
    
    def __getattr__(self, name):
        """Forward attribute access to wrapped environment."""
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)


if __name__ == "__main__":
    """Example usage."""
    print("This module provides discrete action wrappers for DQN.")
    print("\nWrapper types:")
    print("  - DiscretePortfolioWrapper: Fixed discrete actions")
    print("  - AdaptiveDiscreteWrapper: Adaptive actions based on performance")
