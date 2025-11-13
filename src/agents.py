"""
Deep Reinforcement Learning agents for portfolio optimization.

Implements:
1. DQN (Deep Q-Network) - Discrete action space
2. PPO (Proximal Policy Optimization) - Continuous action space
3. DDPG (Deep Deterministic Policy Gradient) - Continuous action space

Uses stable-baselines3 for implementations.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import warnings

from stable_baselines3 import PPO, DDPG, DQN
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym


class PortfolioFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for portfolio state.
    
    Processes the state with:
    1. Fully connected layers
    2. ReLU activations
    3. Layer normalization (optional)
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        net_arch: List[int] = [128, 128],
        activation_fn: nn.Module = nn.ReLU,
        normalize: bool = True
    ):
        """
        Initialize feature extractor.
        
        Args:
            observation_space: Observation space of the environment.
            features_dim: Dimension of the output features.
            net_arch: List of hidden layer sizes.
            activation_fn: Activation function class.
            normalize: Whether to use layer normalization.
        """
        super().__init__(observation_space, features_dim)
        
        # Build network architecture
        layers = []
        input_dim = observation_space.shape[0]
        
        for hidden_dim in net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if normalize:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(activation_fn())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, features_dim))
        
        if normalize:
            layers.append(nn.LayerNorm(features_dim))
        
        layers.append(activation_fn())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            observations: Input observations.
            
        Returns:
            Extracted features.
        """
        return self.network(observations)


class TrainingCallback(BaseCallback):
    """
    Custom callback for tracking training progress.
    """
    
    def __init__(
        self,
        verbose: int = 0,
        log_freq: int = 1000,
        save_freq: int = 10000,
        save_path: Optional[str] = None,
    ):
        """
        Initialize callback.
        
        Args:
            verbose: Verbosity level.
            log_freq: Frequency of logging (in steps).
            save_freq: Frequency of model saving (in steps).
            save_path: Path to save models.
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.save_path = save_path
        
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """
        Called at each step.
        
        Returns:
            Whether to continue training.
        """
        # Log progress
        if self.n_calls % self.log_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                
                if self.verbose > 0:
                    print(f"Steps: {self.n_calls} | "
                          f"Episodes: {len(self.episode_rewards)} | "
                          f"Mean Reward: {mean_reward:.4f} | "
                          f"Mean Length: {mean_length:.1f}")
        
        # Save model
        if self.save_path is not None and self.n_calls % self.save_freq == 0:
            model_path = f"{self.save_path}_step_{self.n_calls}.zip"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved to {model_path}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        # Track episode statistics if available
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])


def create_dqn_agent(
    env: gym.Env,
    learning_rate: float = 0.0001,
    buffer_size: int = 100000,
    learning_starts: int = 1000,
    batch_size: int = 32,
    gamma: float = 0.99,
    tau: float = 1.0,
    target_update_interval: int = 1000,
    exploration_fraction: float = 0.1,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    net_arch: List[int] = [128, 128],
    verbose: int = 1,
    device: str = "auto",
    **kwargs
) -> DQN:
    """
    Create a DQN agent for portfolio optimization.
    
    Note: DQN requires discrete action space. For continuous portfolio
    weights, consider using PPO or DDPG instead.
    
    Args:
        env: Portfolio environment.
        learning_rate: Learning rate.
        buffer_size: Replay buffer size.
        learning_starts: Start learning after this many steps.
        batch_size: Batch size for training.
        gamma: Discount factor.
        tau: Soft update coefficient.
        target_update_interval: Update target network frequency.
        exploration_fraction: Fraction of training for exploration.
        exploration_initial_eps: Initial epsilon for exploration.
        exploration_final_eps: Final epsilon for exploration.
        net_arch: Network architecture.
        verbose: Verbosity level.
        device: Device to use ('cpu', 'cuda', or 'auto').
        
    Returns:
        DQN agent.
    """
    # DQN requires discrete action space
    # Note: For continuous weights, this might need discretization
    warnings.warn(
        "DQN is designed for discrete action spaces. "
        "For continuous portfolio weights, consider using PPO or DDPG."
    )
    
    policy_kwargs = {
        "net_arch": net_arch,
    }
    
    agent = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        device=device,
        **kwargs
    )
    
    return agent


def create_ppo_agent(
    env: gym.Env,
    learning_rate: float = 0.0003,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    net_arch: List[int] = [128, 128],
    verbose: int = 1,
    device: str = "auto",
    **kwargs
) -> PPO:
    """
    Create a PPO agent for portfolio optimization.
    
    PPO is well-suited for continuous action spaces (portfolio weights).
    
    Mathematical Formulation:
        L^CLIP(θ) = E[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]
        where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    
    Args:
        env: Portfolio environment.
        learning_rate: Learning rate.
        n_steps: Number of steps per update.
        batch_size: Batch size for training.
        n_epochs: Number of epochs per update.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_range: Clipping parameter for PPO.
        ent_coef: Entropy coefficient for exploration.
        vf_coef: Value function coefficient.
        max_grad_norm: Maximum gradient norm.
        net_arch: Network architecture [shared, policy, value].
        verbose: Verbosity level.
        device: Device to use.
        
    Returns:
        PPO agent.
    """
    policy_kwargs = {
        "net_arch": dict(
            pi=net_arch,  # Policy network
            vf=net_arch,  # Value network
        ),
        "activation_fn": nn.ReLU,
    }
    
    agent = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        device=device,
        **kwargs
    )
    
    return agent


def create_ddpg_agent(
    env: gym.Env,
    learning_rate: float = 0.001,
    buffer_size: int = 1000000,
    learning_starts: int = 100,
    batch_size: int = 100,
    tau: float = 0.005,
    gamma: float = 0.99,
    action_noise_std: float = 0.1,
    net_arch: Dict[str, List[int]] = None,
    verbose: int = 1,
    device: str = "auto",
    **kwargs
) -> DDPG:
    """
    Create a DDPG agent for portfolio optimization.
    
    DDPG is an actor-critic algorithm for continuous action spaces.
    
    Mathematical Formulation:
        Actor: ∇θ J ≈ E[∇θ μ_θ(s) ∇_a Q_φ(s,a)|_{a=μ_θ(s)}]
        Critic: L(φ) = E[(Q_φ(s,a) - y)^2]
        where y = r + γ Q_φ'(s', μ_θ'(s'))
    
    Args:
        env: Portfolio environment.
        learning_rate: Learning rate.
        buffer_size: Replay buffer size.
        learning_starts: Start learning after this many steps.
        batch_size: Batch size for training.
        tau: Soft update coefficient for target networks.
        gamma: Discount factor.
        action_noise_std: Standard deviation of action noise.
        net_arch: Network architecture for actor and critic.
        verbose: Verbosity level.
        device: Device to use.
        
    Returns:
        DDPG agent.
    """
    if net_arch is None:
        net_arch = dict(pi=[128, 128], qf=[128, 128])
    
    policy_kwargs = {
        "net_arch": net_arch,
        "activation_fn": nn.ReLU,
    }
    
    # Create action noise
    from stable_baselines3.common.noise import NormalActionNoise
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=action_noise_std * np.ones(n_actions)
    )
    
    agent = DDPG(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        device=device,
        **kwargs
    )
    
    return agent


def create_agent(
    agent_type: str,
    env: gym.Env,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    Factory function to create DRL agents.
    
    Args:
        agent_type: Type of agent ('dqn', 'ppo', or 'ddpg').
        env: Portfolio environment.
        config: Configuration dictionary with hyperparameters.
        **kwargs: Additional arguments passed to agent constructor.
        
    Returns:
        Initialized DRL agent.
    """
    if config is None:
        config = {}
    
    # Merge config with kwargs
    params = {**config, **kwargs}
    
    agent_type = agent_type.lower()
    
    if agent_type == "dqn":
        return create_dqn_agent(env, **params)
    elif agent_type == "ppo":
        return create_ppo_agent(env, **params)
    elif agent_type == "ddpg":
        return create_ddpg_agent(env, **params)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. "
                        f"Choose from: 'dqn', 'ppo', 'ddpg'")


def train_agent(
    agent: Any,
    total_timesteps: int,
    callback: Optional[BaseCallback] = None,
    log_interval: int = 10,
    eval_env: Optional[gym.Env] = None,
    eval_freq: int = 5000,
    n_eval_episodes: int = 5,
    save_path: Optional[str] = None,
) -> Any:
    """
    Train a DRL agent.
    
    Args:
        agent: DRL agent to train.
        total_timesteps: Total number of training steps.
        callback: Custom callback for training.
        log_interval: Logging frequency.
        eval_env: Environment for evaluation.
        eval_freq: Evaluation frequency.
        n_eval_episodes: Number of episodes per evaluation.
        save_path: Path to save best model.
        
    Returns:
        Trained agent.
    """
    callbacks = []
    
    # Add custom callback if provided
    if callback is not None:
        callbacks.append(callback)
    
    # Add evaluation callback if eval_env provided
    if eval_env is not None and save_path is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)
    
    # Train agent
    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks if callbacks else None,
        log_interval=log_interval,
    )
    
    return agent


def evaluate_agent(
    agent: Any,
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a trained agent.
    
    Args:
        agent: Trained DRL agent.
        env: Environment for evaluation.
        n_episodes: Number of evaluation episodes.
        deterministic: Whether to use deterministic actions.
        render: Whether to render episodes.
        
    Returns:
        Dictionary with evaluation results.
    """
    episode_rewards = []
    episode_lengths = []
    all_returns = []
    all_values = []
    all_weights = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Get episode history
        history = env.get_portfolio_history()
        all_returns.append(history['returns'])
        all_values.append(history['values'])
        all_weights.append(history['weights'])
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'returns': all_returns,
        'values': all_values,
        'weights': all_weights,
    }
    
    return results


if __name__ == "__main__":
    """Example usage of DRL agents."""
    
    print("This module provides DRL agent creation and training utilities.")
    print("Use the create_agent() function to instantiate agents.")
    print("\nSupported agents:")
    print("  - DQN: Deep Q-Network (discrete actions)")
    print("  - PPO: Proximal Policy Optimization (continuous actions)")
    print("  - DDPG: Deep Deterministic Policy Gradient (continuous actions)")
    print("\nFor portfolio optimization, PPO and DDPG are recommended.")
