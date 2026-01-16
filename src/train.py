"""
Training Script for Policy Conditioning Study

Phase 2: Train PPO agents across all 4 conditions with:
- 5 random seeds per condition
- λ hyperparameter sweep {0.1, 0.5, 1.0}
- TensorBoard logging
- Model checkpointing

Usage:
    python train.py                           # Train all agents, all seeds, λ=0.5
    python train.py --agent conditioned       # Train single agent type
    python train.py --lambda-sweep            # Full hyperparameter sweep
    python train.py --total-timesteps 500000  # Shorter training run
"""

import os
import argparse
import torch
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from custom_lander import make_env, VelocityTrackingLander


# =============================================================================
# Configuration (Table 3 from document)
# =============================================================================

AGENT_TYPES = ['baseline', 'noise', 'shuffled', 'conditioned']
SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds as specified
LAMBDA_VALUES = [0.1, 0.5, 1.0]   # Hyperparameter sweep

# PPO Hyperparameters (Table 3)
PPO_CONFIG = {
    'learning_rate': 3e-4,          # Linear decay applied separately
    'n_steps': 2048,                # Steps per rollout
    'batch_size': 64,               # Minibatch size
    'n_epochs': 10,                 # Epochs per update
    'gamma': 0.99,                  # Discount factor
    'gae_lambda': 0.95,             # GAE lambda
    'clip_range': 0.2,              # PPO clip range
    'ent_coef': 0.0,                # No entropy bonus (per document)
    'vf_coef': 0.5,                 # Value function coefficient
    'max_grad_norm': 0.5,           # Gradient clipping
    'policy_kwargs': dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])  # MlpPolicy [64, 64]
    ),
}

TOTAL_TIMESTEPS = 1_000_000  # 1M steps per seed
EVAL_FREQ = 10_000           # Evaluate every 10k steps
N_EVAL_EPISODES = 10         # Episodes per evaluation
CHECKPOINT_FREQ = 50_000     # Save model every 50k steps


# =============================================================================
# Custom Callbacks
# =============================================================================

class VelocityTrackingCallback(BaseCallback):
    """
    Custom callback to log velocity tracking metrics to TensorBoard.
    Logs: target_vx, current_vx, velocity_error, target_changes
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_errors = []
        self.episode_target_changes = 0
        
    def _on_step(self) -> bool:
        # Get info from vectorized env
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'velocity_error' in info:
                self.episode_errors.append(info['velocity_error'])
            if info.get('target_changed', False):
                self.episode_target_changes += 1
                
            # Log at episode end
            if 'episode' in info:
                if self.episode_errors:
                    mean_error = np.mean(self.episode_errors)
                    self.logger.record('velocity/mean_error', mean_error)
                    self.logger.record('velocity/target_changes', self.episode_target_changes)
                    
                self.episode_errors = []
                self.episode_target_changes = 0
                
        return True


class LinearSchedule:
    """Linear learning rate schedule."""
    
    def __init__(self, initial_value: float):
        self.initial_value = initial_value
        
    def __call__(self, progress_remaining: float) -> float:
        return progress_remaining * self.initial_value


# =============================================================================
# Environment Factory
# =============================================================================

def make_training_env(
    agent_type: str,
    lambda_penalty: float,
    seed: int,
    rank: int = 0,
    log_dir: str = None
) -> Callable:
    """
    Factory function for creating training environments.
    Returns a callable that creates the environment.
    """
    def _init():
        env = make_env(
            agent_type=agent_type,
            change_interval=150,
            lambda_penalty=lambda_penalty,
        )
        env.reset(seed=seed + rank)
        
        if log_dir:
            env = Monitor(env, os.path.join(log_dir, f'rank_{rank}'))
            
        return env
    
    return _init


# =============================================================================
# Training Function
# =============================================================================

def train_agent(
    agent_type: str,
    lambda_penalty: float,
    seed: int,
    total_timesteps: int,
    output_dir: Path,
    n_envs: int = 4,
    verbose: int = 1
) -> dict:
    """
    Train a single agent configuration.
    
    Args:
        agent_type: One of 'baseline', 'noise', 'shuffled', 'conditioned'
        lambda_penalty: Reward penalty coefficient
        seed: Random seed
        total_timesteps: Total training steps
        output_dir: Base output directory
        n_envs: Number of parallel environments
        verbose: Verbosity level
        
    Returns:
        dict with training results
    """
    # Create experiment name
    exp_name = f"{agent_type}_lambda{lambda_penalty}_seed{seed}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = exp_dir / "logs"
    model_dir = exp_dir / "models"
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training: {exp_name}")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}")
    
    # Create vectorized training environment
    env_fns = [
        make_training_env(agent_type, lambda_penalty, seed, rank=i, log_dir=str(log_dir))
        for i in range(n_envs)
    ]
    train_env = DummyVecEnv(env_fns)
    
    # Create evaluation environment
    # NOTE: This eval env is on-distribution (same stochastic target process).
    # Generalization is evaluated separately in Phase 3 (step-response test).
    eval_env = DummyVecEnv([
        make_training_env(agent_type, lambda_penalty, seed, rank=100)
    ])
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ // n_envs,
        save_path=str(model_dir),
        name_prefix=exp_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=EVAL_FREQ // n_envs,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    
    velocity_callback = VelocityTrackingCallback()
    
    callbacks = [checkpoint_callback, eval_callback, velocity_callback]
    
    # Create PPO model with linear LR schedule
    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # Explicitly use GPU
        learning_rate=LinearSchedule(PPO_CONFIG['learning_rate']),
        n_steps=PPO_CONFIG['n_steps'],
        batch_size=PPO_CONFIG['batch_size'],
        n_epochs=PPO_CONFIG['n_epochs'],
        gamma=PPO_CONFIG['gamma'],
        gae_lambda=PPO_CONFIG['gae_lambda'],
        clip_range=PPO_CONFIG['clip_range'],
        ent_coef=PPO_CONFIG['ent_coef'],
        vf_coef=PPO_CONFIG['vf_coef'],
        max_grad_norm=PPO_CONFIG['max_grad_norm'],
        policy_kwargs=PPO_CONFIG['policy_kwargs'],
        tensorboard_log=str(log_dir / "tensorboard"),
        seed=seed,
        verbose=verbose,
    )
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=exp_name,
    )
    
    # Save final model
    final_model_path = model_dir / f"{exp_name}_final"
    model.save(str(final_model_path))
    print(f"✅ Saved final model: {final_model_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return {
        'agent_type': agent_type,
        'lambda_penalty': lambda_penalty,
        'seed': seed,
        'model_path': str(final_model_path),
        'log_dir': str(log_dir),
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def run_full_experiment(
    agent_types: list = None,
    seeds: list = None,
    lambda_values: list = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    output_dir: str = None,
    n_envs: int = 4,
):
    """
    Run the full training experiment.
    
    Args:
        agent_types: List of agent types to train
        seeds: List of random seeds
        lambda_values: List of lambda penalty values
        total_timesteps: Training timesteps per configuration
        output_dir: Output directory
        n_envs: Number of parallel environments
    """
    agent_types = agent_types or AGENT_TYPES
    seeds = seeds or SEEDS
    lambda_values = lambda_values or [0.5]  # Default to single lambda
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"experiments/run_{timestamp}")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    config = {
        'agent_types': agent_types,
        'seeds': seeds,
        'lambda_values': lambda_values,
        'total_timesteps': total_timesteps,
        'ppo_config': PPO_CONFIG,
    }
    
    config_path = output_dir / "config.txt"
    with open(config_path, 'w') as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
    
    print(f"\n🔬 POLICY CONDITIONING STUDY - PHASE 2 TRAINING")
    print(f"{'='*60}")
    print(f"Agent Types: {agent_types}")
    print(f"Seeds: {seeds}")
    print(f"Lambda Values: {lambda_values}")
    print(f"Timesteps per config: {total_timesteps:,}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    total_configs = len(agent_types) * len(seeds) * len(lambda_values)
    print(f"Total configurations to train: {total_configs}")
    
    results = []
    config_idx = 0
    
    for lambda_penalty in lambda_values:
        for agent_type in agent_types:
            for seed in seeds:
                config_idx += 1
                print(f"\n[{config_idx}/{total_configs}] Training...")
                
                result = train_agent(
                    agent_type=agent_type,
                    lambda_penalty=lambda_penalty,
                    seed=seed,
                    total_timesteps=total_timesteps,
                    output_dir=output_dir,
                    n_envs=n_envs,
                )
                results.append(result)
    
    # Save results summary
    results_path = output_dir / "training_results.txt"
    with open(results_path, 'w') as f:
        for r in results:
            f.write(f"{r}\n")
    
    print(f"\n{'='*60}")
    print(f"🎉 TRAINING COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agents for Policy Conditioning Study"
    )
    
    parser.add_argument(
        '--agent', '-a',
        type=str,
        choices=AGENT_TYPES,
        default=None,
        help='Train only this agent type (default: all)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Train only this seed (default: all 5 seeds)'
    )
    
    parser.add_argument(
        '--lambda-penalty', '-l',
        type=float,
        default=0.5,
        help='Lambda penalty coefficient (default: 0.5)'
    )
    
    parser.add_argument(
        '--lambda-sweep',
        action='store_true',
        help='Run full lambda sweep {0.1, 0.5, 1.0}'
    )
    
    parser.add_argument(
        '--total-timesteps', '-t',
        type=int,
        default=TOTAL_TIMESTEPS,
        help=f'Total training timesteps (default: {TOTAL_TIMESTEPS:,})'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: experiments/run_TIMESTAMP)'
    )
    
    parser.add_argument(
        '--n-envs',
        type=int,
        default=4,
        help='Number of parallel environments (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Determine what to train
    agent_types = [args.agent] if args.agent else AGENT_TYPES
    seeds = [args.seed] if args.seed else SEEDS
    lambda_values = LAMBDA_VALUES if args.lambda_sweep else [args.lambda_penalty]
    
    run_full_experiment(
        agent_types=agent_types,
        seeds=seeds,
        lambda_values=lambda_values,
        total_timesteps=args.total_timesteps,
        output_dir=args.output_dir,
        n_envs=args.n_envs,
    )


if __name__ == "__main__":
    main()
