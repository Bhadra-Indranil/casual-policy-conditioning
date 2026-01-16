"""
Evaluation Script for Policy Conditioning Study

Phase 3: Evaluate trained agents using:
- Settling Time (control theory metric)
- CVaR (10%) for tail risk / robustness
- Step-function schedule for stress testing

Usage:
    python evaluate.py --model-dir experiments/run_XXX
    python evaluate.py --model-path path/to/model.zip --agent conditioned
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from custom_lander import make_env


# =============================================================================
# Metrics Configuration
# =============================================================================

SETTLING_TOLERANCE = 0.1     # 10% of step magnitude or 0.1 absolute
SETTLING_HOLD_STEPS = 50     # Must stay within band for this many steps
CVAR_ALPHA = 0.1             # CVaR at 10th percentile (worst 10%)
N_EVAL_EPISODES = 100        # Episodes for statistical evaluation
STEP_TEST_EPISODES = 50      # Episodes for step-function test


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SettlingTimeResult:
    """Results from settling time analysis."""
    mean_settling_time: float
    std_settling_time: float
    median_settling_time: float
    settling_times: List[float]
    n_successful_settles: int
    n_total_transitions: int
    success_rate: float


@dataclass
class CVaRResult:
    """Results from CVaR analysis."""
    mean_reward: float
    std_reward: float
    var_alpha: float       # Value at Risk (α quantile)
    cvar_alpha: float      # Conditional VaR (expected shortfall)
    alpha: float           # The α percentile used
    rewards: List[float]


@dataclass
class EvaluationResults:
    """Complete evaluation results for an agent."""
    agent_type: str
    lambda_penalty: float
    seed: int
    model_path: str
    settling_time: SettlingTimeResult
    cvar: CVaRResult
    step_response: Dict


# =============================================================================
# Settling Time Calculation
# =============================================================================

def calculate_settling_time(
    velocity_trace: np.ndarray,
    target_trace: np.ndarray,
    change_indices: List[int],
    tolerance: float = SETTLING_TOLERANCE,
    hold_steps: int = SETTLING_HOLD_STEPS,
) -> List[float]:
    """
    Calculate settling time after each target change.
    
    Settling Time: Time to reach and stay within tolerance band of new target.
    
    Args:
        velocity_trace: Array of current velocities
        target_trace: Array of target velocities
        change_indices: Indices where target changed
        tolerance: Error tolerance (fraction of step or absolute)
        hold_steps: Steps to maintain within band
        
    Returns:
        List of settling times (NaN if never settled)
    """
    settling_times = []
    
    for i, change_idx in enumerate(change_indices):
        if change_idx >= len(velocity_trace) - hold_steps:
            continue
            
        # Get the new target after change
        new_target = target_trace[change_idx]
        
        # Calculate step magnitude for relative tolerance
        if i > 0:
            old_target = target_trace[change_indices[i-1]]
            step_magnitude = abs(new_target - old_target)
        else:
            step_magnitude = abs(new_target)
            
        # Use stricter of: 10% of step or absolute tolerance
        tol = min(tolerance * step_magnitude, tolerance) if step_magnitude > 0 else tolerance
        
        # Find settling time
        settled = False
        settling_time = np.nan
        
        for t in range(change_idx, len(velocity_trace) - hold_steps):
            error = abs(velocity_trace[t] - new_target)
            
            if error <= tol:
                # Check if it stays within band
                future_errors = np.abs(velocity_trace[t:t+hold_steps] - new_target)
                if np.all(future_errors <= tol):
                    settling_time = t - change_idx
                    settled = True
                    break
                    
        settling_times.append(settling_time)
        
    return settling_times


def run_settling_time_eval(
    model: PPO,
    agent_type: str,
    lambda_penalty: float,
    n_episodes: int = N_EVAL_EPISODES,
    change_interval: int = 150,
) -> SettlingTimeResult:
    """
    Evaluate settling time across multiple episodes.
    """
    env = make_env(
        agent_type=agent_type,
        change_interval=change_interval,
        lambda_penalty=lambda_penalty,
    )
    
    all_settling_times = []
    n_transitions = 0
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        
        velocities = [info['current_vx']]
        targets = [info['target_vx']]
        change_indices = []
        
        done = False
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            velocities.append(info['current_vx'])
            targets.append(info['target_vx'])
            
            if info.get('target_changed', False):
                change_indices.append(step)
        
        # Calculate settling times for this episode
        if change_indices:
            episode_settling = calculate_settling_time(
                np.array(velocities),
                np.array(targets),
                change_indices,
            )
            all_settling_times.extend(episode_settling)
            n_transitions += len(change_indices)
    
    env.close()
    
    # Filter out NaN (failed to settle)
    valid_times = [t for t in all_settling_times if not np.isnan(t)]
    
    return SettlingTimeResult(
        mean_settling_time=np.mean(valid_times) if valid_times else np.nan,
        std_settling_time=np.std(valid_times) if valid_times else np.nan,
        median_settling_time=np.median(valid_times) if valid_times else np.nan,
        settling_times=all_settling_times,
        n_successful_settles=len(valid_times),
        n_total_transitions=n_transitions,
        success_rate=len(valid_times) / n_transitions if n_transitions > 0 else 0,
    )


# =============================================================================
# CVaR Calculation
# =============================================================================

def calculate_cvar(rewards: np.ndarray, alpha: float = CVAR_ALPHA) -> Tuple[float, float]:
    """
    Calculate Value-at-Risk and Conditional Value-at-Risk.
    
    CVaR (Expected Shortfall): Mean of the worst α% of outcomes.
    
    Args:
        rewards: Array of episode rewards
        alpha: Percentile (0.1 = worst 10%)
        
    Returns:
        (VaR, CVaR) tuple
    """
    sorted_rewards = np.sort(rewards)
    var_idx = int(np.floor(alpha * len(sorted_rewards)))
    var_idx = max(1, var_idx)  # At least 1 sample
    
    var_alpha = sorted_rewards[var_idx - 1]  # Value at Risk
    cvar_alpha = np.mean(sorted_rewards[:var_idx])  # Conditional VaR
    
    return var_alpha, cvar_alpha


def run_cvar_eval(
    model: PPO,
    agent_type: str,
    lambda_penalty: float,
    n_episodes: int = N_EVAL_EPISODES,
    change_interval: int = 150,
) -> CVaRResult:
    """
    Evaluate CVaR across multiple episodes.
    """
    env = make_env(
        agent_type=agent_type,
        change_interval=change_interval,
        lambda_penalty=lambda_penalty,
    )
    
    episode_rewards = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
        episode_rewards.append(total_reward)
    
    env.close()
    
    rewards = np.array(episode_rewards)
    var_alpha, cvar_alpha = calculate_cvar(rewards, CVAR_ALPHA)
    
    return CVaRResult(
        mean_reward=np.mean(rewards),
        std_reward=np.std(rewards),
        var_alpha=var_alpha,
        cvar_alpha=cvar_alpha,
        alpha=CVAR_ALPHA,
        rewards=episode_rewards,
    )


# =============================================================================
# Step-Function Response Test
# =============================================================================

def run_step_response_test(
    model: PPO,
    agent_type: str,
    lambda_penalty: float,
    v_initial: float = -0.5,
    v_final: float = 0.5,
    hold_steps: int = 200,
    n_episodes: int = STEP_TEST_EPISODES,
) -> Dict:
    """
    Run step-function test: Hold at v_initial, then switch to v_final.
    
    This stress test evaluates adaptation to a large velocity reversal.
    """
    # Create custom env for step test
    import gymnasium as gym
    from custom_lander import VelocityTrackingLander
    
    base_env = gym.make("LunarLanderContinuous-v3")
    env = VelocityTrackingLander(
        base_env,
        agent_type=agent_type,
        change_interval=hold_steps,  # Switch exactly at hold_steps
        lambda_penalty=lambda_penalty,
    )
    
    all_traces = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        
        # Override the schedule for controlled test
        v_range = env.v_max - env.v_min
        c_initial = (v_initial - env.v_min) / v_range
        c_final = (v_final - env.v_min) / v_range
        
        env.target_schedule = [c_initial, c_final] + [c_final] * 10
        env.shuffled_schedule = env.target_schedule.copy()
        env.current_signal = c_initial
        env.target_vx = v_initial
        env.schedule_index = 0
        
        velocities = []
        targets = []
        
        done = False
        step = 0
        max_steps = hold_steps * 3  # Run for 3x hold period
        
        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            velocities.append(info['current_vx'])
            targets.append(info['target_vx'])
        
        all_traces.append({
            'velocities': velocities,
            'targets': targets,
            'steps': len(velocities),
        })
    
    env.close()
    
    # Aggregate traces
    min_len = min(len(t['velocities']) for t in all_traces)
    velocity_matrix = np.array([t['velocities'][:min_len] for t in all_traces])
    target_matrix = np.array([t['targets'][:min_len] for t in all_traces])
    
    mean_velocity = np.mean(velocity_matrix, axis=0)
    std_velocity = np.std(velocity_matrix, axis=0)
    mean_target = np.mean(target_matrix, axis=0)
    
    return {
        'v_initial': v_initial,
        'v_final': v_final,
        'hold_steps': hold_steps,
        'n_episodes': n_episodes,
        'mean_velocity': mean_velocity.tolist(),
        'std_velocity': std_velocity.tolist(),
        'mean_target': mean_target.tolist(),
        'all_traces': all_traces,
    }


# =============================================================================
# Full Evaluation Pipeline
# =============================================================================

def evaluate_model(
    model_path: str,
    agent_type: str,
    lambda_penalty: float = 0.5,
    seed: int = 0,
) -> EvaluationResults:
    """
    Run full evaluation on a trained model.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {agent_type} (λ={lambda_penalty}, seed={seed})")
    print(f"Model: {model_path}")
    print(f"{'='*60}")
    
    # Load model (explicitly use GPU)
    model = PPO.load(model_path, device= torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Run evaluations
    print("\n📊 Running Settling Time evaluation...")
    settling_result = run_settling_time_eval(model, agent_type, lambda_penalty)
    print(f"   Mean Settling Time: {settling_result.mean_settling_time:.2f} ± {settling_result.std_settling_time:.2f}")
    print(f"   Success Rate: {settling_result.success_rate:.1%}")
    
    print("\n📊 Running CVaR evaluation...")
    cvar_result = run_cvar_eval(model, agent_type, lambda_penalty)
    print(f"   Mean Reward: {cvar_result.mean_reward:.2f} ± {cvar_result.std_reward:.2f}")
    print(f"   CVaR({int(CVAR_ALPHA*100)}%): {cvar_result.cvar_alpha:.2f}")
    
    print("\n📊 Running Step Response test...")
    step_result = run_step_response_test(model, agent_type, lambda_penalty)
    print(f"   Step: {step_result['v_initial']} → {step_result['v_final']}")
    
    return EvaluationResults(
        agent_type=agent_type,
        lambda_penalty=lambda_penalty,
        seed=seed,
        model_path=model_path,
        settling_time=settling_result,
        cvar=cvar_result,
        step_response=step_result,
    )


def evaluate_experiment(experiment_dir: str) -> List[EvaluationResults]:
    """
    Evaluate all models in an experiment directory.
    """
    exp_path = Path(experiment_dir)
    results = []
    
    # Find all model directories
    for model_dir in exp_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        # Parse experiment name: {agent}_lambda{λ}_seed{s}
        name = model_dir.name
        try:
            parts = name.split('_')
            agent_type = parts[0]
            lambda_penalty = float(parts[1].replace('lambda', ''))
            seed = int(parts[2].replace('seed', ''))
        except (IndexError, ValueError):
            print(f"Skipping {name} - couldn't parse name")
            continue
        
        # Find final model
        models_path = model_dir / "models"
        final_model = models_path / f"{name}_final.zip"
        
        if not final_model.exists():
            # Try to find best model
            best_model = models_path / "best_model.zip"
            if best_model.exists():
                final_model = best_model
            else:
                print(f"No model found for {name}")
                continue
        
        result = evaluate_model(
            str(final_model),
            agent_type,
            lambda_penalty,
            seed,
        )
        results.append(result)
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_step_response(results: List[EvaluationResults], output_path: str = None):
    """
    Plot step response comparison across agent types.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    agent_types = ['baseline', 'noise', 'shuffled', 'conditioned']
    colors = {'baseline': 'gray', 'noise': 'orange', 'shuffled': 'purple', 'conditioned': 'green'}
    
    for idx, agent_type in enumerate(agent_types):
        ax = axes[idx]
        
        # Find results for this agent type
        agent_results = [r for r in results if r.agent_type == agent_type]
        
        if not agent_results:
            ax.set_title(f"{agent_type.upper()} - No data")
            continue
        
        # Average across seeds
        all_velocities = []
        for r in agent_results:
            v = np.array(r.step_response['mean_velocity'])
            all_velocities.append(v)
        
        min_len = min(len(v) for v in all_velocities)
        velocity_matrix = np.array([v[:min_len] for v in all_velocities])
        mean_v = np.mean(velocity_matrix, axis=0)
        std_v = np.std(velocity_matrix, axis=0)
        
        # Get target trace
        target = np.array(agent_results[0].step_response['mean_target'][:min_len])
        
        steps = np.arange(len(mean_v))
        
        ax.fill_between(steps, mean_v - std_v, mean_v + std_v, 
                       alpha=0.3, color=colors[agent_type])
        ax.plot(steps, mean_v, color=colors[agent_type], linewidth=2, label='Velocity')
        ax.plot(steps, target, 'k--', linewidth=1.5, label='Target')
        
        ax.axvline(x=agent_results[0].step_response['hold_steps'], 
                  color='red', linestyle=':', label='Step Change')
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Horizontal Velocity')
        ax.set_title(f"{agent_type.upper()}")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()


def plot_metrics_comparison(results: List[EvaluationResults], output_path: str = None):
    """
    Plot bar comparison of settling time and CVaR across agent types.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    agent_types = ['baseline', 'noise', 'shuffled', 'conditioned']
    colors = ['gray', 'orange', 'purple', 'green']
    
    # Aggregate by agent type
    settling_means, settling_stds = [], []
    cvar_means, cvar_stds = [], []
    
    for agent_type in agent_types:
        agent_results = [r for r in results if r.agent_type == agent_type]
        
        if agent_results:
            st_values = [r.settling_time.mean_settling_time for r in agent_results 
                        if not np.isnan(r.settling_time.mean_settling_time)]
            cvar_values = [r.cvar.cvar_alpha for r in agent_results]
            
            settling_means.append(np.mean(st_values) if st_values else np.nan)
            settling_stds.append(np.std(st_values) if st_values else 0)
            cvar_means.append(np.mean(cvar_values))
            cvar_stds.append(np.std(cvar_values))
        else:
            settling_means.append(np.nan)
            settling_stds.append(0)
            cvar_means.append(np.nan)
            cvar_stds.append(0)
    
    x = np.arange(len(agent_types))
    
    # Settling Time plot
    axes[0].bar(x, settling_means, yerr=settling_stds, color=colors, capsize=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([a.upper() for a in agent_types])
    axes[0].set_ylabel('Settling Time (steps)')
    axes[0].set_title('Settling Time (lower is better)')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # CVaR plot
    axes[1].bar(x, cvar_means, yerr=cvar_stds, color=colors, capsize=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([a.upper() for a in agent_types])
    axes[1].set_ylabel(f'CVaR ({int(CVAR_ALPHA*100)}%)')
    axes[1].set_title('CVaR - Worst 10% (higher is better)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained agents for Policy Conditioning Study"
    )
    
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        help='Path to a single model file (.zip)'
    )
    
    parser.add_argument(
        '--model-dir', '-d',
        type=str,
        help='Path to experiment directory with multiple models'
    )
    
    parser.add_argument(
        '--agent', '-a',
        type=str,
        choices=['baseline', 'noise', 'shuffled', 'conditioned'],
        help='Agent type (required with --model-path)'
    )
    
    parser.add_argument(
        '--lambda-penalty', '-l',
        type=float,
        default=0.5,
        help='Lambda penalty used during training'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='evaluation_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate comparison plots'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model_path:
        if not args.agent:
            parser.error("--agent is required when using --model-path")
        results = [evaluate_model(args.model_path, args.agent, args.lambda_penalty)]
    elif args.model_dir:
        results = evaluate_experiment(args.model_dir)
    else:
        parser.error("Either --model-path or --model-dir is required")
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        # Convert to serializable format
        output = []
        for r in results:
            d = {
                'agent_type': r.agent_type,
                'lambda_penalty': r.lambda_penalty,
                'seed': r.seed,
                'model_path': r.model_path,
                'settling_time': {
                    'mean': r.settling_time.mean_settling_time,
                    'std': r.settling_time.std_settling_time,
                    'success_rate': r.settling_time.success_rate,
                },
                'cvar': {
                    'mean_reward': r.cvar.mean_reward,
                    'cvar_alpha': r.cvar.cvar_alpha,
                },
            }
            output.append(d)
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    
    # Print summary table
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"{'Agent':<12} {'Settling Time':<18} {'CVaR(10%)':<15} {'Mean Reward':<15}")
    print("-"*70)
    
    for r in results:
        st = f"{r.settling_time.mean_settling_time:.1f} ± {r.settling_time.std_settling_time:.1f}"
        cvar = f"{r.cvar.cvar_alpha:.1f}"
        mean = f"{r.cvar.mean_reward:.1f}"
        print(f"{r.agent_type:<12} {st:<18} {cvar:<15} {mean:<15}")
    
    print("="*70)
    
    # Generate plots if requested
    if args.plot and len(results) > 1:
        plot_step_response(results, str(output_dir / "step_response.png"))
        plot_metrics_comparison(results, str(output_dir / "metrics_comparison.png"))


if __name__ == "__main__":
    main()
