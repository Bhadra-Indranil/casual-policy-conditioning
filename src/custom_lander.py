"""
VelocityTrackingLander: Non-Stationary Environment for Policy Conditioning Study

Implements the experimental environment from:
"When Does Policy Conditioning Help? A Controlled Study of Online Adaptation 
Speed in Continuous Control"

Key Design Decisions (per document):
- Signal c_t ∈ [0, 1] maps to v_target ∈ [v_min, v_max] via linear transformation
- Shuffled-Control uses permutation of episode's valid targets (information equivalence)
- Reward: R_t = R_env - λ * |v_current - v_target| / (v_max - v_min)
- Target changes every N=150 steps (piecewise-constant schedule)
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class VelocityTrackingLander(gym.Wrapper):
    """
    Wrapper for Causal Policy Conditioning experiments.
    
    Agent Types:
        - 'baseline': 8-D obs (no signal) - unconditional optimum
        - 'noise': 9-D obs with Gaussian noise - controls for input dimensionality
        - 'shuffled': 9-D obs with permuted valid signals - controls for marginal distribution
        - 'conditioned': 9-D obs with true signal - experimental condition
    
    Args:
        env: Base LunarLanderContinuous environment
        agent_type: One of 'baseline', 'noise', 'shuffled', 'conditioned'
        change_interval: Steps between target changes (default: 150)
        lambda_penalty: Velocity tracking penalty coefficient (default: 0.5)
        v_min: Minimum target velocity (default: -1.0)
        v_max: Maximum target velocity (default: 1.0)
        max_episode_steps: Max steps for pre-computing shuffled schedule (default: 1000)
    """

    VALID_AGENT_TYPES = {'baseline', 'noise', 'shuffled', 'conditioned'}

    def __init__(
        self, 
        env, 
        agent_type: str = 'conditioned', 
        change_interval: int = 150,
        lambda_penalty: float = 0.5,
        v_min: float = -1.0,
        v_max: float = 1.0,
        max_episode_steps: int = 1000
    ):
        super().__init__(env)
        
        if agent_type not in self.VALID_AGENT_TYPES:
            raise ValueError(
                f"Unknown agent_type: '{agent_type}'. "
                f"Expected one of {self.VALID_AGENT_TYPES}"
            )
        
        self.agent_type = agent_type
        self.change_interval = change_interval
        self.lambda_penalty = lambda_penalty
        self.v_min = v_min
        self.v_max = v_max
        self.v_range = v_max - v_min
        self.max_episode_steps = max_episode_steps
        
        # Episode state
        self.steps = 0
        self.current_signal = 0.5  # c_t ∈ [0, 1]
        self.target_vx = 0.0       # v_target ∈ [v_min, v_max]
        
        # For shuffled baseline: pre-generated schedule
        self.target_schedule = []       # True schedule of signals
        self.shuffled_schedule = []     # Permuted schedule for shuffled agent
        self.schedule_index = 0
        
        # Noise baseline: use Uniform[0,1] to exactly match signal marginal distribution
        # This ensures the noise control differs only in temporal correlation, not marginals
        
        # Observation space: append 1 dimension for signal (except baseline)
        low = np.array(self.env.observation_space.low, dtype=np.float32)
        high = np.array(self.env.observation_space.high, dtype=np.float32)
        
        if self.agent_type != 'baseline':
            low = np.append(low, 0.0)   # Signal ∈ [0, 1]
            high = np.append(high, 1.0)
            
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def _signal_to_velocity(self, signal: float) -> float:
        """Maps signal c_t ∈ [0,1] to v_target ∈ [v_min, v_max]."""
        return self.v_min + signal * self.v_range

    def _generate_episode_schedule(self) -> list:
        """Pre-generate the full episode's target schedule."""
        num_targets = (self.max_episode_steps // self.change_interval) + 2
        return [np.random.uniform(0.0, 1.0) for _ in range(num_targets)]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.steps = 0
        self.schedule_index = 0
        
        # Generate episode schedule
        self.target_schedule = self._generate_episode_schedule()
        
        # For shuffled: create a random permutation of the same values
        self.shuffled_schedule = self.target_schedule.copy()
        np.random.shuffle(self.shuffled_schedule)
        
        # Set initial target
        self.current_signal = self.target_schedule[0]
        self.target_vx = self._signal_to_velocity(self.current_signal)
        
        # Logging
        info['target_vx'] = self.target_vx
        info['current_vx'] = obs[2]  # vel_x
        info['signal'] = self.current_signal
        info['schedule_index'] = self.schedule_index
        
        return self._get_obs(obs), info

    def step(self, action):
        self.steps += 1
        
        # Check for target change (non-stationarity)
        target_changed = False
        if self.steps % self.change_interval == 0:
            self.schedule_index += 1
            if self.schedule_index < len(self.target_schedule):
                self.current_signal = self.target_schedule[self.schedule_index]
                self.target_vx = self._signal_to_velocity(self.current_signal)
                target_changed = True

        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # Reward shaping: R_t = R_env - λ * |v_current - v_target| / (v_max - v_min)
        current_vx = next_obs[2]  # vel_x index in LunarLander state
        normalized_error = abs(current_vx - self.target_vx) / self.v_range
        reward = reward - (self.lambda_penalty * normalized_error)

        # Logging for Phase 3 analysis
        info['target_vx'] = self.target_vx
        info['current_vx'] = current_vx
        info['signal'] = self.current_signal
        info['velocity_error'] = abs(current_vx - self.target_vx)
        info['normalized_error'] = normalized_error
        info['schedule_index'] = self.schedule_index
        info['target_changed'] = target_changed

        return self._get_obs(next_obs), reward, terminated, truncated, info

    def _get_obs(self, state: np.ndarray) -> np.ndarray:
        """Construct observation vector based on agent type."""
        
        if self.agent_type == 'baseline':
            # No signal - agent must infer from reward dynamics
            return state.astype(np.float32)
        
        elif self.agent_type == 'conditioned':
            # True signal - temporal correlation with reward
            signal = self.current_signal
            
        elif self.agent_type == 'noise':
            # Random noise - same marginal distribution as signal, 
            # but zero mutual information with reward (i.i.d. each step)
            signal = np.random.uniform(0.0, 1.0)
            
        elif self.agent_type == 'shuffled':
            # Permuted valid signal - same marginal distribution, 
            # zero temporal correlation with current reward
            if self.schedule_index < len(self.shuffled_schedule):
                signal = self.shuffled_schedule[self.schedule_index]
            else:
                signal = np.random.uniform(0.0, 1.0)
        
        return np.concatenate([state, [signal]]).astype(np.float32)


def make_env(
    agent_type: str = 'conditioned',
    change_interval: int = 150,
    lambda_penalty: float = 0.5,
    render_mode: str = None
) -> VelocityTrackingLander:
    """
    Factory function to create the wrapped environment.
    
    Args:
        agent_type: One of 'baseline', 'noise', 'shuffled', 'conditioned'
        change_interval: Steps between target changes
        lambda_penalty: Velocity tracking penalty coefficient
        render_mode: Gymnasium render mode ('human', 'rgb_array', None)
    
    Returns:
        VelocityTrackingLander environment
    """
    base_env = gym.make("LunarLanderContinuous-v3", render_mode=render_mode)
    return VelocityTrackingLander(
        base_env,
        agent_type=agent_type,
        change_interval=change_interval,
        lambda_penalty=lambda_penalty
    )