import gymnasium as gym
import numpy as np
from custom_lander import VelocityTrackingLander, make_env
from stable_baselines3.common.env_checker import check_env

def run_lab_test():
    print("🔬 INITIALIZING PHASE 1 VERIFICATION...\n")
    print("=" * 60)
    
    agents = ['baseline', 'noise', 'shuffled', 'conditioned']
    
    for agent in agents:
        print(f"\n--- Testing Agent: {agent.upper()} ---")
        env = make_env(agent_type=agent, lambda_penalty=0.5)
        
        # 1. Check Space Compliance (SB3 requirement)
        try:
            check_env(env)
            print("✅ Gym API Compliance: PASS")
        except Exception as e:
            print(f"❌ Gym API Compliance: FAIL - {e}")
            
        # 2. Check Observation Content
        obs, info = env.reset()
        print(f"   Obs Shape: {obs.shape}")
        print(f"   Obs Space: {env.observation_space.shape}")
        
        if agent == 'baseline':
            if len(obs) == 8: 
                print("✅ Input Dimension: CORRECT (8)")
            else: 
                print(f"❌ Input Dimension: FAIL (got {len(obs)})")
        else:
            signal = obs[-1]
            signal_from_info = info['signal']
            target = info['target_vx']
            
            if len(obs) == 9: 
                print("✅ Input Dimension: CORRECT (9)")
            else:
                print(f"❌ Input Dimension: FAIL (got {len(obs)})")
            
            print(f"   Signal: {signal:.4f}, Target v_x: {target:.4f}")
            
            if agent == 'conditioned':
                # Signal should equal info['signal'] (the c_t value)
                if np.isclose(signal, signal_from_info): 
                    print("✅ Signal Integrity: PASS (obs[-1] == info['signal'])")
                else: 
                    print("❌ Signal Integrity: FAIL")
            
            elif agent == 'shuffled':
                # Check that shuffled schedule is a permutation
                if env.target_schedule != env.shuffled_schedule:
                    if set(env.target_schedule) == set(env.shuffled_schedule):
                        print("✅ Shuffled Schedule: PASS (Permutation verified)")
                    else:
                        print("⚠️ Shuffled Schedule: Values don't match")
                else:
                    print("⚠️ Shuffled Schedule: Same order (rare chance)")
                    
            elif agent == 'noise':
                print(f"   Noise signal range: [0, 1], value: {signal:.4f}")
        
        env.close()

    # Test dynamics change
    print("\n" + "=" * 60)
    print("--- Testing Dynamics Change (change_interval=10) ---")
    env = make_env(agent_type='conditioned', change_interval=10, lambda_penalty=0.5)
    obs, info = env.reset()
    initial_target = info['target_vx']
    initial_signal = info['signal']
    print(f"   Initial: signal={initial_signal:.4f}, target_vx={initial_target:.4f}")
    
    prev_target = initial_target
    for i in range(1, 16):
        _, _, _, _, info = env.step(env.action_space.sample())
        current_target = info['target_vx']
        
        if info.get('target_changed', False):
            print(f"✅ Target Switch at Step {i}: {prev_target:.3f} -> {current_target:.3f}")
            prev_target = current_target
    
    env.close()
    
    # Test signal-to-velocity mapping
    print("\n" + "=" * 60)
    print("--- Testing Signal-to-Velocity Mapping ---")
    env = make_env(agent_type='conditioned')
    
    # Check mapping: c=0 -> v_min, c=1 -> v_max
    test_signals = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("   Signal c_t -> Target v_x mapping (v_min=-1, v_max=1):")
    for c in test_signals:
        v = env._signal_to_velocity(c)
        expected = -1.0 + c * 2.0
        status = "✅" if np.isclose(v, expected) else "❌"
        print(f"   {status} c={c:.2f} -> v_target={v:.2f} (expected {expected:.2f})")
    
    env.close()
    
    # Test reward computation
    print("\n" + "=" * 60)
    print("--- Testing Reward Penalty (λ=0.5) ---")
    env = make_env(agent_type='conditioned', lambda_penalty=0.5)
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    print(f"   velocity_error: {info['velocity_error']:.4f}")
    print(f"   normalized_error: {info['normalized_error']:.4f}")
    print(f"   penalty = λ * normalized_error = {0.5 * info['normalized_error']:.4f}")
    env.close()
    
    print("\n" + "=" * 60)
    print("🔬 PHASE 1 VERIFICATION COMPLETE\n")

if __name__ == "__main__":
    run_lab_test()