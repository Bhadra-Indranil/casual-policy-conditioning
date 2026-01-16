from custom_lander import make_env
env = make_env(agent_type="conditioned")
obs, info = env.reset()
total_steps = 0
for _ in range(300):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    total_steps += 1
    if info["target_changed"]:
        print(f"Step {total_steps}: Target changed to {info['target_vx']:.3f}")
    if terminated or truncated:
        print(f"Episode ended at step {total_steps}, resetting...")
        obs, info = env.reset()
env.close()