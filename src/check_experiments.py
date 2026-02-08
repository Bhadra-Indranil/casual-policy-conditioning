"""
Script to check experiment integrity and generate missing training_results.txt summaries.
"""
import os
import csv
from pathlib import Path
from datetime import datetime

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"

def get_model_info(models_dir: Path) -> dict:
    """Get info about saved models in a directory."""
    if not models_dir.exists():
        return {"exists": False, "count": 0, "files": [], "has_best": False}
    
    files = list(models_dir.glob("*.zip"))
    return {
        "exists": True,
        "count": len(files),
        "files": [f.name for f in files],
        "has_best": any("best_model" in f.name for f in files),
        "total_size_mb": sum(f.stat().st_size for f in files) / (1024 * 1024) if files else 0
    }

def get_log_info(logs_dir: Path) -> dict:
    """Get info about training logs in a directory."""
    if not logs_dir.exists():
        return {"exists": False, "monitor_files": 0, "has_tensorboard": False}
    
    monitor_files = list(logs_dir.glob("*.monitor.csv"))
    tensorboard_dir = logs_dir / "tensorboard"
    
    return {
        "exists": True,
        "monitor_files": len(monitor_files),
        "has_tensorboard": tensorboard_dir.exists(),
        "monitor_paths": [f for f in monitor_files]
    }

def parse_monitor_csv(csv_path: Path) -> list:
    """Parse a monitor CSV file and return episode data."""
    episodes = []
    try:
        with open(csv_path, 'r') as f:
            # Skip the first line (metadata)
            first_line = f.readline()
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    episodes.append({
                        "reward": float(row['r']),
                        "length": int(row['l']),
                        "time": float(row['t'])
                    })
                except (KeyError, ValueError):
                    continue
    except Exception as e:
        print(f"  Warning: Could not parse {csv_path}: {e}")
    return episodes

def aggregate_seed_stats(seed_dir: Path) -> dict:
    """Aggregate statistics for a single seed run."""
    logs_dir = seed_dir / "logs"
    models_dir = seed_dir / "models"
    
    log_info = get_log_info(logs_dir)
    model_info = get_model_info(models_dir)
    
    # Aggregate episode data from all monitor files
    all_episodes = []
    if log_info["exists"] and "monitor_paths" in log_info:
        for monitor_path in log_info["monitor_paths"]:
            all_episodes.extend(parse_monitor_csv(monitor_path))
    
    stats = {
        "seed_name": seed_dir.name,
        "log_info": log_info,
        "model_info": model_info,
        "total_episodes": len(all_episodes),
    }
    
    if all_episodes:
        rewards = [ep["reward"] for ep in all_episodes]
        stats["mean_reward"] = sum(rewards) / len(rewards)
        stats["max_reward"] = max(rewards)
        stats["min_reward"] = min(rewards)
        # Last 100 episodes
        last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
        stats["last_100_mean"] = sum(last_100) / len(last_100)
        stats["total_timesteps_approx"] = sum(ep["length"] for ep in all_episodes)
    
    return stats

def check_run(run_dir: Path) -> dict:
    """Check a single run directory."""
    config_path = run_dir / "config.txt"
    results_path = run_dir / "training_results.txt"
    
    # Find all seed subdirectories
    seed_dirs = [d for d in run_dir.iterdir() if d.is_dir() and "lambda" in d.name]
    
    run_info = {
        "run_name": run_dir.name,
        "has_config": config_path.exists(),
        "has_results": results_path.exists(),
        "seed_count": len(seed_dirs),
        "seeds": []
    }
    
    for seed_dir in sorted(seed_dirs):
        seed_stats = aggregate_seed_stats(seed_dir)
        run_info["seeds"].append(seed_stats)
    
    return run_info

def generate_training_results(run_dir: Path, run_info: dict) -> str:
    """Generate training_results.txt content for a run."""
    lines = []
    lines.append(f"Training Results Summary")
    lines.append(f"=" * 50)
    lines.append(f"Run: {run_info['run_name']}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    for seed in run_info["seeds"]:
        lines.append(f"\n{seed['seed_name']}")
        lines.append("-" * 40)
        lines.append(f"  Total episodes: {seed['total_episodes']}")
        
        if seed['total_episodes'] > 0:
            lines.append(f"  Mean reward: {seed.get('mean_reward', 0):.2f}")
            lines.append(f"  Max reward: {seed.get('max_reward', 0):.2f}")
            lines.append(f"  Min reward: {seed.get('min_reward', 0):.2f}")
            lines.append(f"  Last 100 mean: {seed.get('last_100_mean', 0):.2f}")
            lines.append(f"  Approx timesteps: {seed.get('total_timesteps_approx', 0):,}")
        
        model_info = seed['model_info']
        lines.append(f"  Models saved: {model_info['count']}")
        lines.append(f"  Has best_model: {model_info['has_best']}")
        if model_info['count'] > 0:
            lines.append(f"  Total model size: {model_info.get('total_size_mb', 0):.2f} MB")
    
    return "\n".join(lines)

def main():
    print("=" * 60)
    print("EXPERIMENT INTEGRITY CHECK")
    print("=" * 60)
    
    runs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")])
    
    print(f"\nFound {len(runs)} experiment runs\n")
    
    # Track issues
    missing_results = []
    empty_models = []
    incomplete_runs = []
    
    all_run_info = []
    
    for run_dir in runs:
        run_info = check_run(run_dir)
        all_run_info.append((run_dir, run_info))
        
        print(f"\n{run_info['run_name']}")
        print(f"  config.txt: {'✓' if run_info['has_config'] else '✗'}")
        print(f"  training_results.txt: {'✓' if run_info['has_results'] else '✗ MISSING'}")
        print(f"  Seeds: {run_info['seed_count']}")
        
        if not run_info['has_results']:
            missing_results.append(run_dir)
        
        for seed in run_info['seeds']:
            model_status = "✓" if seed['model_info']['count'] > 0 else "✗ EMPTY"
            best_status = "✓" if seed['model_info']['has_best'] else "✗"
            episodes = seed['total_episodes']
            
            print(f"    {seed['seed_name']}:")
            print(f"      Episodes: {episodes}, Models: {seed['model_info']['count']} ({model_status}), Best: {best_status}")
            
            if seed['model_info']['count'] == 0:
                empty_models.append(f"{run_dir.name}/{seed['seed_name']}")
            
            if episodes > 0 and 'last_100_mean' in seed:
                print(f"      Last 100 mean reward: {seed['last_100_mean']:.2f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nRuns missing training_results.txt: {len(missing_results)}")
    for r in missing_results:
        print(f"  - {r.name}")
    
    print(f"\nSeeds with empty models/ folder: {len(empty_models)}")
    for e in empty_models:
        print(f"  - {e}")
    
    # Generate missing summaries
    print("\n" + "=" * 60)
    print("GENERATING MISSING SUMMARIES")
    print("=" * 60)
    
    for run_dir, run_info in all_run_info:
        if not run_info['has_results'] and run_info['seed_count'] > 0:
            results_content = generate_training_results(run_dir, run_info)
            results_path = run_dir / "training_results.txt"
            
            with open(results_path, 'w') as f:
                f.write(results_content)
            
            print(f"\n✓ Generated: {results_path}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
