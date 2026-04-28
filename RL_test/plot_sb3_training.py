import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Stable-Baselines3 Trackmania episode metrics.")
    parser.add_argument("--run-dir", required=True, help="Run directory under logs/sb3_runs.")
    return parser.parse_args()


def read_metrics(path: Path) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                rows.setdefault(key, [])
                try:
                    rows[key].append(float(value))
                except (TypeError, ValueError):
                    rows[key].append(float("nan"))
    return rows


def save_plot(run_dir: Path, metrics: dict[str, list[float]], y_key: str, filename: str, ylabel: str) -> None:
    episodes = metrics.get("episode", [])
    values = metrics.get(y_key, [])
    if not episodes or not values:
        return
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, values, linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(run_dir / filename, dpi=140)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "episode_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    metrics = read_metrics(metrics_path)
    save_plot(run_dir, metrics, "progress", "sb3_progress.png", "Progress (%)")
    save_plot(run_dir, metrics, "episode_reward", "sb3_episode_reward.png", "Episode Reward")
    save_plot(run_dir, metrics, "time", "sb3_time.png", "Episode Time (s)")
    save_plot(run_dir, metrics, "distance", "sb3_distance.png", "Distance")
    print(f"Saved plots in: {run_dir}")


if __name__ == "__main__":
    main()
