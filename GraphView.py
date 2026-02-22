import argparse
import csv
import os
from typing import Dict, List

import numpy as np


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_generation_summary(run_dir: str) -> Dict[str, np.ndarray]:
    path = os.path.join(run_dir, "generation_summary.csv")
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"Generation summary is empty: {path}")

    def col(name: str, cast=float) -> np.ndarray:
        values = []
        for row in rows:
            raw = row.get(name, "")
            if raw == "":
                values.append(np.nan)
            else:
                values.append(cast(raw))
        dtype = np.int32 if cast is int else np.float32
        return np.asarray(values, dtype=dtype)

    return {
        "generation": col("generation", int),
        "dist_avg": col("dist_avg"),
        "dist_best_gen": col("dist_best_gen"),
        "dist_best_global": col("dist_best_global"),
        "time_avg": col("time_avg"),
        "time_best_gen": col("time_best_gen"),
        "time_best_global": col("time_best_global"),
        "finish_rate": col("finish_rate"),
        "crash_rate": col("crash_rate"),
        "timeout_rate": col("timeout_rate"),
    }


def load_individual_metrics(run_dir: str) -> List[Dict[str, str]]:
    path = os.path.join(run_dir, "individual_metrics.csv")
    return _read_csv_rows(path)


def plot_generation_curves(
    summary: Dict[str, np.ndarray], output_dir: str, prefix: str = "ga"
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it via 'pip install matplotlib'."
        ) from exc

    os.makedirs(output_dir, exist_ok=True)

    gens = summary["generation"]
    if gens.size == 0:
        raise ValueError("No generations in summary data.")

    generated_files: List[str] = []

    dist_file = os.path.join(output_dir, f"{prefix}_distance.png")
    plt.figure(figsize=(8, 4))
    plt.plot(gens, summary["dist_avg"], label="Generation average", linewidth=2)
    plt.plot(gens, summary["dist_best_gen"], label="Generation best", linewidth=2)
    plt.plot(gens, summary["dist_best_global"], label="Global best", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Distance traveled [%]")
    plt.ylim(0, 101)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(dist_file, dpi=200)
    plt.close()
    generated_files.append(dist_file)

    time_file = os.path.join(output_dir, f"{prefix}_time.png")
    plt.figure(figsize=(8, 4))
    plt.plot(gens, summary["time_avg"], label="Generation average", linewidth=2)
    plt.plot(gens, summary["time_best_gen"], label="Generation best", linewidth=2)
    plt.plot(gens, summary["time_best_global"], label="Global best", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Time [s]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(time_file, dpi=200)
    plt.close()
    generated_files.append(time_file)

    rates_file = os.path.join(output_dir, f"{prefix}_termination_rates.png")
    plt.figure(figsize=(8, 4))
    plt.plot(gens, summary["finish_rate"], label="Finish rate", linewidth=2)
    plt.plot(gens, summary["crash_rate"], label="Crash rate", linewidth=2)
    plt.plot(gens, summary["timeout_rate"], label="Timeout rate", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Rate")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(rates_file, dpi=200)
    plt.close()
    generated_files.append(rates_file)

    return generated_files


def generate_plots_for_run(run_dir: str, prefix: str = "ga") -> List[str]:
    summary = load_generation_summary(run_dir)
    output_dir = os.path.join(run_dir, "plots")
    return plot_generation_curves(summary=summary, output_dir=output_dir, prefix=prefix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load GA training logs and generate analysis plots."
    )
    parser.add_argument("run_dir", help="Path to a GA run directory")
    parser.add_argument(
        "--prefix",
        default="ga",
        help="Output filename prefix for generated plots",
    )
    args = parser.parse_args()

    generated = generate_plots_for_run(run_dir=args.run_dir, prefix=args.prefix)
    print("Generated plots:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
