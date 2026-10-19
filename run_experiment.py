
from src.experiments import run_experiment


if __name__ == "__main__":
    results = run_experiment(
        training_scenarios=30,
        evaluation_scenarios=50,
        episodes=500,
        initial_budget=20,
        seed=42,
    )

    print("\n=== HELD-OUT EVALUATION SUMMARY ===")

    for strategy, metrics in results["summary"].items():
        print(f"\n{strategy.upper()}")

        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    print("\n=== STATISTICAL SUMMARY ===")

    from pathlib import Path
    import csv

    summary_path = Path("results/metrics_summary.csv")

    with summary_path.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        for row in csv.DictReader(file):
            print(
                f"{row['strategy']:12} | "
                f"{row['metric_label']:18} | "
                f"Mean: {float(row['mean']):7.3f} | "
                f"Std Dev: {float(row['std_dev']):7.3f}"
            )

    print("\n=== SAVED FILES ===")

    for name, path in results["files"].items():
        print(f"{name}: {path}")
