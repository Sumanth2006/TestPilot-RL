
from src.experiments import run_experiment

if __name__ == "__main__":
    results = run_experiment(
        training_scenarios=5,
        evaluation_scenarios=5,
        episodes=50,
        initial_budget=20,
        seed=42,
    )

    print("\n=== HELD-OUT EVALUATION SUMMARY ===")

    for strategy, metrics in results["summary"].items():
        print(f"\n{strategy.upper()}")

        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    print("\n=== SAVED FILES ===")

    for name, path in results["files"].items():
        print(f"{name}: {path}")

