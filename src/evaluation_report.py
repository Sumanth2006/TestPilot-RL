
import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

# Allows charts to be saved without opening a separate window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


METRICS = {
    "bugs_found": "Bugs Found",
    "features_covered": "Features Covered",
    "tests_executed": "Tests Executed",
    "budget_used": "Budget Used",
    "total_reward": "Total Reward",
}


def generate_evaluation_report(
    evaluation_rows: list[dict[str, Any]],
    output_dir: str = "results",
) -> dict[str, str]:
    """Generate statistical summaries and charts from evaluation results."""

    if not evaluation_rows:
        raise ValueError("evaluation_rows cannot be empty.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    strategies = ("q_learning", "random", "fixed_order")
    summary_rows = []

    # Calculate the mean and standard deviation for every strategy.
    for strategy in strategies:
        strategy_rows = [
            row
            for row in evaluation_rows
            if row["strategy"] == strategy
        ]

        if not strategy_rows:
            raise ValueError(
                f"No evaluation results found for {strategy}."
            )

        for metric, label in METRICS.items():
            values = [float(row[metric]) for row in strategy_rows]

            summary_rows.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "metric_label": label,
                    "mean": round(mean(values), 3),
                    "std_dev": round(stdev(values), 3)
                    if len(values) > 1
                    else 0.0,
                    "num_scenarios": len(values),
                }
            )

    # Save the statistical summary as a CSV file.
    summary_csv = output_path / "metrics_summary.csv"

    with open(summary_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "strategy",
                "metric",
                "metric_label",
                "mean",
                "std_dev",
                "num_scenarios",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Chart 1: average bugs discovered, with standard deviation.
    bug_rows = [
        row for row in summary_rows if row["metric"] == "bugs_found"
    ]

    _plot_metric(
        bug_rows,
        title="Bug Detection Performance",
        ylabel="Average Bugs Found",
        output_file=output_path / "bug_detection_comparison.png",
    )

    # Chart 2: average reward, with standard deviation.
    reward_rows = [
        row for row in summary_rows if row["metric"] == "total_reward"
    ]

    _plot_metric(
        reward_rows,
        title="Total Reward Comparison",
        ylabel="Average Total Reward",
        output_file=output_path / "reward_comparison.png",
    )

    # Chart 3: average feature coverage, with standard deviation.
    coverage_rows = [
        row
        for row in summary_rows
        if row["metric"] == "features_covered"
    ]

    _plot_metric(
        coverage_rows,
        title="Feature Coverage Comparison",
        ylabel="Average Features Covered",
        output_file=output_path / "feature_coverage_comparison.png",
    )

    return {
        "metrics_summary": str(summary_csv),
        "bug_chart": str(output_path / "bug_detection_comparison.png"),
        "reward_chart": str(output_path / "reward_comparison.png"),
        "coverage_chart": str(
            output_path / "feature_coverage_comparison.png"
        ),
    }


def _plot_metric(
    rows: list[dict[str, Any]],
    title: str,
    ylabel: str,
    output_file: Path,
) -> None:
    """Plot strategy means with standard-deviation error bars."""

    labels = [row["strategy"].replace("_", " ").title() for row in rows]
    values = [row["mean"] for row in rows]
    errors = [row["std_dev"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        labels,
        values,
        yerr=errors,
        capsize=6,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Testing Strategy")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Display the average above each bar.
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
