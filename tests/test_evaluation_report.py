
import csv

import pytest

from src.evaluation_report import generate_evaluation_report


def test_report_creates_summary_and_all_charts(tmp_path):
    evaluation_rows = [
        {
            "strategy": strategy,
            "bugs_found": bugs,
            "features_covered": 4,
            "tests_executed": 3,
            "budget_used": 8,
            "total_reward": reward,
        }
        for strategy, bugs, reward in [
            ("q_learning", 3, 30),
            ("q_learning", 5, 50),
            ("random", 2, 20),
            ("random", 4, 40),
            ("fixed_order", 1, 10),
            ("fixed_order", 3, 30),
        ]
    ]

    files = generate_evaluation_report(
        evaluation_rows=evaluation_rows,
        output_dir=str(tmp_path),
    )

    assert set(files) == {
        "metrics_summary",
        "bug_chart",
        "reward_chart",
        "coverage_chart",
    }

    for path in files.values():
        output_file = tmp_path / path.split("/")[-1]
        assert output_file.exists()
        assert output_file.stat().st_size > 0


def test_report_calculates_mean_and_standard_deviation(tmp_path):
    evaluation_rows = [
        {
            "strategy": "q_learning",
            "bugs_found": bugs,
            "features_covered": 4,
            "tests_executed": 3,
            "budget_used": 8,
            "total_reward": 10,
        }
        for bugs in (2, 4)
    ]

    # Include the other strategies so the report has complete results.
    for strategy in ("random", "fixed_order"):
        evaluation_rows.append(
            {
                "strategy": strategy,
                "bugs_found": 3,
                "features_covered": 4,
                "tests_executed": 3,
                "budget_used": 8,
                "total_reward": 10,
            }
        )

    generate_evaluation_report(
        evaluation_rows=evaluation_rows,
        output_dir=str(tmp_path),
    )

    with (tmp_path / "metrics_summary.csv").open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    q_learning_bugs = next(
        row
        for row in rows
        if row["strategy"] == "q_learning"
        and row["metric"] == "bugs_found"
    )

    assert float(q_learning_bugs["mean"]) == 3.0
    assert float(q_learning_bugs["std_dev"]) == pytest.approx(1.414, abs=0.001)
    assert int(q_learning_bugs["num_scenarios"]) == 2


def test_report_rejects_empty_evaluation_rows(tmp_path):
    with pytest.raises(ValueError, match="cannot be empty"):
        generate_evaluation_report(
            evaluation_rows=[],
            output_dir=str(tmp_path),
        )
