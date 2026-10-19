
import json

from src.experiments import run_experiment


def test_experiment_creates_result_files(tmp_path):
    config_path = tmp_path / "experiment.json"

    config_path.write_text(
        json.dumps(
            {
                "num_tests": 6,
                "num_features": 4,
                "num_bugs": 3,
                "reward": {
                    "bug_weight": 10.0,
                    "coverage_weight": 2.0,
                    "cost_weight": 1.0,
                },
                "q_learning": {
                    "learning_rate": 0.1,
                    "discount_factor": 0.9,
                    "initial_epsilon": 1.0,
                    "minimum_epsilon": 0.05,
                    "epsilon_decay": 0.995,
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_experiment(
        training_scenarios=2,
        evaluation_scenarios=3,
        episodes=5,
        initial_budget=8,
        seed=42,
        output_dir=str(tmp_path / "results"),
        config_path=str(config_path),
    )

    assert len(result["evaluation_rows"]) == 9
    assert set(result["summary"]) == {
        "q_learning",
        "random",
        "fixed_order",
    }

    for path in result["files"].values():
        assert (tmp_path / "results" / path.split("/")[-1]).exists()


def test_experiment_is_reproducible(tmp_path):
    config_path = tmp_path / "experiment.json"

    config_path.write_text(
        json.dumps(
            {
                "num_tests": 6,
                "num_features": 4,
                "num_bugs": 3,
                "reward": {},
                "q_learning": {},
            }
        ),
        encoding="utf-8",
    )

    first = run_experiment(
        training_scenarios=2,
        evaluation_scenarios=2,
        episodes=4,
        initial_budget=8,
        seed=99,
        output_dir=str(tmp_path / "first"),
        config_path=str(config_path),
    )

    second = run_experiment(
        training_scenarios=2,
        evaluation_scenarios=2,
        episodes=4,
        initial_budget=8,
        seed=99,
        output_dir=str(tmp_path / "second"),
        config_path=str(config_path),
    )

    assert first["summary"] == second["summary"]
    assert first["evaluation_rows"] == second["evaluation_rows"]


def test_experiment_rejects_invalid_counts(tmp_path):
    config_path = tmp_path / "experiment.json"
    config_path.write_text("{}", encoding="utf-8")

    try:
        run_experiment(
            training_scenarios=0,
            config_path=str(config_path),
            output_dir=str(tmp_path / "results"),
        )
    except ValueError as error:
        assert "training_scenarios" in str(error)
    else:
        raise AssertionError("Expected ValueError for zero training scenarios.")

