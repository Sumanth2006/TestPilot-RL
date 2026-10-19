
from src.baselines import (
    run_fixed_order_baseline,
    run_random_baseline,
)
from src.simulator import generate_scenario


def test_random_baseline_returns_valid_metrics():
    scenario = generate_scenario(seed=42)

    result = run_random_baseline(
        scenario=scenario,
        initial_budget=20,
        seed=123,
    )

    assert result["strategy"] == "random"
    assert 0 <= result["bugs_found"] <= len(scenario["all_bugs"])
    assert 0 <= result["features_covered"] <= len(scenario["all_features"])
    assert result["tests_executed"] >= 0
    assert 0 <= result["budget_used"] <= 20
    assert result["remaining_budget"] == 20 - result["budget_used"]


def test_random_baseline_is_reproducible():
    scenario = generate_scenario(seed=42)

    first = run_random_baseline(scenario, seed=123)
    second = run_random_baseline(scenario, seed=123)

    assert first == second


def test_fixed_order_baseline_returns_valid_metrics():
    scenario = generate_scenario(seed=42)

    result = run_fixed_order_baseline(
        scenario=scenario,
        initial_budget=20,
    )

    assert result["strategy"] == "fixed_order"
    assert result["bugs_found"] >= 0
    assert result["features_covered"] >= 0
    assert result["tests_executed"] >= 0
    assert result["budget_used"] <= 20
    assert result["remaining_budget"] == 20 - result["budget_used"]


def test_baselines_handle_zero_budget():
    scenario = generate_scenario(seed=42)

    random_result = run_random_baseline(
        scenario, initial_budget=0, seed=123
    )
    fixed_result = run_fixed_order_baseline(
        scenario, initial_budget=0
    )

    for result in (random_result, fixed_result):
        assert result["tests_executed"] == 0
        assert result["bugs_found"] == 0
        assert result["features_covered"] == 0
        assert result["budget_used"] == 0
        assert result["remaining_budget"] == 0
