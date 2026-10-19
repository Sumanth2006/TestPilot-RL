
from src.simulator import generate_scenario


def test_same_seed_produces_same_scenario():
    first = generate_scenario(seed=42)
    second = generate_scenario(seed=42)

    assert first == second


def test_scenario_has_expected_number_of_tests():
    scenario = generate_scenario(
        num_tests=12,
        num_features=6,
        num_bugs=4,
        seed=7,
    )

    assert len(scenario["tests"]) == 12
    assert len(scenario["all_features"]) == 6
    assert len(scenario["all_bugs"]) == 4


def test_test_costs_are_positive():
    scenario = generate_scenario(seed=10)

    for test in scenario["tests"]:
        assert test["cost"] > 0


def test_each_test_covers_at_least_one_feature():
    scenario = generate_scenario(seed=10)

    for test in scenario["tests"]:
        assert len(test["features"]) >= 1


def test_invalid_input_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        generate_scenario(num_tests=0)
