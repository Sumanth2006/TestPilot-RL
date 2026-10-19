
import pytest

from src.environment import TestingEnvironment
from src.simulator import generate_scenario


def make_environment(budget=20):
    scenario = {
        "tests": [
            {
                "id": "T1",
                "cost": 2,
                "features": {"F1"},
                "bugs": {"B1"},
            },
            {
                "id": "T2",
                "cost": 3,
                "features": {"F1", "F2"},
                "bugs": {"B1", "B2"},
            },
        ],
        "all_features": {"F1", "F2"},
        "all_bugs": {"B1", "B2"},
    }

    return TestingEnvironment(scenario, initial_budget=budget)


def test_reset_restores_initial_state():
    env = make_environment()
    env.step("T1")

    state = env.reset()

    assert state["remaining_budget"] == 20
    assert state["executed_tests"] == frozenset()
    assert state["discovered_bugs"] == frozenset()


def test_executing_test_reduces_budget():
    env = make_environment(budget=5)

    env.step("T1")

    assert env.remaining_budget == 3


def test_bugs_are_revealed_after_execution():
    env = make_environment()

    assert env.discovered_bugs == set()

    _, _, _, info = env.step("T1")

    assert env.discovered_bugs == {"B1"}
    assert info["new_bugs"] == {"B1"}


def test_repeated_test_is_illegal():
    env = make_environment()
    env.step("T1")

    with pytest.raises(ValueError):
        env.step("T1")


def test_unaffordable_test_is_not_legal():
    env = make_environment(budget=2)

    assert env.get_legal_actions() == ["T1"]


def test_reward_uses_new_bugs_coverage_and_cost():
    env = make_environment(budget=5)

    _, reward, _, _ = env.step("T1")

    # 1 new bug * 10 + 1 new feature * 2 - cost 2 * 1
    assert reward == 10


def test_invalid_budget_is_rejected():
    scenario = generate_scenario(seed=42)

    with pytest.raises(ValueError):
        TestingEnvironment(scenario, initial_budget=-1)
