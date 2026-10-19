
import random
from typing import Any

from src.environment import TestingEnvironment


def run_random_baseline(
    scenario: dict[str, Any],
    initial_budget: int = 20,
    bug_weight: float = 10.0,
    coverage_weight: float = 2.0,
    cost_weight: float = 1.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate random selection among affordable, unexecuted tests."""

    rng = random.Random(seed)

    environment = TestingEnvironment(
        scenario=scenario,
        initial_budget=initial_budget,
        bug_weight=bug_weight,
        coverage_weight=coverage_weight,
        cost_weight=cost_weight,
    )

    total_reward = 0.0
    tests_executed = 0

    while True:
        legal_actions = environment.get_legal_actions()

        if not legal_actions:
            break

        action = rng.choice(legal_actions)
        _, reward, done, _ = environment.step(action)

        total_reward += reward
        tests_executed += 1

        if done:
            break

    return {
        "strategy": "random",
        "total_reward": total_reward,
        "bugs_found": len(environment.discovered_bugs),
        "features_covered": len(environment.covered_features),
        "tests_executed": tests_executed,
        "budget_used": initial_budget - environment.remaining_budget,
        "remaining_budget": environment.remaining_budget,
    }


def run_fixed_order_baseline(
    scenario: dict[str, Any],
    initial_budget: int = 20,
    bug_weight: float = 10.0,
    coverage_weight: float = 2.0,
    cost_weight: float = 1.0,
) -> dict[str, Any]:
    """Evaluate tests in their original order, skipping unaffordable tests."""

    environment = TestingEnvironment(
        scenario=scenario,
        initial_budget=initial_budget,
        bug_weight=bug_weight,
        coverage_weight=coverage_weight,
        cost_weight=cost_weight,
    )

    total_reward = 0.0
    tests_executed = 0

    # Preserve the scenario's original test order.
    for test in environment.tests:
        if test["id"] not in environment.get_legal_actions():
            continue

        _, reward, done, _ = environment.step(test["id"])

        total_reward += reward
        tests_executed += 1

        if done:
            break

    return {
        "strategy": "fixed_order",
        "total_reward": total_reward,
        "bugs_found": len(environment.discovered_bugs),
        "features_covered": len(environment.covered_features),
        "tests_executed": tests_executed,
        "budget_used": initial_budget - environment.remaining_budget,
        "remaining_budget": environment.remaining_budget,
    }
