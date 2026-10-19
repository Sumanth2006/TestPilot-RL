
import random
from typing import Any


def generate_scenario(
    num_tests: int = 20,
    num_features: int = 10,
    num_bugs: int = 10,
    max_test_cost: int = 5,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate one reproducible synthetic software-testing scenario."""

    if num_tests <= 0:
        raise ValueError("num_tests must be positive.")

    if num_features <= 0:
        raise ValueError("num_features must be positive.")

    if num_bugs < 0:
        raise ValueError("num_bugs cannot be negative.")

    if max_test_cost <= 0:
        raise ValueError("max_test_cost must be positive.")

    rng = random.Random(seed)

    features = [
        f"F{i + 1}" for i in range(num_features)
    ]

    bugs = [
        f"B{i + 1}" for i in range(num_bugs)
    ]

    tests = []

    for i in range(num_tests):
        # Every test covers at least one feature.
        num_covered = rng.randint(1, num_features)
        covered_features = set(
            rng.sample(features, num_covered)
        )

        # A test may detect zero, one, or several bugs.
        # This mapping remains internal to the simulator.
        detected_bugs = {
            bug for bug in bugs if rng.random() < 0.20
        }

        tests.append(
            {
                "id": f"T{i + 1}",
                "cost": rng.randint(1, max_test_cost),
                "features": covered_features,
                "bugs": detected_bugs,
            }
        )

    return {
        "tests": tests,
        "all_features": set(features),
        "all_bugs": set(bugs),
    }
