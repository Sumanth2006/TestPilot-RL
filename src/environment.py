
from typing import Any


class TestingEnvironment:
    __test__ = False
    """Environment in which an agent selects tests under a fixed budget."""

    def __init__(
        self,
        scenario: dict[str, Any],
        initial_budget: int = 20,
        bug_weight: float = 10.0,
        coverage_weight: float = 2.0,
        cost_weight: float = 1.0,
    ) -> None:
        if initial_budget < 0:
            raise ValueError("Initial budget cannot be negative.")

        if bug_weight < 0 or coverage_weight < 0 or cost_weight < 0:
            raise ValueError("Reward weights cannot be negative.")

        self.tests = scenario["tests"]
        self.all_features = set(scenario["all_features"])
        self.all_bugs = set(scenario["all_bugs"])

        self.initial_budget = initial_budget
        self.bug_weight = bug_weight
        self.coverage_weight = coverage_weight
        self.cost_weight = cost_weight

        self.reset()

    def reset(self) -> dict[str, Any]:
        """Start a fresh episode and return the initial state."""
        self.remaining_budget = self.initial_budget
        self.executed_tests: set[str] = set()
        self.discovered_bugs: set[str] = set()
        self.covered_features: set[str] = set()

        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """Return only information the agent is allowed to observe."""
        return {
            "remaining_budget": self.remaining_budget,
            "executed_tests": frozenset(self.executed_tests),
            "discovered_bugs": frozenset(self.discovered_bugs),
            "covered_features": frozenset(self.covered_features),
        }

    def get_legal_actions(self) -> list[str]:
        """Return tests that are unexecuted and affordable."""
        return [
            test["id"]
            for test in self.tests
            if test["id"] not in self.executed_tests
            and test["cost"] <= self.remaining_budget
        ]

    def step(
        self, action: str
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute a test and return state, reward, done, and info."""
        if action not in self.get_legal_actions():
            raise ValueError(
                f"Illegal action: {action}. "
                "Choose an unexecuted, affordable test."
            )

        test = next(
            test for test in self.tests if test["id"] == action
        )

        previous_bug_count = len(self.discovered_bugs)
        previous_feature_count = len(self.covered_features)

        self.executed_tests.add(action)
        self.remaining_budget -= test["cost"]

        # Reveal the selected test's outcome only after execution.
        new_bugs = set(test["bugs"]) - self.discovered_bugs
        new_features = set(test["features"]) - self.covered_features

        self.discovered_bugs.update(new_bugs)
        self.covered_features.update(new_features)

        bugs_found = len(self.discovered_bugs) - previous_bug_count
        features_added = (
            len(self.covered_features) - previous_feature_count
        )

        reward = (
            self.bug_weight * bugs_found
            + self.coverage_weight * features_added
            - self.cost_weight * test["cost"]
        )

        done = (
            self.remaining_budget == 0
            or len(self.get_legal_actions()) == 0
        )

        info = {
            "test_id": action,
            "cost": test["cost"],
            "new_bugs": new_bugs,
            "bugs_found": bugs_found,
            "new_features": new_features,
            "features_added": features_added,
        }

        return self.get_state(), reward, done, info
