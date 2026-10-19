
import random
from typing import Any


class QLearningAgent:
    """Learn which software test to execute using tabular Q-learning."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        if not 0 < learning_rate <= 1:
            raise ValueError("Learning rate must be in (0, 1].")

        if not 0 <= discount_factor <= 1:
            raise ValueError("Discount factor must be in [0, 1].")

        
        
        if not 0 <= epsilon <= 1:
             raise ValueError("Epsilon must be in [0, 1].")

        if not 0 <= min_epsilon <= 1:
             raise ValueError("Minimum epsilon must be in [0, 1].")

        if epsilon > 0 and min_epsilon > epsilon:
             raise ValueError(
        "Minimum epsilon cannot exceed a positive initial epsilon."
              )



        if not 0 < epsilon_decay <= 1:
            raise ValueError("Epsilon decay must be in (0, 1].")

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.rng = random.Random(seed)

        # Maps (observable state, test ID) to its estimated Q-value.
        self.q_table: dict[tuple[Any, str], float] = {}

    @staticmethod
    def encode_state(state: dict[str, Any]) -> tuple[Any, ...]:
        """Convert the observable environment state into a hashable key."""
        return (
            state["remaining_budget"],
            tuple(sorted(state["executed_tests"])),
            tuple(sorted(state["discovered_bugs"])),
            tuple(sorted(state["covered_features"])),
        )

    def get_q_value(
        self, state: dict[str, Any], action: str
    ) -> float:
        """Return the learned value, or zero for an unseen state-action pair."""
        key = (self.encode_state(state), action)
        return self.q_table.get(key, 0.0)

    def select_action(
        self,
        state: dict[str, Any],
        legal_actions: list[str],
        explore: bool = True,
    ) -> str:
        """Choose a legal action using epsilon-greedy selection."""
        if not legal_actions:
            raise ValueError("Cannot select an action when none are legal.")

        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        q_values = [
            self.get_q_value(state, action)
            for action in legal_actions
        ]
        best_value = max(q_values)

        best_actions = [
            action
            for action, value in zip(legal_actions, q_values)
            if value == best_value
        ]

        return self.rng.choice(best_actions)

    def update(
        self,
        state: dict[str, Any],
        action: str,
        reward: float,
        next_state: dict[str, Any],
        next_legal_actions: list[str],
        done: bool,
    ) -> None:
        """Apply one Q-learning update."""
        current_q = self.get_q_value(state, action)

        if done or not next_legal_actions:
            future_q = 0.0
        else:
            future_q = max(
                self.get_q_value(next_state, next_action)
                for next_action in next_legal_actions
            )

        target = reward + self.discount_factor * future_q

        updated_q = current_q + self.learning_rate * (
            target - current_q
        )

        key = (self.encode_state(state), action)
        self.q_table[key] = updated_q

    def decay_epsilon(self) -> None:
        """Reduce exploration gradually without going below the minimum."""
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )
