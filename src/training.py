
from typing import Any

from src.environment import TestingEnvironment
from src.q_learning import QLearningAgent


def train_agent(
    agent: QLearningAgent,
    scenario: dict[str, Any],
    episodes: int = 500,
    initial_budget: int = 20,
    bug_weight: float = 10.0,
    coverage_weight: float = 2.0,
    cost_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Train a Q-learning agent and return per-episode metrics."""

    if episodes <= 0:
        raise ValueError("Episodes must be positive.")

    environment = TestingEnvironment(
        scenario=scenario,
        initial_budget=initial_budget,
        bug_weight=bug_weight,
        coverage_weight=coverage_weight,
        cost_weight=cost_weight,
    )

    history: list[dict[str, Any]] = []

    for episode in range(episodes):
        state = environment.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            legal_actions = environment.get_legal_actions()

            if not legal_actions:
                break

            action = agent.select_action(
                state=state,
                legal_actions=legal_actions,
            )

            next_state, reward, done, info = environment.step(action)

            next_legal_actions = (
                [] if done else environment.get_legal_actions()
            )

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_legal_actions=next_legal_actions,
                done=done,
            )

            state = next_state
            total_reward += reward
            steps += 1

        history.append(
            {
                "episode": episode + 1,
                "total_reward": total_reward,
                "bugs_found": len(environment.discovered_bugs),
                "features_covered": len(environment.covered_features),
                "tests_executed": steps,
                "remaining_budget": environment.remaining_budget,
                "epsilon": agent.epsilon,
            }
        )

        agent.decay_epsilon()

    return history
