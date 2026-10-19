
import pytest

from src.q_learning import QLearningAgent
from src.simulator import generate_scenario
from src.training import train_agent


def test_training_returns_one_record_per_episode():
    scenario = generate_scenario(seed=42)
    agent = QLearningAgent(seed=42)

    history = train_agent(
        agent=agent,
        scenario=scenario,
        episodes=5,
        initial_budget=20,
    )

    assert len(history) == 5
    assert [row["episode"] for row in history] == [1, 2, 3, 4, 5]


def test_training_records_valid_episode_metrics():
    scenario = generate_scenario(seed=42)
    agent = QLearningAgent(seed=42)

    history = train_agent(
        agent=agent,
        scenario=scenario,
        episodes=3,
        initial_budget=20,
    )

    for row in history:
        assert isinstance(row["total_reward"], float)
        assert 0 <= row["bugs_found"] <= len(scenario["all_bugs"])
        assert 0 <= row["features_covered"] <= len(scenario["all_features"])
        assert row["tests_executed"] >= 0
        assert 0 <= row["remaining_budget"] <= 20
        assert 0 <= row["epsilon"] <= 1


def test_training_populates_q_table():
    scenario = generate_scenario(seed=42)
    agent = QLearningAgent(seed=42)

    train_agent(
        agent=agent,
        scenario=scenario,
        episodes=5,
        initial_budget=20,
    )

    assert agent.q_table


def test_training_rejects_nonpositive_episode_count():
    scenario = generate_scenario(seed=42)
    agent = QLearningAgent(seed=42)

    with pytest.raises(ValueError):
        train_agent(agent=agent, scenario=scenario, episodes=0)
