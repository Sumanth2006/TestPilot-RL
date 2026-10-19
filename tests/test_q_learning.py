
import pytest

from src.q_learning import QLearningAgent


@pytest.fixture
def state():
    return {
        "remaining_budget": 10,
        "executed_tests": frozenset({"T1"}),
        "discovered_bugs": frozenset({"B1"}),
        "covered_features": frozenset({"F1", "F2"}),
    }


def test_agent_initializes_with_empty_q_table():
    agent = QLearningAgent(seed=42)

    assert agent.q_table == {}
    assert agent.epsilon == 1.0


def test_action_selection_returns_only_a_legal_action(state):
    agent = QLearningAgent(epsilon=1.0, seed=42)
    legal_actions = ["T2", "T3", "T4"]

    for _ in range(20):
        action = agent.select_action(state, legal_actions)

        assert action in legal_actions


def test_exploitation_selects_highest_q_value(state):
    agent = QLearningAgent(epsilon=0.0, seed=42)

    agent.q_table[(agent.encode_state(state), "T2")] = 2.0
    agent.q_table[(agent.encode_state(state), "T3")] = 9.0
    agent.q_table[(agent.encode_state(state), "T4")] = 4.0

    action = agent.select_action(
        state, ["T2", "T3", "T4"], explore=False
    )

    assert action == "T3"


def test_select_action_rejects_empty_legal_actions(state):
    agent = QLearningAgent(seed=42)

    with pytest.raises(ValueError):
        agent.select_action(state, [])


def test_q_learning_update_uses_learning_equation(state):
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.0,
        seed=42,
    )

    next_state = {
        "remaining_budget": 5,
        "executed_tests": frozenset({"T1", "T2"}),
        "discovered_bugs": frozenset({"B1", "B2"}),
        "covered_features": frozenset({"F1", "F2", "F3"}),
    }

    agent.q_table[(agent.encode_state(state), "T2")] = 4.0
    agent.q_table[(agent.encode_state(next_state), "T3")] = 8.0

    agent.update(
        state=state,
        action="T2",
        reward=10.0,
        next_state=next_state,
        next_legal_actions=["T3"],
        done=False,
    )

    # Target = 10 + 0.9 * 8 = 17.2
    # New Q = 4 + 0.1 * (17.2 - 4) = 5.32
    assert agent.get_q_value(state, "T2") == pytest.approx(5.32)


def test_terminal_update_does_not_use_future_q_value(state):
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.0,
        seed=42,
    )

    next_state = {
        "remaining_budget": 0,
        "executed_tests": frozenset({"T1", "T2"}),
        "discovered_bugs": frozenset({"B1"}),
        "covered_features": frozenset({"F1", "F2"}),
    }

    agent.q_table[(agent.encode_state(state), "T2")] = 4.0
    agent.q_table[(agent.encode_state(next_state), "T3")] = 100.0

    agent.update(
        state=state,
        action="T2",
        reward=10.0,
        next_state=next_state,
        next_legal_actions=["T3"],
        done=True,
    )

    # Terminal target = reward = 10
    # New Q = 4 + 0.1 * (10 - 4) = 4.6
    assert agent.get_q_value(state, "T2") == pytest.approx(4.6)


def test_epsilon_decay_respects_minimum():
    agent = QLearningAgent(
        epsilon=1.0,
        min_epsilon=0.1,
        epsilon_decay=0.5,
        seed=42,
    )

    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.5)

    for _ in range(20):
        agent.decay_epsilon()

    assert agent.epsilon == pytest.approx(0.1)


def test_invalid_hyperparameters_are_rejected():
    with pytest.raises(ValueError):
        QLearningAgent(learning_rate=0.0)

    with pytest.raises(ValueError):
        QLearningAgent(discount_factor=1.5)

    with pytest.raises(ValueError):
        QLearningAgent(epsilon=0.2, min_epsilon=0.5)

    with pytest.raises(ValueError):
        QLearningAgent(epsilon_decay=0.0)
