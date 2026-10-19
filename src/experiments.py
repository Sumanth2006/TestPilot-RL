
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from src.baselines import (
    run_fixed_order_baseline,
    run_random_baseline,
)
from src.q_learning import QLearningAgent
from src.simulator import generate_scenario
from src.training import train_agent


def run_experiment(
    training_scenarios: int = 30,
    evaluation_scenarios: int = 20,
    episodes: int = 500,
    initial_budget: int = 20,
    seed: int = 42,
    output_dir: str = "results",
    config_path: str = "configs/experiment.json",
) -> dict[str, Any]:
    """Train Q-learning and compare it against two baselines."""

    if training_scenarios <= 0:
        raise ValueError("training_scenarios must be positive.")
    if evaluation_scenarios <= 0:
        raise ValueError("evaluation_scenarios must be positive.")
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if initial_budget < 0:
        raise ValueError("initial_budget cannot be negative.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read reward weights and Q-learning hyperparameters.
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    reward_config = config.get("reward", {})
    q_config = config.get("q_learning", {})

    bug_weight = float(reward_config.get("bug_weight", 10.0))
    coverage_weight = float(reward_config.get("coverage_weight", 2.0))
    cost_weight = float(reward_config.get("cost_weight", 1.0))

    # Use independent, reproducible seeds for each scenario.
    seed_rng = random.Random(seed)
    training_seeds = [
        seed_rng.randrange(1, 2**31)
        for _ in range(training_scenarios)
    ]
    evaluation_seeds = [
        seed_rng.randrange(1, 2**31)
        for _ in range(evaluation_scenarios)
    ]

    # Train on generated training scenarios only.
    agent = QLearningAgent(
        learning_rate=float(q_config.get("learning_rate", 0.1)),
        discount_factor=float(q_config.get("discount_factor", 0.9)),
        epsilon=float(q_config.get("initial_epsilon", 1.0)),
        min_epsilon=float(q_config.get("minimum_epsilon", 0.05)),
        epsilon_decay=float(q_config.get("epsilon_decay", 0.995)),
        seed=seed,
    )

    training_history: list[dict[str, Any]] = []

    for scenario_seed in training_seeds:
        scenario = generate_scenario(
            num_tests=int(config.get("num_tests", 20)),
            num_features=int(config.get("num_features", 10)),
            num_bugs=int(config.get("num_bugs", 10)),
            seed=scenario_seed,
        )

        history = train_agent(
            agent=agent,
            scenario=scenario,
            episodes=episodes,
            initial_budget=initial_budget,
            bug_weight=bug_weight,
            coverage_weight=coverage_weight,
            cost_weight=cost_weight,
        )

        training_history.extend(history)

    # Evaluate every strategy on the same held-out scenarios.
    evaluation_rows: list[dict[str, Any]] = []

    for scenario_index, scenario_seed in enumerate(
        evaluation_seeds, start=1
    ):
        scenario = generate_scenario(
            num_tests=int(config.get("num_tests", 20)),
            num_features=int(config.get("num_features", 10)),
            num_bugs=int(config.get("num_bugs", 10)),
            seed=scenario_seed,
        )

        # Greedy evaluation: no epsilon exploration.
        environment_results = _evaluate_q_learning(
            agent=agent,
            scenario=scenario,
            initial_budget=initial_budget,
            bug_weight=bug_weight,
            coverage_weight=coverage_weight,
            cost_weight=cost_weight,
        )

        random_results = run_random_baseline(
            scenario=scenario,
            initial_budget=initial_budget,
            bug_weight=bug_weight,
            coverage_weight=coverage_weight,
            cost_weight=cost_weight,
            seed=seed + scenario_index,
        )

        fixed_results = run_fixed_order_baseline(
            scenario=scenario,
            initial_budget=initial_budget,
            bug_weight=bug_weight,
            coverage_weight=coverage_weight,
            cost_weight=cost_weight,
        )

        for result in (
            environment_results,
            random_results,
            fixed_results,
        ):
            evaluation_rows.append(
                {
                    "scenario_id": scenario_index,
                    "scenario_seed": scenario_seed,
                    **result,
                }
            )

    # Aggregate results across held-out scenarios.
    strategies = ("q_learning", "random", "fixed_order")
    metric_names = (
        "bugs_found",
        "features_covered",
        "tests_executed",
        "budget_used",
        "total_reward",
    )

    summary: dict[str, Any] = {}

    for strategy in strategies:
        strategy_rows = [
            row for row in evaluation_rows
            if row["strategy"] == strategy
        ]

        summary[strategy] = {
            metric: round(
                mean(float(row[metric]) for row in strategy_rows), 3
            )
            for metric in metric_names
        }

    # Save raw evaluation data and aggregate summaries.
    csv_path = output_path / "evaluation_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "scenario_id",
            "scenario_seed",
            "strategy",
            "total_reward",
            "bugs_found",
            "features_covered",
            "tests_executed",
            "budget_used",
            "remaining_budget",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluation_rows)

    summary_path = output_path / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "settings": {
                    "seed": seed,
                    "training_scenarios": training_scenarios,
                    "evaluation_scenarios": evaluation_scenarios,
                    "episodes_per_training_scenario": episodes,
                    "initial_budget": initial_budget,
                },
                "summary": summary,
            },
            file,
            indent=2,
        )

    history_path = output_path / "training_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as file:
        if training_history:
            writer = csv.DictWriter(
                file, fieldnames=list(training_history[0].keys())
            )
            writer.writeheader()
            writer.writerows(training_history)

    return {
        "summary": summary,
        "evaluation_rows": evaluation_rows,
        "training_history": training_history,
        "files": {
            "evaluation": str(csv_path),
            "summary": str(summary_path),
            "training_history": str(history_path),
        },
    }


def _evaluate_q_learning(
    agent: QLearningAgent,
    scenario: dict[str, Any],
    initial_budget: int,
    bug_weight: float,
    coverage_weight: float,
    cost_weight: float,
) -> dict[str, Any]:
    """Evaluate the learned policy without exploration."""

    from src.environment import TestingEnvironment

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
        state = environment.get_state()
        legal_actions = environment.get_legal_actions()

        if not legal_actions:
            break

        action = agent.select_action(
            state=state,
            legal_actions=legal_actions,
            explore=False,
        )

        _, reward, done, _ = environment.step(action)
        total_reward += reward
        tests_executed += 1

        if done:
            break

    return {
        "strategy": "q_learning",
        "total_reward": total_reward,
        "bugs_found": len(environment.discovered_bugs),
        "features_covered": len(environment.covered_features),
        "tests_executed": tests_executed,
        "budget_used": initial_budget - environment.remaining_budget,
        "remaining_budget": environment.remaining_budget,
    }
