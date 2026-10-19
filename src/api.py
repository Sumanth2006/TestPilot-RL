
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.experiments import run_experiment


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "experiment.json"
RESULTS_DIR = BASE_DIR / "results"

app = FastAPI(
    title="TestPilot-RL API",
    description=(
        "API for running reinforcement-learning software-testing "
        "experiments and comparing them with baseline strategies."
    ),
    version="1.0.0",
)

# Allow the local frontend to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "null",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Prevent two experiments from running concurrently in this process.
experiment_lock = Lock()


class ExperimentRequest(BaseModel):
    training_scenarios: int = Field(default=30, ge=1, le=100)
    evaluation_scenarios: int = Field(default=50, ge=1, le=500)
    episodes: int = Field(default=500, ge=1, le=5000)
    initial_budget: int = Field(default=20, ge=0, le=1000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""

    return {
        "project": "TestPilot-RL",
        "message": "Software testing with reinforcement learning",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Check whether the API is running."""

    return {"status": "ok"}


@app.get("/api/results")
def get_results() -> dict[str, Any]:
    """Return the latest saved experiment summary, if available."""

    summary_path = RESULTS_DIR / "summary.json"

    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No experiment results found. Run an experiment first.",
        )

    import json

    try:
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not read the saved experiment summary.",
        ) from exc

    return summary


@app.post("/api/experiments/run")
def run_experiment_endpoint(
    request: ExperimentRequest,
) -> dict[str, Any]:
    """Run an experiment and return its actual evaluation results."""

    if not experiment_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="An experiment is already running. Please try again later.",
        )

    try:
        result = run_experiment(
            training_scenarios=request.training_scenarios,
            evaluation_scenarios=request.evaluation_scenarios,
            episodes=request.episodes,
            initial_budget=request.initial_budget,
            seed=request.seed,
            output_dir=str(RESULTS_DIR),
            config_path=str(CONFIG_PATH),
        )

        return {
            "status": "completed",
            "settings": {
                "training_scenarios": request.training_scenarios,
                "evaluation_scenarios": request.evaluation_scenarios,
                "episodes": request.episodes,
                "initial_budget": request.initial_budget,
                "seed": request.seed,
            },
            "summary": result["summary"],
            "files": result["files"],
        }

    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Experiment failed: {exc}",
        ) from exc

    finally:
        experiment_lock.release()
