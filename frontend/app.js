
const API_BASE = "http://127.0.0.1:8000";

const STRATEGIES = [
  {
    key: "q_learning",
    label: "Q-learning",
    color: "#9a83ff"
  },
  {
    key: "random",
    label: "Random",
    color: "#60a5fa"
  },
  {
    key: "fixed_order",
    label: "Fixed-order",
    color: "#42d9a2"
  }
];

const METRICS = {
  bugs_found: {
    label: "Bugs found",
    canvas: "bugsChart"
  },
  features_covered: {
    label: "Features covered",
    canvas: "coverageChart"
  },
  total_reward: {
    label: "Total reward",
    canvas: "rewardChart"
  },
  tests_executed: {
    label: "Tests executed",
    canvas: "testsChart"
  }
};

const charts = {};
let toastTimer = null;

const $ = (id) => document.getElementById(id);

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("visible");

  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove("visible");
  }, 3500);
}

function setConnection(isOnline) {
  const dot = $("connectionDot");
  dot.classList.toggle("online", isOnline);
  dot.classList.toggle("offline", !isOnline);
  $("connectionText").textContent = isOnline
    ? "API connected"
    : "API disconnected";
}

function setProgress(percent, message) {
  $("progressBar").style.width = `${percent}%`;
  $("progressText").textContent = message;
}

function setRunning(isRunning) {
  $("runButton").disabled = isRunning;
  $("refreshButton").disabled = isRunning;

  $("runButtonText").textContent = isRunning
    ? "Running experiment..."
    : "▶ Run experiment";

  $("runStatus").textContent = isRunning
    ? "The backend is executing your experiment."
    : "Ready to run an experiment.";

  $("runStatusIcon").textContent = isRunning ? "◌" : "●";
}

function showError(message) {
  const box = $("errorBox");
  box.textContent = message;
  box.hidden = false;
}

function clearError() {
  const box = $("errorBox");
  box.textContent = "";
  box.hidden = true;
}

function formatNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }

  return Number.isInteger(value)
    ? String(value)
    : value.toFixed(2);
}

function getStrategyRows(summary) {
  return STRATEGIES.map((strategy) => ({
    ...strategy,
    metrics: summary?.[strategy.key] ?? {}
  }));
}

function updateMetricCards(summary) {
  const qLearning = summary?.q_learning;

  if (!qLearning) {
    $("bugsMetric").textContent = "—";
    $("coverageMetric").textContent = "—";
    $("testsMetric").textContent = "—";
    $("rewardMetric").textContent = "—";
    return;
  }

  $("bugsMetric").textContent = formatNumber(qLearning.bugs_found);
  $("coverageMetric").textContent = formatNumber(qLearning.features_covered);
  $("testsMetric").textContent = formatNumber(qLearning.tests_executed);
  $("rewardMetric").textContent = formatNumber(qLearning.total_reward);
}

function updateResultsTable(summary) {
  const tbody = $("resultsTable");
  const rows = getStrategyRows(summary);

  if (!rows.some((row) => Object.keys(row.metrics).length > 0)) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" class="empty-cell">
          No evaluation results are available yet.
        </td>
      </tr>
    `;
    return;
  }

  const columns = [
    "bugs_found",
    "features_covered",
    "tests_executed",
    "budget_used",
    "total_reward"
  ];

  tbody.replaceChildren();

  rows.forEach((row) => {
    const tr = document.createElement("tr");

    const nameCell = document.createElement("td");
    const nameWrapper = document.createElement("div");
    nameWrapper.className = "strategy-name";

    const marker = document.createElement("span");
    marker.className = "strategy-marker";
    marker.style.backgroundColor = row.color;

    const name = document.createElement("span");
    name.textContent = row.label;

    nameWrapper.append(marker, name);
    nameCell.appendChild(nameWrapper);
    tr.appendChild(nameCell);

    columns.forEach((column) => {
      const cell = document.createElement("td");
      cell.textContent = formatNumber(row.metrics[column]);
      tr.appendChild(cell);
    });

    tbody.appendChild(tr);
  });
}

function createOrUpdateChart(metricKey, summary) {
  const metric = METRICS[metricKey];
  const canvas = $(metric.canvas);

  const rows = getStrategyRows(summary);
  const values = rows.map((row) => {
    const value = row.metrics[metricKey];
    return typeof value === "number" && Number.isFinite(value)
      ? value
      : null;
  });

  if (charts[metricKey]) {
    charts[metricKey].destroy();
    delete charts[metricKey];
  }

  charts[metricKey] = new Chart(canvas, {
    type: "bar",
    data: {
      labels: rows.map((row) => row.label),
      datasets: [{
        label: metric.label,
        data: values,
        backgroundColor: rows.map((row) => row.color + "cc"),
        borderColor: rows.map((row) => row.color),
        borderWidth: 1.5,
        borderRadius: 7,
        maxBarThickness: 65
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 350
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          backgroundColor: "#1b2640",
          titleColor: "#edf2ff",
          bodyColor: "#c9d4ea",
          borderColor: "#35425e",
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label(context) {
              return ` ${metric.label}: ${formatNumber(context.parsed.y)}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            display: false
          },
          ticks: {
            color: "#a0aec7",
            font: {
              family: "DM Sans",
              size: 11
            }
          },
          border: {
            color: "#34415b"
          }
        },
        y: {
          beginAtZero: true,
          grid: {
            color: "#28344a"
          },
          ticks: {
            color: "#8392ad",
            precision: 2
          },
          border: {
            display: false
          }
        }
      }
    }
  });
}

function updateCharts(summary) {
  Object.keys(METRICS).forEach((metricKey) => {
    createOrUpdateChart(metricKey, summary);
  });
}

function displayResults(summary, settings = null) {
  if (!summary || typeof summary !== "object") {
    throw new Error("The API returned an unexpected results format.");
  }

  updateMetricCards(summary);
  updateResultsTable(summary);
  updateCharts(summary);

  $("resultLabel").textContent = "Results loaded";

  if (settings) {
    const evalCount = settings.evaluation_scenarios;
    $("progressText").textContent =
      typeof evalCount === "number"
        ? `Evaluation scenarios: ${evalCount} · Seed: ${settings.seed ?? "—"}`
        : "Latest saved evaluation results";
  }

  setConnection(true);
}

async function fetchJson(url, options = {}) {
  let response;

  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new Error(
      "Could not connect to the API. Make sure FastAPI is running at " +
      API_BASE + " and try again."
    );
  }

  const text = await response.text();
  let data;

  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error("The server returned a response that was not valid JSON.");
  }

  if (!response.ok) {
    const detail = data?.detail;
    const message = typeof detail === "string"
      ? detail
      : `Request failed with HTTP ${response.status}.`;

    throw new Error(message);
  }

  return data;
}

async function checkHealth() {
  try {
    await fetchJson(`${API_BASE}/health`);
    setConnection(true);
    return true;
  } catch {
    setConnection(false);
    return false;
  }
}

async function loadResults({ silent = false } = {}) {
  clearError();

  if (!silent) {
    $("resultLabel").textContent = "Loading...";
  }

  try {
    const data = await fetchJson(`${API_BASE}/api/results`);

    displayResults(data.summary, data.settings);
    setConnection(true);

    if (!silent) {
      showToast("Latest saved results loaded.");
    }
  } catch (error) {
    setConnection(false);
    $("resultLabel").textContent = "Results unavailable";

    if (!silent) {
      showError(error.message);
      showToast("Could not load saved results.");
    }
  }
}

function readExperimentSettings() {
  const settings = {
    training_scenarios: Number($("trainingScenarios").value),
    evaluation_scenarios: Number($("evaluationScenarios").value),
    episodes: Number($("episodes").value),
    initial_budget: Number($("budget").value),
    seed: Number($("seed").value)
  };

  const limits = {
    training_scenarios: [1, 100],
    evaluation_scenarios: [1, 500],
    episodes: [1, 5000],
    initial_budget: [0, 1000],
    seed: [0, 2147483647]
  };

  for (const [key, value] of Object.entries(settings)) {
    const [min, max] = limits[key];

    if (!Number.isSafeInteger(value) || value < min || value > max) {
      throw new Error(
        `${key.replaceAll("_", " ")} must be a whole number between ${min} and ${max}.`
      );
    }
  }

  return settings;
}

async function runExperiment(event) {
  event.preventDefault();
  clearError();

  let settings;

  try {
    settings = readExperimentSettings();
  } catch (error) {
    showError(error.message);
    return;
  }

  setRunning(true);
  $("resultLabel").textContent = "Experiment running";
  setProgress(15, "Sending experiment settings to FastAPI...");
  showToast("Experiment started.");

  try {
    setProgress(40, "Training the agent and evaluating the strategies...");

    const data = await fetchJson(`${API_BASE}/api/experiments/run`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify(settings)
    });

    if (data.status !== "completed" || !data.summary) {
      throw new Error("The API did not return a completed experiment.");
    }

    setProgress(85, "Updating charts and evaluation metrics...");

    displayResults(data.summary, data.settings);
    $("resultLabel").textContent = "Experiment completed";

    setProgress(
      100,
      `Completed · ${settings.evaluation_scenarios} evaluation scenarios · Seed ${settings.seed}`
    );

    $("runStatus").textContent = "Experiment completed successfully.";
    $("runStatusIcon").textContent = "✓";

    showToast("Experiment completed successfully.");
  } catch (error) {
    $("resultLabel").textContent = "Experiment failed";
    $("runStatus").textContent = "The experiment could not be completed.";
    $("runStatusIcon").textContent = "!";
    setProgress(0, "Experiment failed");

    showError(error.message);
    showToast("Experiment failed. Check the error message.");
  } finally {
    setRunning(false);
  }
}

$("experimentForm").addEventListener("submit", runExperiment);

$("refreshButton").addEventListener("click", async () => {
  const online = await checkHealth();

  if (online) {
    await loadResults();
  } else {
    showError(
      "FastAPI is not responding. Start the backend and click Refresh results again."
    );
  }
});

async function initializeDashboard() {
  Object.keys(METRICS).forEach((metricKey) => {
    const canvas = $(METRICS[metricKey].canvas);

    // Draw empty chart axes until saved results load.
    charts[metricKey] = new Chart(canvas, {
      type: "bar",
      data: {
        labels: STRATEGIES.map((strategy) => strategy.label),
        datasets: [{
          label: METRICS[metricKey].label,
          data: [0, 0, 0],
          backgroundColor: STRATEGIES.map((strategy) => strategy.color + "cc"),
          borderColor: STRATEGIES.map((strategy) => strategy.color),
          borderWidth: 1.5,
          borderRadius: 7
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: "#a0aec7" }
          },
          y: {
            beginAtZero: true,
            grid: { color: "#28344a" },
            ticks: { color: "#8392ad" }
          }
        }
      }
    });
  });

  const online = await checkHealth();

  if (online) {
    await loadResults({ silent: true });
  } else {
    $("resultLabel").textContent = "API offline";
    showError(
      "The dashboard loaded, but the API is unavailable. Start FastAPI with " +
      "'python -m uvicorn src.api:app --reload' from your project root, then refresh."
    );
  }
}

initializeDashboard();
