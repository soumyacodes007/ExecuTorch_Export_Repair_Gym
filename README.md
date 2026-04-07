---
title: ExecuTorch Export Repair Gym
emoji: "🛠️"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - executorch
  - edge-ai
  - pytorch
  - model-export
  - reinforcement-learning
---

# ExecuTorch Export Repair Gym

🚀 **Live Space:** [https://huggingface.co/spaces/rick00767/executorch_export_repair_gym](https://huggingface.co/spaces/rick00767/executorch_export_repair_gym)

An OpenEnv environment for training and evaluating agents that **repair small PyTorch models so they can be exported for edge deployment with ExecuTorch** without breaking numerical behavior.

This environment models a real deployment workflow: a model fails edge export, the agent inspects the failure, applies a targeted repair, validates parity, and submits the repaired version.

## Why this environment is useful

- **Real-world utility**: Edge and mobile teams regularly hit `torch.export` and lowering failures when moving PyTorch models into deployable runtimes.
- **Agent-relevant task**: This is a practical repair workflow that coding and ML systems agents should learn.
- **Deterministic grading**: Success is measured with export checks, hidden parity checks, and lowering validation.
- **Fast and lightweight**: Tasks use tiny CPU-friendly models and run comfortably on modest hardware.
- **Novel problem setting**: It stays in the edge deployment domain instead of repeating common merge-conflict or SQL-cleaning tasks.

## What the agent is actually doing

This is a **repair environment**, not a generic benchmark.

For each episode, the agent:

1. inspects a broken model or task context,
2. identifies export-hostile logic,
3. applies a focused repair patch,
4. runs validation checks,
5. submits the final repaired model once it is edge-deployable.

The goal is not to rewrite arbitrary code. The goal is to make a broken model export cleanly while preserving behavior.

## Task curriculum

The environment includes three deterministic tasks with increasing difficulty.

### 1. `control_flow_guard` — Easy

Repairs a model that uses Python-side scalar control flow in a way that breaks export.  
The correct fix replaces the branch with tensor-friendly logic.

### 2. `numpy_escape_hatch` — Medium

Repairs a model that leaves the PyTorch graph through a NumPy round-trip.  
The correct fix keeps the computation inside PyTorch so export remains valid.

### 3. `edge_score_block` — Hard

Repairs a two-stage pipeline with issues in multiple code slots.  
This task requires sequential reasoning: partial fixes improve the score, but only a full repair reaches success.

## Action space

The action space is intentionally constrained to keep the environment stable and learnable.

Supported actions include:

- `inspect_source`
- `apply_patch`
- `run_checks`
- `submit_final`

`apply_patch` uses a task-specific repair catalog instead of unrestricted code generation. This keeps the environment deterministic while still supporting meaningful multi-step reasoning.

## Observation space

Observations expose the information an agent needs to work like an edge deployment engineer:

- task metadata
- current source
- source preview for UI readability
- patch catalog by repair slot
- repair summary
- recommended next action
- parity score
- export success
- lowering status
- current score

The built-in OpenEnv web interface at `/web` surfaces these fields for interactive debugging and demos.

## Grading and reward design

The environment returns scores in `[0, 1]`.

The grader rewards the behaviors that matter in a real repair workflow:

- preserving numerical parity on hidden inputs
- making `torch.export` succeed
- lowering cleanly for ExecuTorch when available
- completing the repair efficiently

Reward shaping is intentionally lightweight:

- a small one-time bonus encourages early inspection,
- `run_checks` provides dense feedback after repairs,
- `submit_final` returns the final repair score.

This means agents can learn from intermediate progress instead of receiving only sparse end-of-episode signals.

## Project structure

```text
.
├── __init__.py
├── client.py
├── models.py
├── tasks.py
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── README.md
├── server/
│   ├── __init__.py
│   ├── app.py
│   └── executorch_env_environment.py
└── tests/
    └── test_environment.py
```

## Local development

Install dependencies:

```bash
uv sync --frozen --no-dev
```

Run the environment locally:

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Then open:

- API: `http://localhost:7860`
- Web UI: `http://localhost:7860/web`

## Running the inference agent

The repository includes a root-level `inference.py` that follows the hackathon requirements:

- uses the OpenAI client,
- reads `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME`,
- emits strict `[START]`, `[STEP]`, and `[END]` logs,
- reports normalized scores in `[0, 1]`.

Example local run:

```bash
export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

## Docker

Build the image:

```bash
docker build -t executorch-repair-gym .
```

Run it:

```bash
docker run -p 7860:7860 executorch-repair-gym
```

Then visit:

- `http://localhost:7860/health`
- `http://localhost:7860/web`

## Hugging Face Spaces

This project is designed for Docker-based deployment on Hugging Face Spaces.

Recommended Space secrets:

- `HF_TOKEN`
- `API_BASE_URL`
- `MODEL_NAME`

Suggested values:

```text
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

For Spaces, you do **not** need `ENV_BASE_URL` or `LOCAL_IMAGE_NAME`.

## Validation checklist

Before submission, make sure:

- `openenv validate` passes
- the Docker image builds successfully
- `/web` loads correctly
- `python inference.py` runs end-to-end
- the inference logs keep the exact required format

## Why it stands out

ExecuTorch Export Repair Gym is built around a task that is narrow enough to verify rigorously, broad enough to matter in practice, and distinctive enough to stand out in a crowded field. It gives agents a realistic edge-deployment repair loop with clean state, meaningful rewards, and a clear path from partial progress to full success.
