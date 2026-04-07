---
title: ExecuTorch Export Repair Gym
emoji: "T"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - executorch
  - edge-ai
  - pytorch
  - model-export
---

# ExecuTorch Export Repair Gym

An OpenEnv environment where an agent acts like an edge deployment engineer and fixes tiny PyTorch modules so they become edge-deployable with ExecuTorch without changing their behavior.

## Why this is a strong hackathon environment

- Real-world utility: teams really do hit export failures when moving PyTorch models to edge runtimes, and fixing those failures is a real engineering workflow.
- Verifiable grading: every task is scored by hidden parity checks, `torch.export.export(...)`, and ExecuTorch lowering when the package is available.
- Multi-step repair loop: agents inspect the broken model, choose repair patches, validate the fix, and only then submit.
- Novelty: this is not a generic coding env or a plain quantization tuner; it focuses on a specific and underrepresented edge deployment repair task.

## What the agent is actually learning

This environment is about repair, not generic benchmarking.

The agent is trained or evaluated on the workflow an edge engineer actually follows:

1. inspect a model that currently fails edge export
2. identify the export-hostile pattern
3. apply a focused repair
4. validate that behavior is preserved
5. confirm the model is now exportable and lowerable

The benchmark is only the scoring layer around that real repair task.

## What the environment teaches

The tasks mirror common export repair work:

1. Remove data-dependent Python control flow (`.item()` inside `forward`)
2. Remove NumPy escape hatches from the forward graph
3. Rewrite harder export and lowering blockers using edge-friendly tensor primitives

## Action space

`ExicutorchAction`

- `inspect_task`
- `inspect_source`
- `inspect_fixes`
- `inspect_report`
- `apply_patch` with a `patch_id`
- `run_checks`
- `submit_final`

The environment is deliberately structured: instead of arbitrary code execution, the agent works with realistic repair candidates. That keeps the environment deterministic, learnable, and safe on HF Spaces while still modeling a real debugging workflow.

## Observation space

Each observation includes:

- task metadata and difficulty
- current source code
- patch catalog by slot
- current slot selections
- export success / failure details
- parity score against hidden reference behavior
- operator compatibility score from the exported graph
- lowering success and `.pte` buffer size (when ExecuTorch is available)
- current score, best score, and success threshold

## Reward design

`run_checks` and `submit_final` return a score in `[0, 1]` built from:

- hidden-behavior parity
- `torch.export` success
- supported operator coverage in the exported graph
- ExecuTorch lowering success (when available)
- minimal-edit bonus

Inspection actions give tiny one-time shaping rewards so agents are encouraged to inspect before editing, which mirrors a sensible repair workflow instead of random patching.

## Task progression

### Task 1: Control Flow Guard Cleanup (easy)
Fix a `.item()`-based Python branch using tensorized control flow.

### Task 2: NumPy Escape Hatch Removal (medium)
Remove a NumPy round-trip while preserving the exact preprocessing behavior.

### Task 3: Two-Stage Edge Score Block Repair (hard)
Repair both preprocessing and scoring logic so the module becomes edge-export-safe end to end while preserving hidden behavior.

## Local development

```bash
cd C:\Users\soumy\OneDrive\Desktop\pytorch-gym\exicutorch_env
uv sync --python 3.11
openenv validate
uv run pytest
```

Run the server:

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Open the UI at:

```text
http://localhost:7860/web
```

## Docker

The root `Dockerfile` is the one intended for HF Spaces and local validation.

```bash
docker build -t executorch-export-repair .
docker run -p 7860:7860 executorch-export-repair
```

## Inference

The required root-level `inference.py` uses the OpenAI client with:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

It emits strict `[START]`, `[STEP]`, and `[END]` logs for evaluator compatibility.

## Validation checklist

Before submitting:

```bash
openenv validate
uv run pytest
```

And run the external pre-submission validator against the deployed Space URL.
