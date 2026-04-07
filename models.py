from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class ExecutorchAction(Action):
    """Structured fixing action for the ExecuTorch Export Repair Gym."""

    action_type: str = Field(
        default='inspect_task',
        description=(
            'One of: inspect_task, inspect_source, inspect_fixes, inspect_report, '
            'apply_patch, run_checks, submit_final'
        ),
    )
    slot_name: str = Field(
        default='',
        description='Optional patch slot to inspect or modify (for example: body, preprocess, score).',
    )
    patch_id: str = Field(
        default='',
        description='Patch identifier to apply when action_type is apply_patch.',
    )
    rationale: str = Field(
        default='',
        description='Optional reasoning string for logging only. Never executed.',
    )


class ExecutorchObservation(Observation):
    """Observation returned after each interaction."""

    task_id: str = Field(default='')
    task_title: str = Field(default='')
    difficulty: str = Field(default='easy')
    task_description: str = Field(default='')
    current_source: str = Field(default='')
    available_slots: list[str] = Field(default_factory=list)
    slot_status: dict[str, str] = Field(default_factory=dict)
    patch_catalog: dict[str, list[dict[str, str]]] = Field(default_factory=dict)
    supported_op_hints: list[str] = Field(default_factory=list)
    export_success: bool = Field(default=False)
    lowering_success: bool = Field(default=False)
    lowering_available: bool = Field(default=False)
    parity_score: float = Field(default=0.0)
    operator_compatibility: float = Field(default=0.0)
    buffer_size_kb: float = Field(default=0.0)
    unsupported_ops: list[str] = Field(default_factory=list)
    export_error: str = Field(default='')
    lowering_error: str = Field(default='')
    compile_error: str = Field(default='')
    runtime_error: str = Field(default='')
    checks_run: int = Field(default=0)
    steps_taken: int = Field(default=0)
    max_steps: int = Field(default=0)
    current_score: float = Field(default=0.0)
    best_score: float = Field(default=0.0)
    success_threshold: float = Field(default=0.0)
    is_success: bool = Field(default=False)
    message: str = Field(default='')
    last_action_error: str = Field(default='')
    final_score: float = Field(default=0.0)
    possible_actions: list[str] = Field(default_factory=list)
    last_report: dict[str, Any] = Field(default_factory=dict)


class ExecutorchState(State):
    """Server-side episode state."""

    task_id: str = Field(default='')
    task_title: str = Field(default='')
    difficulty: str = Field(default='easy')
    selected_patches: dict[str, str] = Field(default_factory=dict)
    patch_history: list[str] = Field(default_factory=list)
    current_source: str = Field(default='')
    checks_run: int = Field(default=0)
    best_score: float = Field(default=0.0)
    submitted: bool = Field(default=False)
    last_report: dict[str, Any] = Field(default_factory=dict)
    seen_inspections: list[str] = Field(default_factory=list)
