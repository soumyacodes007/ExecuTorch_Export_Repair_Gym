from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import ExecutorchAction, ExecutorchObservation, ExecutorchState


class ExecutorchEnv(EnvClient[ExecutorchAction, ExecutorchObservation, ExecutorchState]):
    """Client for the ExecuTorch Export Repair Gym."""

    def _step_payload(self, action: ExecutorchAction) -> Dict[str, Any]:
        return {
            'action_type': action.action_type,
            'slot_name': action.slot_name,
            'patch_id': action.patch_id,
            'rationale': action.rationale,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ExecutorchObservation]:
        obs_data = payload.get('observation', {})
        observation = ExecutorchObservation(
            task_id=obs_data.get('task_id', ''),
            task_title=obs_data.get('task_title', ''),
            difficulty=obs_data.get('difficulty', 'easy'),
            task_description=obs_data.get('task_description', ''),
            current_source=obs_data.get('current_source', ''),
            current_source_preview=obs_data.get('current_source_preview', ''),
            available_slots=obs_data.get('available_slots', []),
            slot_status=obs_data.get('slot_status', {}),
            patch_catalog=obs_data.get('patch_catalog', {}),
            supported_op_hints=obs_data.get('supported_op_hints', []),
            repair_summary=obs_data.get('repair_summary', ''),
            recommended_next_action=obs_data.get('recommended_next_action', ''),
            export_success=obs_data.get('export_success', False),
            lowering_success=obs_data.get('lowering_success', False),
            lowering_available=obs_data.get('lowering_available', False),
            parity_score=obs_data.get('parity_score', 0.0),
            operator_compatibility=obs_data.get('operator_compatibility', 0.0),
            buffer_size_kb=obs_data.get('buffer_size_kb', 0.0),
            unsupported_ops=obs_data.get('unsupported_ops', []),
            export_error=obs_data.get('export_error', ''),
            lowering_error=obs_data.get('lowering_error', ''),
            compile_error=obs_data.get('compile_error', ''),
            runtime_error=obs_data.get('runtime_error', ''),
            checks_run=obs_data.get('checks_run', 0),
            steps_taken=obs_data.get('steps_taken', 0),
            max_steps=obs_data.get('max_steps', 0),
            current_score=obs_data.get('current_score', 0.0),
            best_score=obs_data.get('best_score', 0.0),
            success_threshold=obs_data.get('success_threshold', 0.0),
            is_success=obs_data.get('is_success', False),
            message=obs_data.get('message', ''),
            last_action_error=obs_data.get('last_action_error', ''),
            final_score=obs_data.get('final_score', 0.0),
            possible_actions=obs_data.get('possible_actions', []),
            last_report=obs_data.get('last_report', {}),
            done=payload.get('done', False),
            reward=payload.get('reward'),
            metadata=obs_data.get('metadata', {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get('reward'),
            done=payload.get('done', False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ExecutorchState:
        return ExecutorchState(
            episode_id=payload.get('episode_id'),
            step_count=payload.get('step_count', 0),
            task_id=payload.get('task_id', ''),
            task_title=payload.get('task_title', ''),
            difficulty=payload.get('difficulty', 'easy'),
            selected_patches=payload.get('selected_patches', {}),
            patch_history=payload.get('patch_history', []),
            current_source=payload.get('current_source', ''),
            checks_run=payload.get('checks_run', 0),
            best_score=payload.get('best_score', 0.0),
            submitted=payload.get('submitted', False),
            last_report=payload.get('last_report', {}),
            seen_inspections=payload.get('seen_inspections', []),
        )
