from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

import torch
from openenv.core.env_server.interfaces import Environment

try:
    from exicutorch_env.models import ExecutorchAction, ExecutorchObservation, ExecutorchState
    from exicutorch_env.tasks import DEFAULT_TASK_ID, REPAIR_TASKS, RepairTask, correct_source, patch_catalog, render_source
except ModuleNotFoundError:
    from models import ExecutorchAction, ExecutorchObservation, ExecutorchState  # type: ignore[import-not-found]
    from tasks import DEFAULT_TASK_ID, REPAIR_TASKS, RepairTask, correct_source, patch_catalog, render_source  # type: ignore[import-not-found]

SUPPORTED_EDGE_OPS = (
    'aten.add',
    'aten.sub',
    'aten.mul',
    'aten.relu',
    'aten.mean',
    'aten.sum',
    'aten.sqrt',
    'aten.abs',
    'aten.where',
    'aten.gt',
    'aten.ge',
    'aten.clone',
    'aten.view',
    'aten.unsqueeze',
    'aten.squeeze',
    'aten.expand',
    'aten.scalar_tensor',
    'aten.full_like',
)

POSSIBLE_ACTIONS = [
    'inspect_task',
    'inspect_source',
    'inspect_fixes',
    'inspect_report',
    'apply_patch',
    'run_checks',
    'submit_final',
]


class ExecutorchEnvironment(Environment[ExecutorchAction, ExecutorchObservation, ExecutorchState]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._task = REPAIR_TASKS[DEFAULT_TASK_ID]
        self._state = ExecutorchState(episode_id=str(uuid4()), step_count=0)
        self._reset_to_task(self._task)

    def reset(self, seed: int | None = None, task_id: str | None = None, **kwargs: Any) -> ExecutorchObservation:
        del seed
        requested_task_id = task_id or kwargs.get('task_id') or DEFAULT_TASK_ID
        if requested_task_id not in REPAIR_TASKS:
            raise ValueError(f'Unknown task_id: {requested_task_id}')
        self._task = REPAIR_TASKS[requested_task_id]
        self._state = ExecutorchState(episode_id=str(uuid4()), step_count=0)
        self._reset_to_task(self._task)
        return self._make_observation(
            message='Environment reset. Inspect the broken source, apply focused fixes, run checks, then submit.',
            error='',
            reward=0.0,
            done=False,
        )

    def step(self, action: ExecutorchAction) -> ExecutorchObservation:
        if not isinstance(action, ExecutorchAction):
            raise TypeError(f'Expected ExecutorchAction, got {type(action)}')

        self._state.step_count += 1
        reward = 0.0
        done = False
        error = ''
        message = ''
        report = self._state.last_report or None

        try:
            if action.action_type == 'inspect_task':
                reward = self._mark_inspection('inspect_task')
                message = self._task.description
            elif action.action_type == 'inspect_source':
                reward = self._mark_inspection('inspect_source')
                message = 'Current broken source returned for inspection.'
            elif action.action_type == 'inspect_fixes':
                reward = self._mark_inspection(f"inspect_fixes:{action.slot_name or 'all'}")
                if action.slot_name and action.slot_name not in self._task.slot_order:
                    raise ValueError(f'Unknown slot_name: {action.slot_name}')
                target = action.slot_name or 'all slots'
                message = f'Repair catalog returned for {target}.'
            elif action.action_type == 'inspect_report':
                reward = self._mark_inspection('inspect_report')
                message = 'Latest validation report returned.' if self._state.last_report else 'No validation report yet. Run checks first.'
            elif action.action_type == 'apply_patch':
                slot_name, patch_id = self._resolve_patch(action.patch_id)
                self._state.selected_patches[slot_name] = patch_id
                self._state.patch_history.append(patch_id)
                self._state.current_source = render_source(self._task, self._state.selected_patches)
                message = f'Applied repair patch {patch_id} to slot {slot_name}.'
            elif action.action_type == 'run_checks':
                report = self._evaluate_current_source()
                self._state.last_report = report
                self._state.checks_run += 1
                self._state.best_score = max(self._state.best_score, report['current_score'])
                reward = report['current_score']
                message = self._summarize_report(report)
            elif action.action_type == 'submit_final':
                report = self._state.last_report or self._evaluate_current_source()
                self._state.last_report = report
                self._state.best_score = max(self._state.best_score, report['current_score'])
                self._state.submitted = True
                reward = report['current_score']
                done = True
                message = 'Submitted final repaired model for scoring.'
            else:
                raise ValueError(f'Unsupported action_type: {action.action_type}')
        except Exception as exc:
            error = str(exc)
            message = 'Action failed. Review the error and try a safer repair.'

        if self._state.step_count >= self._task.max_steps and not done:
            report = self._state.last_report or self._evaluate_current_source()
            self._state.last_report = report
            self._state.best_score = max(self._state.best_score, report['current_score'])
            reward = max(reward, report['current_score'])
            done = True
            message = (message + ' ' if message else '') + 'Episode limit reached; auto-submitting current repair.'

        final_score = float((report or self._state.last_report or {}).get('current_score', 0.0))
        return self._make_observation(
            message=message,
            error=error,
            reward=reward,
            done=done,
            final_score=final_score if done else 0.0,
        )

    @property
    def state(self) -> ExecutorchState:
        return self._state

    def _reset_to_task(self, task: RepairTask) -> None:
        self._state.task_id = task.task_id
        self._state.task_title = task.title
        self._state.difficulty = task.difficulty
        self._state.selected_patches = dict(task.default_patch_ids)
        self._state.patch_history = []
        self._state.current_source = render_source(task, self._state.selected_patches)
        self._state.checks_run = 0
        self._state.best_score = 0.0
        self._state.submitted = False
        self._state.last_report = {}
        self._state.seen_inspections = []

    def _mark_inspection(self, key: str) -> float:
        if key not in self._state.seen_inspections:
            self._state.seen_inspections.append(key)
            return 0.02
        return 0.0

    def _resolve_patch(self, patch_id: str) -> tuple[str, str]:
        if not patch_id:
            raise ValueError('patch_id is required for apply_patch')
        for slot_name, options in self._task.patches_by_slot.items():
            for option in options:
                if option.patch_id == patch_id:
                    return slot_name, patch_id
        raise ValueError(f'Unknown patch_id: {patch_id}')

    def _evaluate_current_source(self) -> Dict[str, Any]:
        task = self._task
        current_source = self._state.current_source
        report: Dict[str, Any] = {
            'parity_score': 0.0,
            'export_success': False,
            'export_error': '',
            'compile_error': '',
            'runtime_error': '',
            'lowering_available': False,
            'lowering_success': False,
            'lowering_error': '',
            'buffer_size_kb': 0.0,
            'operator_compatibility': 0.0,
            'unsupported_ops': [],
            'current_score': 0.0,
            'is_success': False,
        }

        try:
            candidate = self._instantiate_module(current_source, task.class_name)
            reference = self._instantiate_module(correct_source(task), task.class_name)
        except Exception as exc:
            report['compile_error'] = str(exc)
            return report

        parity_score = self._parity_score(candidate, reference, task, report)
        report['parity_score'] = parity_score

        exported_program = None
        try:
            export_inputs = tuple(t.clone() for t in task.example_inputs)
            exported_program = torch.export.export(candidate, export_inputs)
            report['export_success'] = True
        except Exception as exc:
            report['export_error'] = str(exc)

        if exported_program is not None:
            op_names = self._extract_ops(exported_program)
            unsupported = sorted({name for name in op_names if not self._is_supported_edge_op(name)})
            report['operator_compatibility'] = (len(op_names) - len(unsupported)) / len(op_names) if op_names else 1.0
            report['unsupported_ops'] = unsupported[:10]
            self._attempt_lowering(exported_program, report)

        edit_budget = len(task.correct_patch_ids)
        extra_edits = max(0, len(self._state.patch_history) - edit_budget)
        minimal_edit_score = max(0.0, 1.0 - 0.15 * extra_edits)

        weights = {
            'parity': 0.45,
            'export': 0.25,
            'operators': 0.15,
            'edits': 0.10,
            'lowering': 0.05,
        }
        if not report['lowering_available']:
            weights.pop('lowering')
        weight_total = sum(weights.values())
        weighted_sum = (
            weights.get('parity', 0.0) * parity_score
            + weights.get('export', 0.0) * float(report['export_success'])
            + weights.get('operators', 0.0) * report['operator_compatibility']
            + weights.get('edits', 0.0) * minimal_edit_score
            + weights.get('lowering', 0.0) * float(report['lowering_success'])
        )
        report['current_score'] = round(weighted_sum / weight_total, 4) if weight_total else 0.0
        report['is_success'] = bool(
            parity_score >= 0.999
            and report['export_success']
            and report['operator_compatibility'] >= 0.999
            and (report['lowering_success'] or not report['lowering_available'])
            and report['current_score'] >= task.success_threshold
        )
        return report

    def _instantiate_module(self, source: str, class_name: str) -> torch.nn.Module:
        namespace: Dict[str, Any] = {}
        exec(source, namespace, namespace)
        model_cls = namespace[class_name]
        model = model_cls()
        model.eval()
        return model

    def _parity_score(
        self,
        candidate: torch.nn.Module,
        reference: torch.nn.Module,
        task: RepairTask,
        report: Dict[str, Any],
    ) -> float:
        passes = 0
        total = len(task.hidden_inputs)
        with torch.no_grad():
            for inputs in task.hidden_inputs:
                cloned_inputs = tuple(t.clone() for t in inputs)
                reference_inputs = tuple(t.clone() for t in inputs)
                try:
                    candidate_out = candidate(*cloned_inputs)
                    reference_out = reference(*reference_inputs)
                except Exception as exc:
                    report['runtime_error'] = str(exc)
                    return 0.0
                if torch.allclose(candidate_out, reference_out, atol=task.atol, rtol=task.rtol):
                    passes += 1
        return passes / total if total else 0.0

    def _attempt_lowering(self, exported_program: Any, report: Dict[str, Any]) -> None:
        try:
            from executorch.exir import to_edge_transform_and_lower  # type: ignore[import-not-found]
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner  # type: ignore[import-not-found]
        except Exception as exc:
            report['lowering_available'] = False
            report['lowering_error'] = f'ExecuTorch unavailable in current runtime: {exc}'
            return

        report['lowering_available'] = True
        try:
            lowered = to_edge_transform_and_lower(
                exported_program,
                partitioner=[XnnpackPartitioner()],
            ).to_executorch()
            report['lowering_success'] = True
            report['buffer_size_kb'] = round(len(lowered.buffer) / 1024.0, 3)
        except Exception as exc:
            report['lowering_error'] = str(exc)

    def _extract_ops(self, exported_program: Any) -> list[str]:
        names: list[str] = []
        try:
            graph = exported_program.graph_module.graph
            for node in graph.nodes:
                if node.op == 'call_function':
                    names.append(str(node.target))
        except Exception:
            pass
        return names

    def _is_supported_edge_op(self, op_name: str) -> bool:
        return any(keyword in op_name for keyword in SUPPORTED_EDGE_OPS)

    def _slot_status(self) -> dict[str, str]:
        status: dict[str, str] = {}
        for slot_name in self._task.slot_order:
            patch_id = self._state.selected_patches.get(slot_name, self._task.default_patch_ids[slot_name])
            option = next(opt for opt in self._task.patches_by_slot[slot_name] if opt.patch_id == patch_id)
            status[slot_name] = f'{option.patch_id}: {option.title}'
        return status

    def _summarize_report(self, report: Dict[str, Any]) -> str:
        summary = [
            f"parity={report['parity_score']:.2f}",
            f"export={str(report['export_success']).lower()}",
            f"ops={report['operator_compatibility']:.2f}",
        ]
        if report['lowering_available']:
            summary.append(f"lowering={str(report['lowering_success']).lower()}")
        if report['unsupported_ops']:
            summary.append(f"unsupported={', '.join(report['unsupported_ops'][:3])}")
        return 'Validation report: ' + ' | '.join(summary)

    def _make_observation(
        self,
        message: str,
        error: str,
        reward: float,
        done: bool,
        final_score: float = 0.0,
    ) -> ExecutorchObservation:
        report = self._state.last_report or {}
        return ExecutorchObservation(
            task_id=self._task.task_id,
            task_title=self._task.title,
            difficulty=self._task.difficulty,
            task_description=self._task.description,
            current_source=self._state.current_source,
            available_slots=list(self._task.slot_order),
            slot_status=self._slot_status(),
            patch_catalog=patch_catalog(self._task),
            supported_op_hints=list(self._task.supported_op_hints),
            export_success=bool(report.get('export_success', False)),
            lowering_success=bool(report.get('lowering_success', False)),
            lowering_available=bool(report.get('lowering_available', False)),
            parity_score=float(report.get('parity_score', 0.0)),
            operator_compatibility=float(report.get('operator_compatibility', 0.0)),
            buffer_size_kb=float(report.get('buffer_size_kb', 0.0)),
            unsupported_ops=list(report.get('unsupported_ops', [])),
            export_error=str(report.get('export_error', '')),
            lowering_error=str(report.get('lowering_error', '')),
            compile_error=str(report.get('compile_error', '')),
            runtime_error=str(report.get('runtime_error', '')),
            checks_run=self._state.checks_run,
            steps_taken=self._state.step_count,
            max_steps=self._task.max_steps,
            current_score=float(report.get('current_score', 0.0)),
            best_score=float(self._state.best_score),
            success_threshold=float(self._task.success_threshold),
            is_success=bool(report.get('is_success', False)),
            message=message,
            last_action_error=error,
            final_score=float(final_score),
            possible_actions=POSSIBLE_ACTIONS,
            last_report=report,
            reward=reward,
            done=done,
            metadata={
                'task_id': self._task.task_id,
                'checks_run': self._state.checks_run,
            },
        )
