import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from exicutorch_env import ExecutorchAction, ExecutorchEnv
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from exicutorch_env import ExecutorchAction, ExecutorchEnv

API_BASE_URL = os.getenv('API_BASE_URL', 'https://router.huggingface.co/v1')
MODEL_NAME = os.getenv('MODEL_NAME', 'Qwen/Qwen2.5-72B-Instruct')
HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('API_KEY')
ENV_BASE_URL = os.getenv('ENV_BASE_URL', 'http://localhost:7860')
LOCAL_IMAGE_NAME = os.getenv('LOCAL_IMAGE_NAME') or os.getenv('IMAGE_NAME')
BENCHMARK = 'executorch_export_repair_gym'
TASK_IDS = ['control_flow_guard', 'numpy_escape_hatch', 'edge_score_block']
MAX_STEPS_PER_TASK = 20   # hard ceiling; actual limit comes from observation.max_steps
SUCCESS_THRESHOLD = 0.90

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = textwrap.dedent(
    '''
    You are an edge deployment engineer fixing tiny PyTorch modules so they can be exported and lowered for ExecuTorch.

    Your job is to repair export-blocking model code while preserving behavior on hidden inputs.
    Prioritize these goals in order:
    1. Preserve behavior on hidden inputs (parity_score must be 1.0).
    2. Make torch.export succeed (export_success must be True).
    3. Improve operator compatibility and lower successfully for ExecuTorch.
    4. Avoid unnecessary edits (extra patches reduce your score).

    STRICT RULES — follow these exactly:
    - ALWAYS call run_checks immediately after applying a patch. Never apply two patches in a row.
    - NEVER apply the same patch_id twice — if it did not help, try a different approach.
    - If current_score >= success_threshold AND checks_run > 0, call submit_final immediately.
    - Use inspect actions only at the very start if you need context.
    - The correct workflow is: [inspect once if needed] → apply_patch → run_checks → apply_patch → run_checks → submit_final.

    Use the structured action schema only. Return exactly one valid JSON object with keys:
    {
      "action_type": "inspect_task|inspect_source|inspect_fixes|inspect_report|apply_patch|run_checks|submit_final",
      "slot_name": "optional slot name",
      "patch_id": "required only for apply_patch",
      "rationale": "brief optional note"
    }

    Do not return markdown. Do not wrap the JSON in code fences.
    '''
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f'[START] task={task} env={env} model={model}', flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else 'null'
    print(
        f'[STEP] step={step} action={action} reward={reward:.2f} '
        f'done={str(done).lower()} error={error_value}',
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    reward_blob = ','.join(f'{reward:.2f}' for reward in rewards)
    print(
        f'[END] success={str(success).lower()} steps={steps} '
        f'score={score:.2f} rewards={reward_blob}',
        flush=True,
    )


async def _connect_env() -> ExecutorchEnv:
    if LOCAL_IMAGE_NAME:
        return await ExecutorchEnv.from_docker_image(LOCAL_IMAGE_NAME)
    return ExecutorchEnv(base_url=ENV_BASE_URL)


def _build_user_prompt(observation: Any) -> str:
    report = observation.last_report or {}
    return textwrap.dedent(
        f'''
        Task: {observation.task_title} ({observation.difficulty})
        Description: {observation.task_description}

        Current source:
        {observation.current_source}

        Slot status:
        {json.dumps(observation.slot_status, indent=2)}

        Patch catalog:
        {json.dumps(observation.patch_catalog, indent=2)}

        Supported op hints:
        {json.dumps(observation.supported_op_hints, indent=2)}

        Latest metrics:
        - parity_score: {observation.parity_score:.2f}
        - export_success: {observation.export_success}
        - operator_compatibility: {observation.operator_compatibility:.2f}
        - lowering_available: {observation.lowering_available}
        - lowering_success: {observation.lowering_success}
        - current_score: {observation.current_score:.2f}
        - best_score: {observation.best_score:.2f}
        - checks_run: {observation.checks_run}
        - steps_taken: {observation.steps_taken}/{observation.max_steps}
        - last_message: {observation.message}
        - export_error: {observation.export_error or 'none'}
        - lowering_error: {observation.lowering_error or 'none'}
        - unsupported_ops: {json.dumps(observation.unsupported_ops)}

        Last report JSON:
        {json.dumps(report, indent=2)}

        Choose the next best action.
        '''
    ).strip()


def _call_llm(observation: Any) -> dict[str, Any]:
    user_prompt = _build_user_prompt(observation)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=0.0,
            max_tokens=180,
        )
        text = (response.choices[0].message.content or '').strip()
    except Exception:
        return {'action_type': 'run_checks', 'slot_name': '', 'patch_id': '', 'rationale': 'fallback after llm error'}

    try:
        # Strip optional markdown code fences that some LLMs emit
        cleaned = text.strip()
        if cleaned.startswith('```'):
            # Remove first and last fence lines
            lines = cleaned.splitlines()
            cleaned = '\n'.join(
                line for line in lines
                if not line.strip().startswith('```')
            ).strip()
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError('No JSON object found')
        payload = json.loads(cleaned[start:end])
    except Exception:
        return {'action_type': 'run_checks', 'slot_name': '', 'patch_id': '', 'rationale': 'fallback after parse error'}

    if payload.get('action_type') not in {
        'inspect_task', 'inspect_source', 'inspect_fixes', 'inspect_report', 'apply_patch', 'run_checks', 'submit_final'
    }:
        payload['action_type'] = 'run_checks'
    payload.setdefault('slot_name', '')
    payload.setdefault('patch_id', '')
    payload.setdefault('rationale', '')
    return payload


def _action_to_string(payload: dict[str, Any]) -> str:
    parts = [payload.get('action_type', 'run_checks')]
    if payload.get('slot_name'):
        parts.append(f"slot={payload['slot_name']}")
    if payload.get('patch_id'):
        parts.append(f"patch={payload['patch_id']}")
    return '|'.join(parts)


async def run_task(env: ExecutorchEnv, task_id: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    result = await env.reset(task_id=task_id)
    observation = result.observation
    # Use the environment's own step limit so hard tasks get full budget
    task_max_steps = observation.max_steps or MAX_STEPS_PER_TASK
    rewards: list[float] = []
    score = 0.0
    steps_taken = 0
    success = False

    try:
        last_was_patch = False  # force run_checks after every apply_patch

        for step in range(1, task_max_steps + 1):
            # Hard rule: always check after patching — don't trust LLM to do it
            if last_was_patch:
                payload = {
                    'action_type': 'run_checks',
                    'slot_name': '',
                    'patch_id': '',
                    'rationale': 'enforced check after patch',
                }
            elif observation.current_score >= observation.success_threshold and observation.checks_run > 0:
                payload = {
                    'action_type': 'submit_final',
                    'slot_name': '',
                    'patch_id': '',
                    'rationale': 'current report already satisfies target',
                }
            else:
                payload = _call_llm(observation)

            action = ExecutorchAction(
                action_type=payload.get('action_type', 'run_checks'),
                slot_name=payload.get('slot_name', ''),
                patch_id=payload.get('patch_id', ''),
                rationale=payload.get('rationale', ''),
            )
            action_str = _action_to_string(payload)
            result = await env.step(action)
            observation = result.observation
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            error = observation.last_action_error or None

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=result.done,
                error=error,
            )

            if result.done:
                break

            last_was_patch = (action.action_type == 'apply_patch')

        # Safety net: if loop exhausted without submitting, force run_checks
        # then submit to capture any patches the LLM applied but never validated.
        if not result.done and observation.checks_run == 0:
            result = await env.step(ExecutorchAction(action_type='run_checks'))
            observation = result.observation
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken += 1
            log_step(step=steps_taken, action='run_checks', reward=reward, done=result.done, error=None)

        if not result.done:
            result = await env.step(ExecutorchAction(action_type='submit_final'))
            observation = result.observation
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken += 1
            log_step(step=steps_taken, action='submit_final', reward=reward, done=result.done, error=None)

        score = float(observation.final_score or observation.current_score or observation.best_score)
        score = max(0.0, min(1.0, score))
        success = bool(observation.is_success) or score >= SUCCESS_THRESHOLD
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    env = await _connect_env()
    try:
        scores = []
        for task_id in TASK_IDS:
            score = await run_task(env, task_id)
            scores.append(score)
    finally:
        try:
            await env.close()
        except Exception:
            pass

    if scores:
        average = sum(scores) / len(scores)
        print(f'Overall average score: {average:.2f}', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
