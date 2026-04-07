import asyncio
import sys

from exicutorch_env.client import ExecutorchEnv
from exicutorch_env.models import ExecutorchAction


async def main():
    env = ExecutorchEnv(base_url="http://localhost:7860")

    print("=== 1. RESET ============================================")
    result = await env.reset(task_id="control_flow_guard")
    o = result.observation
    print(f"task_id      : {o.task_id}")
    print(f"steps_taken  : {o.steps_taken}  (expect 0)")
    print(f"current_score: {o.current_score}  (expect 0.0)")
    assert o.task_id == "control_flow_guard"
    assert o.steps_taken == 0

    print("\n=== 2. INSPECT SOURCE — shaping reward =================")
    r = await env.step(ExecutorchAction(action_type="inspect_source"))
    print(f"reward       : {r.reward}  (expect 0.02)")
    print(f"steps_taken  : {r.observation.steps_taken}  (expect 1)")
    assert r.reward == 0.02, f"Expected shaping reward 0.02, got {r.reward}"
    assert r.observation.steps_taken == 1

    print("\n=== 3. APPLY CORRECT PATCH =============================")
    r = await env.step(ExecutorchAction(action_type="apply_patch", patch_id="body_tensor_where_fix"))
    print(f"message      : {r.observation.message}")
    print(f"steps_taken  : {r.observation.steps_taken}  (expect 2)")
    assert r.observation.steps_taken == 2, f"Expected steps_taken=2, got {r.observation.steps_taken}"

    print("\n=== 4. RUN CHECKS ======================================")
    r = await env.step(ExecutorchAction(action_type="run_checks"))
    o = r.observation
    print(f"parity_score : {o.parity_score}  (expect 1.0)")
    print(f"export_success:{o.export_success}")
    print(f"current_score: {o.current_score}")
    print(f"reward       : {r.reward}")
    print(f"checks_run   : {o.checks_run}  (expect 1)")
    print(f"steps_taken  : {o.steps_taken}  (expect 3)")
    assert o.parity_score == 1.0, "Parity should be perfect with correct patch"
    assert o.checks_run == 1
    assert o.steps_taken == 3, f"Expected steps_taken=3, got {o.steps_taken}"

    print("\n=== 5. STATE PERSISTED — inspect again (no 2nd reward)==")
    r2 = await env.step(ExecutorchAction(action_type="inspect_source"))
    print(f"reward (expect 0.0) : {r2.reward}  (one-time shaping)")
    print(f"steps_taken         : {r2.observation.steps_taken}  (expect 4)")
    assert r2.reward == 0.0, "Shaping reward should only fire once"
    assert r2.observation.steps_taken == 4

    print("\n=== 6. SUBMIT FINAL ====================================")
    r = await env.step(ExecutorchAction(action_type="submit_final"))
    o = r.observation
    print(f"final_score  : {o.final_score}")
    print(f"done         : {r.done}  (expect True)")
    print(f"is_success   : {o.is_success}")
    assert r.done is True

    print("\n=== 7. RESET STARTS FRESH SESSION ======================")
    result2 = await env.reset(task_id="numpy_escape_hatch")
    o2 = result2.observation
    print(f"new task_id  : {o2.task_id}")
    print(f"steps_taken  : {o2.steps_taken}  (expect 0)")
    print(f"current_score: {o2.current_score}  (expect 0.0)")
    assert o2.task_id == "numpy_escape_hatch"
    assert o2.steps_taken == 0
    assert o2.current_score == 0.0

    print("\n=== 8. WRONG PATCH SCORES BELOW THRESHOLD ==============")
    await env.step(ExecutorchAction(action_type="apply_patch", patch_id="body_zero_fill"))
    r = await env.step(ExecutorchAction(action_type="run_checks"))
    o = r.observation
    print(f"parity_score : {o.parity_score}  (expect 0.0)")
    print(f"current_score: {o.current_score}  (expect < {o.success_threshold})")
    assert o.parity_score < 1.0
    assert o.current_score < o.success_threshold
    print("PASS: wrong patch correctly scored below threshold")

    await env.close()

    print("\n" + "=" * 55)
    print("   ALL 8 WS STATE PERSISTENCE TESTS PASSED ✓")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
