from exicutorch_env.models import ExecutorchAction
from exicutorch_env.server.executorch_env_environment import ExecutorchEnvironment


def test_reset_returns_task_metadata():
    env = ExecutorchEnvironment()
    observation = env.reset(task_id='control_flow_guard')
    assert observation.task_id == 'control_flow_guard'
    assert observation.task_title
    assert 'body' in observation.available_slots
    assert observation.patch_catalog['body']


def test_easy_task_can_be_solved_with_correct_patch():
    env = ExecutorchEnvironment()
    env.reset(task_id='control_flow_guard')
    env.step(ExecutorchAction(action_type='apply_patch', patch_id='body_tensor_where_fix'))
    checked = env.step(ExecutorchAction(action_type='run_checks'))
    assert checked.export_success is True
    assert checked.parity_score == 1.0
    final = env.step(ExecutorchAction(action_type='submit_final'))
    assert final.done is True
    assert final.final_score >= 0.90


def test_wrong_patch_keeps_score_below_success_threshold():
    env = ExecutorchEnvironment()
    env.reset(task_id='numpy_escape_hatch')
    env.step(ExecutorchAction(action_type='apply_patch', patch_id='body_zero_fill'))
    checked = env.step(ExecutorchAction(action_type='run_checks'))
    assert checked.parity_score < 1.0
    assert checked.current_score < checked.success_threshold


def test_hard_task_requires_two_repairs():
    env = ExecutorchEnvironment()
    env.reset(task_id='edge_score_block')
    env.step(ExecutorchAction(action_type='apply_patch', patch_id='pre_torch_centering'))
    mid = env.step(ExecutorchAction(action_type='run_checks'))
    assert mid.current_score < mid.success_threshold
    env.step(ExecutorchAction(action_type='apply_patch', patch_id='score_tensor_where_norm'))
    final_check = env.step(ExecutorchAction(action_type='run_checks'))
    assert final_check.parity_score == 1.0
    assert final_check.current_score >= 0.90
