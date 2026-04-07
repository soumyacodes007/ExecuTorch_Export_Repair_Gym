from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


TensorTuple = Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class PatchOption:
    patch_id: str
    slot_name: str
    title: str
    description: str
    code: str


@dataclass(frozen=True)
class RepairTask:
    task_id: str
    title: str
    difficulty: str
    description: str
    template_source: str
    class_name: str
    slot_order: Tuple[str, ...]
    default_patch_ids: Dict[str, str]
    correct_patch_ids: Dict[str, str]
    patches_by_slot: Dict[str, Tuple[PatchOption, ...]]
    example_inputs: TensorTuple
    hidden_inputs: Tuple[TensorTuple, ...]
    success_threshold: float
    max_steps: int
    supported_op_hints: Tuple[str, ...]
    atol: float = 1e-4
    rtol: float = 1e-4


def _t(data: List[List[float]] | List[float]) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32)


def _patch(slot_name: str, patch_id: str, title: str, description: str, code: str) -> PatchOption:
    return PatchOption(
        patch_id=patch_id,
        slot_name=slot_name,
        title=title,
        description=description,
        code=code.strip("\n"),
    )


TASK_1 = RepairTask(
    task_id='control_flow_guard',
    title='Control Flow Guard Cleanup',
    difficulty='easy',
    description=(
        'This tiny module is numerically correct in eager mode but uses tensor-to-Python '
        'scalar extraction (`.item()`) to branch inside `forward`. Repair it so the model '
        'remains behaviorally correct and becomes export-friendly for edge deployment.'
    ),
    template_source='''import torch\nimport torch.nn as nn\n\n\nclass BranchyGate(nn.Module):\n    def forward(self, x):\n{{body}}\n''',
    class_name='BranchyGate',
    slot_order=('body',),
    default_patch_ids={'body': 'body_broken_item_branch'},
    correct_patch_ids={'body': 'body_tensor_where_fix'},
    patches_by_slot={
        'body': (
            _patch(
                'body',
                'body_broken_item_branch',
                'Keep Python branch',
                'Leaves the `.item()`-based branch in place. Eager execution works, but export should still fail.',
                '''if x.mean().item() > 0:\n    return torch.relu(x) + 0.25\nreturn x - 0.25''',
            ),
            _patch(
                'body',
                'body_tensor_where_fix',
                'Tensorized branch with torch.where',
                'Replace data-dependent Python control flow with tensor operations so export can trace the full graph.',
                '''cond = x.mean() > 0\npositive = torch.relu(x) + 0.25\nnegative = x - 0.25\nreturn torch.where(cond, positive, negative)''',
            ),
            _patch(
                'body',
                'body_always_positive',
                'Always take positive branch',
                'This is exportable but changes semantics whenever the mean is not positive.',
                '''return torch.relu(x) + 0.25''',
            ),
            _patch(
                'body',
                'body_detached_item_branch',
                'Detach then branch in Python',
                'Still converts a tensor value into Python control flow, so export should remain unhappy.',
                '''score = x.detach().mean().item()\nif score > 0:\n    return torch.relu(x) + 0.25\nreturn x - 0.25''',
            ),
        )
    },
    example_inputs=(_t([[0.5, -0.2, 1.0, -1.5]]),),
    hidden_inputs=(
        (_t([[1.2, -0.5, 0.3, -0.1]]),),
        (_t([[-2.0, -1.0, -0.5, -0.1]]),),
        (_t([[0.1, 0.2, -0.3, 0.4]]),),
    ),
    success_threshold=0.93,
    max_steps=8,
    supported_op_hints=('aten.relu', 'aten.where', 'aten.add', 'aten.sub', 'aten.gt', 'aten.mean'),
)


TASK_2 = RepairTask(
    task_id='numpy_escape_hatch',
    title='Numpy Escape Hatch Removal',
    difficulty='medium',
    description=(
        'A small preprocessing block jumps out to NumPy in the middle of `forward`. '
        'That eager pattern breaks edge export. Keep the math identical while staying '
        'inside tensor operations only.'
    ),
    template_source='''import numpy as np\nimport torch\nimport torch.nn as nn\n\n\nclass ChannelCenter(nn.Module):\n    def forward(self, x):\n{{body}}\n''',
    class_name='ChannelCenter',
    slot_order=('body',),
    default_patch_ids={'body': 'body_numpy_roundtrip'},
    correct_patch_ids={'body': 'body_torch_centering'},
    patches_by_slot={
        'body': (
            _patch(
                'body',
                'body_numpy_roundtrip',
                'Keep NumPy round-trip',
                'Converts the tensor to NumPy, does the work outside torch, then converts back. Realistic bug, still export-hostile.',
                '''arr = x.detach().cpu().numpy()\ncentered = arr - arr.mean(axis=-1, keepdims=True)\nreturn torch.from_numpy(centered).to(x.device)''',
            ),
            _patch(
                'body',
                'body_torch_centering',
                'Pure torch centering',
                'Compute the same centering transform with tensor ops so export and lowering can keep everything on-graph.',
                '''return x - x.mean(dim=-1, keepdim=True)''',
            ),
            _patch(
                'body',
                'body_zero_fill',
                'Replace with zeros',
                'Exportable, but destroys the numerical behavior entirely.',
                '''return torch.zeros_like(x)''',
            ),
            _patch(
                'body',
                'body_list_roundtrip',
                'Round-trip through Python lists',
                'Avoids NumPy, but still leaves tensor land and should not be considered edge-friendly.',
                '''rows = x.detach().tolist()\ncentered = []\nfor row in rows:\n    row_mean = sum(row) / len(row)\n    centered.append([value - row_mean for value in row])\nreturn torch.tensor(centered, dtype=x.dtype, device=x.device)''',
            ),
        )
    },
    example_inputs=(_t([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]]),),
    hidden_inputs=(
        (_t([[0.2, 0.4, 0.8, 1.6], [1.0, 0.0, -1.0, -2.0]]),),
        (_t([[5.0, 4.0, 3.0, 2.0], [1.5, 1.5, 1.5, 1.5]]),),
        (_t([[10.0, 0.0, -10.0, 5.0], [-3.0, -1.0, 1.0, 3.0]]),),
    ),
    success_threshold=0.95,
    max_steps=10,
    supported_op_hints=('aten.mean', 'aten.sub'),
)


TASK_3 = RepairTask(
    task_id='edge_score_block',
    title='Two-Stage Edge Score Block Repair',
    difficulty='hard',
    description=(
        'This model has two separate edge-export problems: a NumPy preprocessing path and '
        'a data-dependent scalar branch that also uses a less deployment-friendly norm op. '
        'Repair both stages so the module exports cleanly and lowers to ExecuTorch.'
    ),
    template_source='''import numpy as np\nimport torch\nimport torch.nn as nn\n\n\nclass EdgeScoreBlock(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.register_buffer("bias", torch.tensor([0.10], dtype=torch.float32))\n\n    def forward(self, x):\n{{preprocess}}\n{{score}}\n        return y + self.bias\n''',
    class_name='EdgeScoreBlock',
    slot_order=('preprocess', 'score'),
    default_patch_ids={
        'preprocess': 'pre_numpy_centering',
        'score': 'score_item_branch_norm',
    },
    correct_patch_ids={
        'preprocess': 'pre_torch_centering',
        'score': 'score_tensor_where_norm',
    },
    patches_by_slot={
        'preprocess': (
            _patch(
                'preprocess',
                'pre_numpy_centering',
                'NumPy centering path',
                'Current buggy path: convert to NumPy, center there, then wrap back into a tensor.',
                '''arr = x.detach().cpu().numpy()\ncentered = arr - arr.mean(axis=-1, keepdims=True)\nx = torch.from_numpy(centered).to(x.device)''',
            ),
            _patch(
                'preprocess',
                'pre_torch_centering',
                'Torch-native centering',
                'Keep preprocessing entirely in torch so both export and lowering can see it.',
                '''x = x - x.mean(dim=-1, keepdim=True)''',
            ),
            _patch(
                'preprocess',
                'pre_identity',
                'Skip preprocessing',
                'Easy to export, but silently changes the numerical behavior.',
                '''x = x''',
            ),
        ),
        'score': (
            _patch(
                'score',
                'score_item_branch_norm',
                'Item-based branch with vector_norm',
                'Leaves Python control flow and a harder-to-lower operator in the graph.',
                '''if x.norm().item() > 1.0:\n    y = torch.linalg.vector_norm(x, dim=-1, keepdim=True)\nelse:\n    y = x.abs().mean(dim=-1, keepdim=True)''',
            ),
            _patch(
                'score',
                'score_tensor_where_norm',
                'Tensorized branch with supported ops',
                'Compute the norm via primitive ops and use `torch.where` to stay export-friendly.',
                '''norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True))\nfallback = x.abs().mean(dim=-1, keepdim=True)\ncond = norm > 1.0\ny = torch.where(cond, norm, fallback)''',
            ),
            _patch(
                'score',
                'score_always_abs_mean',
                'Always use abs-mean path',
                'Exportable, but discards the intended branch logic.',
                '''y = x.abs().mean(dim=-1, keepdim=True)''',
            ),
            _patch(
                'score',
                'score_detached_branch',
                'Detach then branch in Python',
                'Still uses Python scalar control flow, so export remains fragile.',
                '''score = x.detach().norm().item()\nif score > 1.0:\n    y = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True))\nelse:\n    y = x.abs().mean(dim=-1, keepdim=True)''',
            ),
        ),
    },
    example_inputs=(_t([[1.0, 0.0, -1.0, 2.0], [0.5, -0.5, 0.5, -0.5]]),),
    hidden_inputs=(
        (_t([[2.0, 1.0, -2.0, -1.0], [0.2, -0.2, 0.2, -0.2]]),),
        (_t([[0.1, 0.1, 0.1, 0.1], [3.0, 4.0, 0.0, 0.0]]),),
        (_t([[-1.5, -0.5, 0.5, 1.5], [2.0, 2.0, -2.0, -2.0]]),),
    ),
    success_threshold=0.96,
    max_steps=12,
    supported_op_hints=('aten.mean', 'aten.sub', 'aten.mul', 'aten.sum', 'aten.sqrt', 'aten.where', 'aten.gt', 'aten.abs', 'aten.add'),
)


REPAIR_TASKS: Dict[str, RepairTask] = {
    task.task_id: task for task in (TASK_1, TASK_2, TASK_3)
}

DEFAULT_TASK_ID = TASK_1.task_id
TASK_ORDER = tuple(REPAIR_TASKS.keys())


def render_source(task: RepairTask, selected_patch_ids: Dict[str, str]) -> str:
    source = task.template_source
    for slot_name in task.slot_order:
        patch_id = selected_patch_ids.get(slot_name, task.default_patch_ids[slot_name])
        patch = next(option for option in task.patches_by_slot[slot_name] if option.patch_id == patch_id)
        indented = '\n'.join(f'        {line}' if line else '' for line in patch.code.splitlines())
        source = source.replace(f'{{{{{slot_name}}}}}', indented)
    return source


def correct_source(task: RepairTask) -> str:
    return render_source(task, task.correct_patch_ids)


def patch_catalog(task: RepairTask) -> Dict[str, List[Dict[str, str]]]:
    catalog: Dict[str, List[Dict[str, str]]] = {}
    for slot_name, options in task.patches_by_slot.items():
        catalog[slot_name] = [
            {
                'patch_id': option.patch_id,
                'title': option.title,
                'description': option.description,
            }
            for option in options
        ]
    return catalog
