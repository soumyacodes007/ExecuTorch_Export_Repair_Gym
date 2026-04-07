from .models import ExecutorchAction, ExecutorchObservation, ExecutorchState


def __getattr__(name: str):
    if name == 'ExecutorchEnv':
        from .client import ExecutorchEnv

        return ExecutorchEnv
    raise AttributeError(name)

__all__ = [
    'ExecutorchAction',
    'ExecutorchObservation',
    'ExecutorchState',
    'ExecutorchEnv',
]
