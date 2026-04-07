"""FastAPI app for the ExecuTorch Export Repair Gym.

This environment is framed as an edge deployment repair workflow:
agents inspect broken model code, apply fixes, validate parity, and
submit only when export readiness has been restored.
"""

from __future__ import annotations

import os

from fastapi.responses import JSONResponse
from openenv.core.env_server import create_app

try:
    from exicutorch_env.models import ExecutorchAction, ExecutorchObservation
    from exicutorch_env.server.exicutorch_env_environment import ExecutorchEnvironment
except ModuleNotFoundError:
    from models import ExecutorchAction, ExecutorchObservation  # type: ignore[import-not-found]
    from server.exicutorch_env_environment import ExecutorchEnvironment  # type: ignore[import-not-found]

MAX_CONCURRENT_ENVS = int(os.environ.get('MAX_CONCURRENT_ENVS', '32'))
ENABLE_WEB_INTERFACE = os.environ.get('ENABLE_WEB_INTERFACE', 'true').lower() == 'true'
ENV_NAME = 'executorch_export_repair_gym'


def create_environment() -> ExecutorchEnvironment:
    return ExecutorchEnvironment()


if ENABLE_WEB_INTERFACE:
    try:
        from openenv.core.env_server import create_web_interface_app

        app = create_web_interface_app(
            create_environment,
            ExecutorchAction,
            ExecutorchObservation,
            env_name=ENV_NAME,
            max_concurrent_envs=MAX_CONCURRENT_ENVS,
        )
    except (ModuleNotFoundError, ImportError):
        ENABLE_WEB_INTERFACE = False

if not ENABLE_WEB_INTERFACE:
    app = create_app(
        create_environment,
        ExecutorchAction,
        ExecutorchObservation,
        env_name=ENV_NAME,
        max_concurrent_envs=MAX_CONCURRENT_ENVS,
    )


@app.get('/manifest.json', include_in_schema=False)
async def web_manifest():
    return JSONResponse(
        {
            'name': 'ExecuTorch Export Repair Gym',
            'short_name': 'ExecRepair',
            'description': 'Fix tiny PyTorch models so they become edge-deployable with ExecuTorch.',
            'start_url': '/web/',
            'display': 'standalone',
            'background_color': '#0f172a',
            'theme_color': '#2563eb',
            'icons': [
                {
                    'src': 'https://huggingface.co/front/assets/huggingface_logo-noborder.svg',
                    'sizes': 'any',
                    'type': 'image/svg+xml',
                }
            ],
        }
    )


@app.get('/.well-known/appspecific/com.chrome.devtools.json', include_in_schema=False)
async def chrome_devtools():
    return JSONResponse({})


def main(host: str = '0.0.0.0', port: int = 7860):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
