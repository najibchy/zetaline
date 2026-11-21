#!/usr/bin/env bash
# Run the test suite in isolation from system/ROS pytest plugins.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec python -m pytest -p asyncio "$@"
