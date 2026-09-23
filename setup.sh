#!/usr/bin/env bash
# Everything lands in ./.venv. Remove it by deleting this directory.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || uv venv --python 3.12 .venv
# the results were measured with laya 0.3.8, since removed from PyPI; 0.3.11 truncates the state the same way
VIRTUAL_ENV="$PWD/.venv" uv pip install "laya>=0.3.8"
.venv/bin/python jevlab.py selftest
echo "ready: .venv/bin/python jevlab.py bench"
