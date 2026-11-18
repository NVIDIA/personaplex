#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

venv_dir="${PERSONAPLEX_VENV:-.venv-nix}"
python_bin="${venv_dir}/bin/python"

if [[ ! -x "$python_bin" ]]; then
  python -m venv --copies "$venv_dir"
fi

"$python_bin" -m pip install --upgrade pip
"$python_bin" -m pip install -e 'moshi/.[mlx]'
"$python_bin" -m pip check

echo "PersonaPlex Python environment is ready at ${venv_dir}"
