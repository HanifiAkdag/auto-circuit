#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
prefix=$repo_root/.venv-thesis
if [[ ${1:-} == --help ]]; then
    echo 'Usage: scripts/thesis.sh setup'
    echo 'Creates an independent conda Python 3.10.20 environment; run on a compute node.'
    exit 0
fi
[[ $# -eq 0 ]] || { echo "Unknown setup option: $1" >&2; exit 2; }
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Setup must run on a SLURM compute node via scripts/thesis.sh setup.' >&2; exit 2; }
export PIP_PROGRESS_BAR=off
if [[ -n ${CONDA_ROOT:-} ]]; then
    conda_executable=$CONDA_ROOT/bin/conda
else
    conda_executable=$(type -P conda) || { echo 'Set CONDA_ROOT to your conda installation.' >&2; exit 2; }
fi
[[ -x $conda_executable ]] || { echo "Missing conda: $conda_executable" >&2; exit 2; }
if [[ -e $prefix && ! -f $prefix/.thesis-environment ]]; then
    echo "Refusing to alter an existing environment without a thesis marker: $prefix" >&2
    exit 2
fi
if [[ ! -x $prefix/bin/python ]]; then
    "$conda_executable" create --yes --quiet --prefix "$prefix" --override-channels --channel conda-forge python=3.10.20 pip
    touch "$prefix/.thesis-environment"
fi
"$prefix/bin/python" -m pip install --upgrade 'pip==26.0.1'
"$prefix/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu130 'torch==2.11.0+cu130'
"$prefix/bin/python" -m pip install -r "$repo_root/requirements-thesis.txt"
"$prefix/bin/python" -m pip install --no-deps --editable "$repo_root[thesis]"
"$prefix/bin/python" -m pip check
"$prefix/bin/python" -m pip freeze > "$prefix/installed-packages.txt"
printf 'Environment ready: %s\n' "$prefix"
