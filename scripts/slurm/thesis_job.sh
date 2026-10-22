#!/usr/bin/env bash
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Run this template through scripts/thesis.sh.' >&2; exit 2; }
[[ $# -ge 6 ]] || { echo 'Missing SLURM job arguments.' >&2; exit 2; }
repo_root=$1; action=$2; mode=$3; run_dir=$4; registry=$5; protocol=$6
shift 6
cd -- "$repo_root"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
if [[ $action == setup ]]; then
    bash "$repo_root/scripts/setup_environment.sh"
    action=verify
fi
thesis_python=${THESIS_PYTHON:-$repo_root/.venv-thesis/bin/python}
[[ -x $thesis_python ]] || { echo "Missing interpreter: $thesis_python. Run scripts/thesis.sh setup first." >&2; exit 2; }
command=("$thesis_python" -m experiments.pipeline "$action"
    --run-dir "$run_dir" --registry "$registry" --protocol "$protocol")
case "$mode" in
    regular) command+=(--cell-index "${SLURM_ARRAY_TASK_ID:?}") ;;
    partial) command+=(--partial-index "${SLURM_ARRAY_TASK_ID:?}") ;;
    finalize) command+=(--finalize) ;;
    single) ;;
    *) echo "Unknown job mode: $mode" >&2; exit 2 ;;
esac
exec "${command[@]}" "$@"
