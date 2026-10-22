#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
usage() {
    cat <<'EOF'
Usage: scripts/thesis.sh ACTION [OPTIONS] [-- PIPELINE_OPTIONS...]

Actions: setup verify reproduce discover evaluate export port-validation
         head-diagnostics hpo smoke test
Options:
  --run-dir PATH   Output directory (default: results/reproduction)
  --registry PATH  Artifact registry (default: artifacts/final/registry.json)
  --protocol NAME  thesis (default) or fixed-holdout
  --dry-run        Print submissions without creating files or environments
  --help          Show this help

Every action runs on UKP SLURM. setup creates a separate .venv-thesis.
discover submits regular cells, then ACDC tau partials, then merges them.
reproduce re-evaluates the packaged evidence; it does not rediscover circuits.

Environment overrides:
  THESIS_PYTHON     Interpreter (default: REPO/.venv-thesis/bin/python)
  CONDA_ROOT        Conda installation root (otherwise conda is found on PATH)
  THESIS_PARTITION / THESIS_QOS / THESIS_ACCOUNT
  THESIS_CPU_PARTITION=cpu THESIS_CPU_QOS=cpu (CPU jobs)
  THESIS_GPU_PARTITION=gpu THESIS_GPU_QOS (GPU QOS defaults to the account setting)
  THESIS_CPUS=4 THESIS_CPU_MEM=16G THESIS_GPU_MEM=64G
  THESIS_TIME=2-00:00:00 THESIS_GPU_COUNT=1 THESIS_CONCURRENCY=2
  THESIS_GPU_CONSTRAINT=gpu_mem:80gb
Use THESIS_PARTITION=yolo THESIS_QOS=yolo together for the yolo queue.
EOF
}
fail() { printf 'thesis: %s\n' "$*" >&2; exit 2; }
[[ $# -gt 0 ]] || { usage; exit 2; }
if [[ $1 == --help || $1 == -h ]]; then usage; exit 0; fi
action=$1; shift
case "$action" in
    setup|verify|reproduce|discover|evaluate|export|port-validation|head-diagnostics|hpo|smoke|test) ;;
    *) fail "unknown action: $action" ;;
esac
run_dir=results/reproduction
registry=artifacts/final/registry.json
protocol=thesis
dry_run=0
extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir|--registry|--protocol)
            [[ $# -gt 1 && -n $2 && $2 != --* ]] || fail "$1 requires a value"
            case "$1" in
                --run-dir) run_dir=$2 ;;
                --registry) registry=$2 ;;
                --protocol) protocol=$2 ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; extra=("$@"); break ;;
        *) fail "unknown option: $1 (pass pipeline options after --)" ;;
    esac
done
[[ $protocol == thesis || $protocol == fixed-holdout ]] || fail "invalid protocol: $protocol"
[[ $run_dir == /* ]] || run_dir=$repo_root/$run_dir
[[ $registry == /* ]] || registry=$repo_root/$registry
export THESIS_PYTHON=${THESIS_PYTHON:-$repo_root/.venv-thesis/bin/python}
concurrency=${THESIS_CONCURRENCY:-2}
[[ $concurrency =~ ^[1-9][0-9]*$ ]] || fail 'THESIS_CONCURRENCY must be a positive integer'

submit() {
    local mode=$1 resources=$2 dependency=$3 array=$4
    local partition qos
    local -a command=(sbatch --parsable --job-name="thesis-$action-$mode"
        --chdir="$repo_root" --open-mode=append --export=ALL
        --output="$run_dir/logs/%x-%A_%a.out" --error="$run_dir/logs/%x-%A_%a.err"
        --cpus-per-task="${THESIS_CPUS:-4}" --time="${THESIS_TIME:-2-00:00:00}")
    if [[ $resources == gpu ]]; then
        partition=${THESIS_PARTITION:-${THESIS_GPU_PARTITION:-gpu}}
        qos=${THESIS_QOS:-${THESIS_GPU_QOS:-}}
    else
        partition=${THESIS_PARTITION:-${THESIS_CPU_PARTITION:-cpu}}
        qos=${THESIS_QOS:-${THESIS_CPU_QOS:-cpu}}
    fi
    command+=(--partition="$partition")
    [[ -z $qos ]] || command+=(--qos="$qos")
    [[ -z ${THESIS_ACCOUNT:-} ]] || command+=(--account="$THESIS_ACCOUNT")
    if [[ $resources == gpu ]]; then
        command+=(--gres="gpu:${THESIS_GPU_COUNT:-1}" --mem="${THESIS_GPU_MEM:-64G}"
            --constraint="${THESIS_GPU_CONSTRAINT:-gpu_mem:80gb}")
    else
        command+=(--mem="${THESIS_CPU_MEM:-16G}")
    fi
    [[ -z $dependency ]] || command+=(--dependency="afterok:$dependency")
    [[ -z $array ]] || command+=(--array="$array%$concurrency")
    command+=("$repo_root/scripts/slurm/thesis_job.sh" "$repo_root" "$action" "$mode"
        "$run_dir" "$registry" "$protocol" "${extra[@]}")
    if (( dry_run )); then
        printf '%q ' "${command[@]}" >&2; printf '\n' >&2
        printf 'DRY_%s\n' "$mode"
    else
        mkdir -p -- "$run_dir/logs"
        local submitted
        submitted=$("${command[@]}") || fail 'sbatch submission failed'
        submitted=${submitted%%;*}
        [[ $submitted =~ ^[0-9]+$ ]] || fail "unexpected sbatch job id: $submitted"
        printf '%s\n' "$submitted"
    fi
}

if [[ $action == discover ]]; then
    table=$repo_root/scripts/discovery_cells.tsv
    [[ -f $table ]] || fail "missing discovery table: $table"
    regular_indices=$(awk -F '\t' 'NR > 1 && $7 == "0" { if (n++) printf ","; printf "%s", $1 }' "$table")
    partial_count=$(awk -F '\t' 'NR > 1 && $7 == "1" { n++ } END { print n * 5 }' "$table")
    [[ -n $regular_indices && $partial_count -gt 0 ]] || fail 'invalid discovery table'
    regular_job=$(submit regular gpu '' "$regular_indices")
    # Sequence arrays so their combined GPU concurrency remains bounded.
    partial_job=$(submit partial gpu "$regular_job" "0-$((partial_count - 1))")
    final_job=$(submit finalize cpu "$regular_job:$partial_job" '')
    printf 'regular=%s partial=%s finalize=%s\n' "$regular_job" "$partial_job" "$final_job"
else
    resources=gpu
    case "$action" in setup|verify|export|test) resources=cpu ;; esac
    job=$(submit single "$resources" '' '')
    printf '%s=%s\n' "$action" "$job"
fi
