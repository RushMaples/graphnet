#!/bin/bash
# Mocked launcher regression tests. No Slurm submission, Python, GPU or databases.
set -euo pipefail
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TEMP_BASE="${TEMP:-${TMP:-/tmp}}"
if command -v cygpath >/dev/null 2>&1; then TEMP_BASE="$(cygpath -u "$TEMP_BASE")"; fi
TEMP_BASE="$(cd "$TEMP_BASE" && pwd -P)"
case "${TEMP_BASE,,}/" in
    *'/onedrive/'*|*'/onedrive - '*) echo 'Refusing temporary fixtures inside OneDrive.' >&2; exit 1 ;;
esac
SCRATCH="$(mktemp -d "${TEMP_BASE}/icemix-launcher.XXXXXXXX")"
SCRATCH="$(cd "$SCRATCH" && pwd -P)"
validate_scratch() {
    local resolved
    resolved="$(cd "$SCRATCH" && pwd -P)" || return 1
    [[ "$resolved" == "$SCRATCH" && "$resolved" == "$TEMP_BASE"/icemix-launcher.* ]] || return 1
    [[ "$resolved" != "$SOURCE_ROOT" && "$resolved" != "$SOURCE_ROOT"/* ]] || return 1
    case "${resolved,,}/" in *'/onedrive/'*|*'/onedrive - '*) return 1 ;; esac
}
cleanup() {
    local status=$?
    if validate_scratch; then rm -rf -- "$SCRATCH"; else
        echo "Refusing cleanup of unverified temporary path: $SCRATCH" >&2
        status=1
    fi
    exit "$status"
}
trap cleanup EXIT
validate_scratch
mkdir -p "$SCRATCH"/{bin,checkout/comparison,parent/both/checkpoints,selection,node-tmp,runs}
mkdir -p "$SCRATCH/conda/etc/profile.d"
touch "$SCRATCH/checkout/comparison/pace_full.sbatch" "$SCRATCH/checkout/comparison/full_augmentation.py"
touch "$SCRATCH/parent/both/"{complete.json,selected_checkpoint.json,checkpoint_source.json,timing.csv}
printf 'checkpoint fixture\n' > "$SCRATCH/parent/both/checkpoints/last.ckpt"
touch "$SCRATCH/conda/etc/profile.d/conda.sh"
export FIXTURE="$SCRATCH" ICEMIX_CHECKOUT="$SCRATCH/checkout"
export ENV_PREFIX="$SCRATCH" PILOT_SELECTION_ROOT="$SCRATCH/selection"
export TMPDIR="$SCRATCH/node-tmp" AUGMENTATION_RUN_ROOT="$SCRATCH/runs"
export SOURCE_DATA_ROOT="$SCRATCH/missing-databases"
export PATH="$SCRATCH/bin:$PATH"
unset SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES
cat > "$SCRATCH/bin/sbatch" <<'SH'
#!/bin/bash
set -eu
printf 'CALL\n' >> "$FIXTURE/sbatch.log"
printf 'ARG:%s\n' "$@" >> "$FIXTURE/sbatch.log"
printf 'SBATCH_EXCLUDE:%s\n' "${SBATCH_EXCLUDE-<unset>}" >> "$FIXTURE/sbatch.log"
if [[ "$1" != --test-only ]]; then printf '14509999;phoenix\n'; fi
SH
cat > "$SCRATCH/bin/scontrol" <<'SH'
#!/bin/bash
set -eu
printf 'CALL:%s\n' "$*" >> "$FIXTURE/scontrol.log"
if [[ "$1 $2" == 'show hostnames' ]]; then
    [[ "${MOCK_MODE:-}" != expansion_failure ]] || exit 93
    IFS=, read -r -a parts <<< "$3"
    for part in "${parts[@]}"; do
        case "$part" in
            'custom[01-02]') printf 'custom01\ncustom02\n' ;;
            *) printf '%s\n' "$part" ;;
        esac
    done
elif [[ "$1 $2 ${3-}" == '--oneliner show job' ]]; then
    [[ "$4" == 14509999 ]]
    mode="${MOCK_MODE:-valid}"
    [[ "$mode" != controller_failure ]] || exit 92
    jobid=14509999 state=PENDING priority='Priority=0' exclusions="ExcNodeList=${MOCK_ACCEPTED_EXCLUSIONS-$EXCLUDE_NODES}"
    case "$mode" in
        wrong_id) jobid=14509998 ;;
        running) state=RUNNING ;;
        nonzero_priority) priority='Priority=1' ;;
        missing_priority) priority='' ;;
        missing_exclusions) exclusions='' ;;
        null_exclusions) exclusions='ExcNodeList=(null)' ;;
        duplicate_id) jobid='14509999 JobId=14509999' ;;
    esac
    printf 'JobId=%s JobState=%s %s %s NodeList=(null)\n' "$jobid" "$state" "$priority" "$exclusions"
elif [[ "$1" == release ]]; then
    [[ "$2" == 14509999 ]]
    [[ "${MOCK_MODE:-}" != release_failure ]] || exit 94
else exit 91
fi
SH
cat > "$SCRATCH/bin/conda" <<'SH'
#!/bin/bash
if [[ "$1 $2" == 'info --base' ]]; then printf '%s/conda\n' "$FIXTURE"; fi
exit 0
SH
cat > "$SCRATCH/bin/python" <<'SH'
#!/bin/bash
printf 'python\n' >> "$FIXTURE/work.log"
exit 0
SH
cat > "$SCRATCH/bin/srun" <<'SH'
#!/bin/bash
set -eu
printf 'srun\n' >> "$FIXTURE/work.log"
cat > "$FIXTURE/gpu-smoke.py.txt"
exit "${MOCK_CUDA_STATUS:-0}"
SH
for stub in module nvidia-smi; do
    printf '#!/bin/bash\nexit 0\n' > "$SCRATCH/bin/$stub"
done
chmod +x "$SCRATCH/bin/"*
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
contains() { grep -Fq -- "$2" "$1" || fail "Missing '$2' in $1"; }
absent() { if grep -Fq -- "$2" "$1"; then fail "Unexpected '$2' in $1"; fi; }
count_is() {
    local count
    count="$(grep -Fc -- "$2" "$1" || true)"
    [[ "$count" == "$3" ]] || fail "Expected $3 occurrences of '$2', found $count"
}
run_helper() {
    local name="$1" choice="$2" expected="${3:-0}" mode="${4:-valid}" accepted="${5-default}" status=0
    : > "$SCRATCH/sbatch.log"
    : > "$SCRATCH/scontrol.log"
    (
        export SBATCH_EXCLUDE=stale-node
        export MOCK_MODE="$mode"
        unset MOCK_ACCEPTED_EXCLUSIONS
        if [[ "$accepted" != default ]]; then export MOCK_ACCEPTED_EXCLUSIONS="$accepted"; fi
        if [[ "$choice" == default ]]; then unset EXCLUDE_NODES; else export EXCLUDE_NODES="$choice"; fi
        bash "$SOURCE_ROOT/comparison/submit_full.sh" "$SCRATCH/parent"
    ) > "$SCRATCH/$name.out" 2>&1 || status=$?
    [[ "$status" == "$expected" ]] || { cat "$SCRATCH/$name.out" >&2; fail "Helper $name expected status $expected, got $status"; }
    if [[ "$expected" == 2 ]]; then
        [[ ! -s "$SCRATCH/sbatch.log" ]] || fail "Rejected exclusions submitted a job: $name"
        [[ ! -s "$SCRATCH/scontrol.log" ]] || fail "Rejected exclusions queried controller: $name"
        return
    fi
    count_is "$SCRATCH/sbatch.log" 'CALL' 2
    count_is "$SCRATCH/sbatch.log" 'ARG:--test-only' 1
    count_is "$SCRATCH/sbatch.log" 'ARG:--parsable' 1
    count_is "$SCRATCH/sbatch.log" 'ARG:--hold' 1
    absent "$SCRATCH/sbatch.log" 'SBATCH_EXCLUDE:stale-node'
    contains "$SCRATCH/$name.out" 'Submitted batch job 14509999'
    contains "$SCRATCH/$name.out" 'Checkout:'
    contains "$SCRATCH/$name.out" 'Revision:'
    contains "$SCRATCH/scontrol.log" 'CALL:--oneliner show job 14509999'
    [[ ! -e "$SCRATCH/runs/full-14509999" ]] || fail 'Helper created allocation directory'
    contains "$SCRATCH/runs/submission-14509999.txt" 'job_id=14509999'
    if [[ "$expected" == 0 ]]; then
        contains "$SCRATCH/$name.out" 'Controller accepted:'
        contains "$SCRATCH/$name.out" 'Priority=0'
        contains "$SCRATCH/$name.out" 'Released verified job 14509999.'
        count_is "$SCRATCH/scontrol.log" 'CALL:release 14509999' 1
        contains "$SCRATCH/runs/submission-14509999.txt" 'Released verified job 14509999.'
        [[ "$(tail -n 1 "$SCRATCH/scontrol.log")" == 'CALL:release 14509999' ]] || fail 'Release preceded verification'
    elif [[ "$expected" == 6 ]]; then
        count_is "$SCRATCH/scontrol.log" 'CALL:release 14509999' 1
        contains "$SCRATCH/$name.out" 'Release command failed for job 14509999'
        contains "$SCRATCH/$name.out" 'do not resubmit'
    else
        absent "$SCRATCH/scontrol.log" 'CALL:release'
        contains "$SCRATCH/$name.out" 'job 14509999 was NOT released by this helper'
        contains "$SCRATCH/$name.out" 'do not resubmit'
        contains "$SCRATCH/runs/submission-14509999.txt" 'was NOT released'
    fi
}
contains "$SOURCE_ROOT/comparison/pace_full.sbatch" '#SBATCH --exclude=atl1-1-01-004-31-0'
run_helper default default
count_is "$SCRATCH/sbatch.log" 'ARG:--exclude=atl1-1-01-004-31-0' 2
count_is "$SCRATCH/sbatch.log" 'ARG:atl1-1-01-004-31-0' 2
contains "$SCRATCH/default.out" 'ExcNodeList=atl1-1-01-004-31-0'
run_helper custom 'custom[01-02]'
count_is "$SCRATCH/sbatch.log" 'ARG:--exclude=atl1-1-01-004-31-0,custom[01-02]' 2
count_is "$SCRATCH/sbatch.log" 'ARG:atl1-1-01-004-31-0,custom[01-02]' 2
contains "$SCRATCH/custom.out" 'ExcNodeList=atl1-1-01-004-31-0,custom[01-02]'
# Equivalent expanded host lists and supersets are safe; partial lists are not.
run_helper expanded_range 'custom[01-02]' 0 valid 'custom02,atl1-1-01-004-31-0,custom01'
run_helper accepted_range 'custom01,custom02' 0 valid 'atl1-1-01-004-31-0,custom[01-02]'
run_helper superset default 0 valid 'other-node,atl1-1-01-004-31-0'
run_helper missing_custom 'custom[01-02]' 5 valid 'atl1-1-01-004-31-0,custom01'
run_helper wrong_exclusions default 5 valid other-node
run_helper empty_exclusions default 5 valid ''
for mode in missing_exclusions null_exclusions controller_failure missing_priority nonzero_priority wrong_id duplicate_id running expansion_failure; do
    run_helper "$mode" default 5 "$mode"
done
run_helper release_failure default 6 release_failure
run_helper none none 2
contains "$SCRATCH/none.out" 'EXCLUDE_NODES cannot be empty or none'
run_helper empty '' 2
contains "$SCRATCH/empty.out" 'EXCLUDE_NODES cannot be empty or none'
printf 'PASS: mandatory exclusions, held verification, fail-closed release and receipts\n'

run_allocation() {
    local id="$1" node="$2" cuda_status="$3" excluded="$4" expected="$5" status=0
    : > "$SCRATCH/work.log"
    (
        unset EXCLUDE_NODES
        export SLURM_JOB_ID="$id" SLURMD_NODENAME="$node" MOCK_CUDA_STATUS="$cuda_status"
        bash "$SOURCE_ROOT/comparison/pace_full.sbatch" "$SCRATCH/parent" "$excluded"
    ) > "$SCRATCH/$id.out" 2>&1 || status=$?
    [[ "$status" == "$expected" ]] || {
        cat "$SCRATCH/$id.out" >&2
        fail "Allocation $id expected status $expected, got $status"
    }
    contains "$SCRATCH/runs/full-$id/slurm_status.txt" "exit_code=$expected"
}
run_allocation 101 atl1-1-01-004-31-0 0 atl1-1-01-004-31-0 5
[[ ! -s "$SCRATCH/work.log" ]] || fail 'Excluded host ran Python or srun'
contains "$SCRATCH/runs/full-101/slurm_status.txt" 'last_phase=allocation_guard'
absent "$SCRATCH/runs/full-101/phase_timing.csv" 'database_staging'
# The positional argument must survive a missing exported EXCLUDE_NODES variable.
run_allocation 102 custom02 0 'custom[01-02]' 5
[[ ! -s "$SCRATCH/work.log" ]] || fail 'Custom excluded host ran work'
run_allocation 103 safe-node 23 atl1-1-01-004-31-0 23
contains "$SCRATCH/runs/full-103/slurm_status.txt" 'last_phase=gpu_health'
absent "$SCRATCH/runs/full-103/phase_timing.csv" 'database_staging'
absent "$SCRATCH/work.log" 'python'
contains "$SCRATCH/gpu-smoke.py.txt" 'torch.arange('
contains "$SCRATCH/gpu-smoke.py.txt" 'device="cuda"'
contains "$SCRATCH/gpu-smoke.py.txt" 'total = (values + 1).sum()'
contains "$SCRATCH/gpu-smoke.py.txt" 'torch.cuda.synchronize()'
contains "$SCRATCH/gpu-smoke.py.txt" 'torch.isfinite(total).item()'
contains "$SCRATCH/gpu-smoke.py.txt" 'total.item() != count * (count + 1) / 2'
# Missing fixture databases deliberately stop immediately after the phase begins.
run_allocation 104 safe-node 0 atl1-1-01-004-31-0 1
contains "$SCRATCH/runs/full-104/slurm_status.txt" 'last_phase=database_staging'
contains "$SCRATCH/work.log" 'python'
[[ ! -e "$SCRATCH/node-tmp/icemix-full-104" ]] || fail 'Unexpected fixture database copy'
printf 'PASS: assigned-node guards, CUDA failure ordering and mocked staging readiness\n'
printf 'All full-launcher shell regressions passed (mocked; no GPU validation).\n'
