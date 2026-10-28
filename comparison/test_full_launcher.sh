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
# Fail if any inherited submission input survived sanitation.
for variable in "${!SBATCH_@}"; do printf 'LEAK:%s\n' "$variable" >> "$FIXTURE/sbatch.log"; done
for variable in SLURM_CLUSTERS SLURM_HINT SLURM_MEM_BIND EXCLUDE_NODES; do
    if [[ -v "$variable" ]]; then printf 'LEAK:%s\n' "$variable" >> "$FIXTURE/sbatch.log"; fi
done
[[ "${KEEP_USER_SETTING:-}" == preserved ]]
[[ "${SLURM_CONF:-}" == /site/slurm.conf ]]
[[ "${SLURM_JOB_ID:-}" == parent-allocation ]]
printf '%s\n' "${MOCK_RESPONSE:-14509999;phoenix}"
SH
cat > "$SCRATCH/bin/scontrol" <<'SH'
#!/bin/bash
set -eu
printf 'CALL:%s\n' "$*" >> "$FIXTURE/scontrol.log"
[[ "$*" == '--oneliner show job 14509999' || "$*" == 'show job --oneliner '* ]] || exit 91
[[ "${MOCK_MODE:-}" != controller_failure ]] || exit 92
printf 'JobId=14509999 JobState=RUNNING Priority=100 ExcNodeList=site-policy-node NodeList=atl1-1-01-004-31-0\n'
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
    local name="$1" mode="${2:-valid}" expected="${3:-0}" status=0
    local account=custom-account qos=custom-qos time_limit=07:00:00
    if [[ "$mode" == defaults ]]; then
        account=gts-itaboada3 qos=inferno time_limit=08:00:00
    fi
    : > "$SCRATCH/sbatch.log"
    : > "$SCRATCH/scontrol.log"
    (
        export SBATCH_ACCOUNT=custom-account SBATCH_QOS=custom-qos FULL_TIME=07:00:00
        export SBATCH_EXCLUDE=stale-node SBATCH_NODELIST=stale-node SBATCH_CONSTRAINT=stale
        export SBATCH_HOLD=yes SBATCH_DEPENDENCY=afterok:1 SBATCH_RESERVATION=stale
        export SBATCH_PARTITION=wrong SBATCH_GRES=gpu:wrong:4 SBATCH_MEM_PER_GPU=999G
        export SBATCH_MEM_PER_CPU=999G SBATCH_GPUS=4 SBATCH_CPUS_PER_GPU=99
        export SBATCH_EXPORT=NONE SBATCH_CLUSTERS=wrong SBATCH_WAIT=yes
        export SLURM_CLUSTERS=wrong SLURM_HINT=nomultithread SLURM_MEM_BIND=local
        export EXCLUDE_NODES=atl1-1-01-004-31-0 KEEP_USER_SETTING=preserved
        export SLURM_CONF=/site/slurm.conf SLURM_JOB_ID=parent-allocation MOCK_MODE="$mode"
        if [[ "$mode" == defaults ]]; then
            for variable in "${!SBATCH_@}"; do unset "$variable"; done
            unset FULL_TIME
        fi
        if [[ "$mode" == unexpected ]]; then export MOCK_RESPONSE='Submitted something unclear'; fi
        if [[ "$mode" == dry ]]; then
            bash "$SOURCE_ROOT/comparison/submit_full.sh" "$SCRATCH/parent" --dry-run
        else
            bash "$SOURCE_ROOT/comparison/submit_full.sh" "$SCRATCH/parent"
        fi
    ) > "$SCRATCH/$name.out" 2>&1 || status=$?
    [[ "$status" == "$expected" ]] || { cat "$SCRATCH/$name.out" >&2; fail "$name expected $expected, got $status"; }
    contains "$SCRATCH/$name.out" 'Command: sbatch --parsable'
    if [[ "$mode" == dry ]]; then
        [[ ! -s "$SCRATCH/sbatch.log" && ! -s "$SCRATCH/scontrol.log" ]] || fail 'Dry run invoked Slurm'
        contains "$SCRATCH/$name.out" 'Dry run: no job submitted.'
        return
    fi
    count_is "$SCRATCH/sbatch.log" 'CALL' 1
    absent "$SCRATCH/sbatch.log" 'LEAK:'
    for arg in --parsable "--account=$account" "--qos=$qos" --partition=gpu-l40s \
        --gres=gpu:l40s:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G "--time=$time_limit"; do
        contains "$SCRATCH/sbatch.log" "ARG:$arg"
    done
    for arg in --hold --exclude --nodelist --constraint --test-only --dependency --reservation; do
        absent "$SCRATCH/sbatch.log" "ARG:$arg"
    done
    count_is "$SCRATCH/sbatch.log" 'ARG:' 12
    if [[ "$mode" == unexpected ]]; then
        [[ ! -s "$SCRATCH/scontrol.log" ]] || fail 'Unexpected response queried controller'
        contains "$SCRATCH/$name.out" 'do not resubmit blindly'
        return
    fi
    count_is "$SCRATCH/scontrol.log" 'CALL:' 1
    contains "$SCRATCH/scontrol.log" 'CALL:--oneliner show job 14509999'
    contains "$SCRATCH/$name.out" 'Submitted batch job 14509999.'
    contains "$SCRATCH/runs/submission-14509999.txt" 'job_id=14509999'
    contains "$SCRATCH/runs/submission-14509999.txt" 'host_policy=any_l40s'
    contains "$SCRATCH/runs/submission-14509999.txt" 'requested_exclusions=none'
    [[ ! -e "$SCRATCH/runs/full-14509999" ]] || fail 'Helper created allocation directory'
    if [[ "$mode" == controller_failure ]]; then
        contains "$SCRATCH/$name.out" 'submitted successfully'
        contains "$SCRATCH/runs/submission-14509999.txt" 'controller inspection failed'
    else
        contains "$SCRATCH/runs/submission-14509999.txt" 'ExcNodeList=site-policy-node'
        contains "$SCRATCH/$name.out" 'JobState=RUNNING'
    fi
}
run_helper dry dry
run_helper valid
run_helper defaults defaults
run_helper controller_failure controller_failure
run_helper unexpected unexpected 4
# Direct submission has fixed L40S defaults and no host-based restrictions.
for arg in '--partition=gpu-l40s' '--nodes=1' '--ntasks=1' '--cpus-per-task=4' '--gres=gpu:l40s:1' '--mem=64G' '--time=08:00:00'; do
    contains "$SOURCE_ROOT/comparison/pace_full.sbatch" "#SBATCH $arg"
done
for restriction in '#SBATCH --exclude' '#SBATCH --nodelist' '#SBATCH --constraint' 'EXCLUDE_NODES' 'REQUIRED_EXCLUDE_NODE' 'show hostnames'; do
    absent "$SOURCE_ROOT/comparison/pace_full.sbatch" "$restriction"
done
printf 'PASS: single unrestricted L40S submission, sanitized inputs and read-only receipts\n'
run_allocation() {
    local id="$1" node="$2" cuda_status="$3" expected="$4" status=0
    : > "$SCRATCH/work.log"
    (
        export EXCLUDE_NODES="$node" SLURM_JOB_ID="$id" SLURMD_NODENAME="$node"
        export MOCK_CUDA_STATUS="$cuda_status" MOCK_MODE=controller_failure
        bash "$SOURCE_ROOT/comparison/pace_full.sbatch" "$SCRATCH/parent"
    ) > "$SCRATCH/$id.out" 2>&1 || status=$?
    [[ "$status" == "$expected" ]] || { cat "$SCRATCH/$id.out" >&2; fail "Allocation $id expected $expected, got $status"; }
    contains "$SCRATCH/runs/full-$id/slurm_status.txt" "exit_code=$expected"
    contains "$SCRATCH/runs/full-$id/allocation.txt" "assigned_node=$node"
    contains "$SCRATCH/runs/full-$id/allocation.txt" 'host_policy=any_l40s'
}
run_allocation 101 atl1-1-01-004-31-0 23 23
contains "$SCRATCH/runs/full-101/slurm_status.txt" 'last_phase=gpu_health'
absent "$SCRATCH/runs/full-101/phase_timing.csv" 'database_staging'
absent "$SCRATCH/work.log" 'python'
for statement in 'torch.arange(' 'device="cuda"' 'total = (values + 1).sum()' \
    'torch.cuda.synchronize()' 'torch.isfinite(total).item()' 'total.item() != count * (count + 1) / 2'; do
    contains "$SCRATCH/gpu-smoke.py.txt" "$statement"
done
# Missing fixture databases deliberately stop after entering staging.
for node in atl1-1-01-004-31-0 another-l40s; do
    id=102
    [[ "$node" == another-l40s ]] && id=103
    run_allocation "$id" "$node" 0 1
    contains "$SCRATCH/runs/full-$id/slurm_status.txt" 'last_phase=database_staging'
    contains "$SCRATCH/work.log" 'python'
    [[ ! -e "$SCRATCH/node-tmp/icemix-full-$id" ]] || fail 'Unexpected fixture database copy'
done
printf 'PASS: any assigned host, CUDA failure ordering and mocked staging readiness\n'
printf 'All full-launcher shell regressions passed (mocked; no GPU validation).\n'
