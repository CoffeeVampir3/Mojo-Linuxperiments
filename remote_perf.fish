#!/usr/bin/env fish

set REMOTE_USER blackroot
set REMOTE_HOST 192.168.50.93
set REMOTE_PATH /home/blackroot/Desktop/linuxperiments
set DEFAULT_TARGET amx_ablations.mojo

if test (count $argv) -gt 0
    set TARGET $argv[1]
else
    set TARGET $DEFAULT_TARGET
end

if not test -f $TARGET
    echo "Target not found: $TARGET"
    exit 1
end

set BINARY (string replace -r '\.mojo$' '' (basename $TARGET))

# ---- perf events --------------------------------------------------------
# Note: SPR has 8 GP PMCs + fixed counters. This list exceeds that, so
# perf will multiplex — check the [%] column in the output.

# Baseline
set PERF_EVENTS instructions cycles

# FP/SIMD assists + pipeline clears
set -a PERF_EVENTS assists.fp assists.sse_avx_mix machine_clears.count

# Memory pipeline stalls
set -a PERF_EVENTS ld_blocks.store_forward mem_inst_retired.split_loads dtlb_load_misses.walk_completed

# NUMA: retired-load attribution (where do L3 misses actually land?)
set -a PERF_EVENTS mem_load_l3_miss_retired.local_dram mem_load_l3_miss_retired.remote_dram
set -a PERF_EVENTS mem_load_l3_miss_retired.remote_fwd mem_load_l3_miss_retired.remote_hitm

# NUMA: broader off-core view — catches stores (RFOs) and HW prefetches
set -a PERF_EVENTS ocr.reads_to_core.local_dram ocr.reads_to_core.remote_dram

set PERF_EVENTS_CSV (string join , $PERF_EVENTS)
# -------------------------------------------------------------------------

rsync -av \
    --exclude='.*' \
    --exclude='pixi.lock' \
    --exclude='__pycache__' \
    --exclude='validation/.venv' \
    --exclude='test_smollm2_bin' \
    --exclude='test_smollm2_tp3_bin' \
    --exclude='test_tp3_bin' \
    --exclude='test_tp_bin' \
    --exclude='test_rings_bin' \
    --exclude='fence_experiment_bin' \
    --exclude='tp_param_bin' \
    --include='checkpoints/SmolLM2/model.safetensors' \
    --exclude='checkpoints/**/*.safetensors' \
    . \
    $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/

echo "✓ Synced to $REMOTE_HOST:$REMOTE_PATH"
echo "→ Building $TARGET on $REMOTE_HOST"

ssh $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_PATH && env MOJO_ENABLE_RUNTIME=0 pixi run mojo build -I . -D ASSERT=all $TARGET && echo '=== PERF STAT ===' && perf stat -e $PERF_EVENTS_CSV ./$BINARY"
