#!/usr/bin/env fish
# Run the dispatch-latency matrix: 4 variants, each with its own perf stat.

set REMOTE_USER blackroot
set REMOTE_HOST 192.168.50.93
set REMOTE_PATH /home/blackroot/Desktop/linuxperiments
set TARGET bench_dispatch_latency.mojo
set BINARY bench_dispatch_latency
set SSH ssh -F /dev/null
set RSYNC_SSH "ssh -F /dev/null"

set VARIANTS inline_remote inline_local dispatch_seq dispatch_batched

# Same perf event list as bench_m27_matrix.fish (minus stalls_mem_any which
# the remote's perf rejects).
set PERF_EVENTS instructions cycles
set -a PERF_EVENTS assists.fp assists.sse_avx_mix machine_clears.count
set -a PERF_EVENTS ld_blocks.store_forward mem_inst_retired.split_loads dtlb_load_misses.walk_completed
set -a PERF_EVENTS mem_load_retired.l1_hit mem_load_retired.l1_miss mem_load_retired.fb_hit
set -a PERF_EVENTS l1d_pend_miss.pending l1d_pend_miss.pending_cycles l1d_pend_miss.fb_full
set -a PERF_EVENTS mem_load_retired.l2_hit mem_load_retired.l2_miss
set -a PERF_EVENTS l2_rqsts.all_demand_data_rd l2_rqsts.all_demand_miss
set -a PERF_EVENTS mem_load_retired.l3_hit mem_load_retired.l3_miss
set -a PERF_EVENTS cycle_activity.stalls_l3_miss
set -a PERF_EVENTS mem_load_l3_miss_retired.local_dram mem_load_l3_miss_retired.remote_dram
set -a PERF_EVENTS mem_load_l3_miss_retired.remote_fwd mem_load_l3_miss_retired.remote_hitm
set -a PERF_EVENTS ocr.reads_to_core.local_dram ocr.reads_to_core.remote_dram
set PERF_EVENTS_CSV (string join , $PERF_EVENTS)

if not test -f $TARGET
    echo "Target not found: $TARGET"
    exit 1
end

rsync -av \
    -e "$RSYNC_SSH" \
    --exclude='.*' \
    --exclude='pixi.lock' \
    --exclude='__pycache__' \
    --exclude='validation/.venv' \
    --exclude='checkpoints/**/*.safetensors' \
    . \
    $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/ > /dev/null

echo "✓ Synced to $REMOTE_HOST"
echo "→ Building $TARGET"
$SSH $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_PATH && env MOJO_ENABLE_RUNTIME=0 pixi run mojo build -I . -D ASSERT=all $TARGET" \
    || begin
        echo "build failed"
        exit 1
    end
echo "✓ Built $BINARY"
echo ""

for variant in $VARIANTS
    echo "=============================================================="
    echo "==  variant=$variant"
    echo "=============================================================="
    $SSH $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_PATH && perf stat -e $PERF_EVENTS_CSV ./$BINARY $variant"
    echo ""
end
