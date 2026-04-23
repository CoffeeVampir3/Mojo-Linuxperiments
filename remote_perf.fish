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
# perf will multiplex — check the [%] column in the output. Ratios among
# events with similar [%] coverage are still meaningful; ratios across
# very different coverage should be treated as indicative only.
#
# Baseline
set PERF_EVENTS instructions cycles
# FP/SIMD assists + pipeline clears
set -a PERF_EVENTS assists.fp assists.sse_avx_mix machine_clears.count
# Memory pipeline stalls
set -a PERF_EVENTS ld_blocks.store_forward mem_inst_retired.split_loads dtlb_load_misses.walk_completed
# L1 dcache + LFB occupancy — LFB saturation is the hidden MLP ceiling.
#   fb_full/cycles high  => MLP-bound, more DRAM BW won't help
#   pending/pending_cycles => avg outstanding L1 misses (vs ~16 LFBs)
set -a PERF_EVENTS mem_load_retired.l1_hit mem_load_retired.l1_miss mem_load_retired.fb_hit
set -a PERF_EVENTS l1d_pend_miss.pending l1d_pend_miss.pending_cycles l1d_pend_miss.fb_full
# L2 demand traffic + demand miss rate.
# (SPR JSON doesn't define l2_rqsts.pf_hit/pf_miss — those were SKX
# umasks. To assess HW prefetcher utility, rerun with the L2 streamer
# disabled via `wrmsr -a 0x1a4 0x1` and compare throughput.)
set -a PERF_EVENTS mem_load_retired.l2_hit mem_load_retired.l2_miss
set -a PERF_EVENTS l2_rqsts.all_demand_data_rd l2_rqsts.all_demand_miss
# L3 + memory stall attribution (TMA-style) — tells you whether stalls
# resolve in L2/L3 or reach DRAM.
set -a PERF_EVENTS mem_load_retired.l3_hit mem_load_retired.l3_miss
set -a PERF_EVENTS cycle_activity.stalls_l3_miss cycle_activity.stalls_total
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