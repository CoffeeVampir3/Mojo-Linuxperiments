"""Run ButterQuant quantization on MiniMax M2.7."""

from std.pathlib import Path
from std.time import perf_counter_ns
from std.memory import UnsafePointer
from modeling.minimax_m27_moe_butterquant_tp import MiniMaxM27ButterQuant
from quant.butterquant import (
    Planner, Executor, parse_source_headers,
    build_header, RingIO, GammaCache, PtrU8,
)
from modeling.loader import discover_shards
from linux.io_uring import ReadWriteMode


def main():
    var source_dir = Path("checkpoints/Minimax-M2.7")
    var output = Path("quantized_models/minimax_m27_butterquant.safetensors")

    print("source: " + String(source_dir))
    print("output: " + String(output))
    print("")

    var t0 = Int(perf_counter_ns())

    var headers_opt = parse_source_headers(source_dir)
    if not headers_opt:
        print("quantization FAILED")
        return
    var headers = headers_opt.take()

    var planner = Planner(headers^)
    if not MiniMaxM27ButterQuant[1].describe_quantization(planner):
        print("quantization FAILED (planning)")
        return
    print("planned " + String(planner.count) + " tasks")

    var header_bytes = build_header(planner.entries)
    var data_start = len(header_bytes)

    var shard_paths = discover_shards(source_dir)
    var rio = RingIO()
    var all_paths = List[Path]()
    for i in range(len(shard_paths)):
        all_paths.append(shard_paths[i])
    all_paths.append(output)
    var output_file_idx = len(shard_paths)
    try:
        _ = rio.ring.register_files[ReadWriteMode](all_paths)
    except:
        print("quantization FAILED (register files)")
        return

    if not rio.write(output_file_idx, 0,
            PtrU8(unsafe_from_address=Int(UnsafePointer(to=header_bytes[0]))),
            len(header_bytes)):
        print("quantization FAILED (write header)")
        return
    print("header: " + String(len(header_bytes)) + " bytes, "
        + String(len(planner.entries)) + " entries")

    var gamma_cache = GammaCache(planner.max_gamma_cols)
    var executor = Executor(planner^, rio^, gamma_cache^,
        data_start, output_file_idx, 2048, 16 * 1024 * 1024)

    if not MiniMaxM27ButterQuant[1].describe_quantization(executor):
        print("\nquantization FAILED")
        return

    var elapsed_ms = (Int(perf_counter_ns()) - t0) // 1_000_000
    print("\nquantize: " + String(executor.num_quantized) + " quantized, "
        + String(executor.num_passthrough) + " passthrough, "
        + String(executor.total_bytes // (1024 * 1024)) + " MB, "
        + String(elapsed_ms) + " ms")
    print("quantization succeeded")
