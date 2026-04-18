"""Full pipeline test through butterquant's plan_quantization + build_header."""

from std.pathlib import Path

from safetensors.parser import parse_safetensors_header, SafetensorsHeader
from modeling.gemma_4_moe_butterquant_tp import Gemma4ButterQuant
from modeling.loader import discover_shards
from quant.butterquant import build_header, plan_quantization


def main():
    var source_dir = Path("checkpoints/gemma-4-26B-A4B")
    var shard_paths = discover_shards(source_dir)
    print("shards: " + String(len(shard_paths)))

    var headers = List[SafetensorsHeader]()
    for i in range(len(shard_paths)):
        var h = parse_safetensors_header(shard_paths[i])
        if not h:
            print("failed to parse " + String(shard_paths[i]))
            return
        headers.append(h.take())

    var tasks = Gemma4ButterQuant[1].build_quantizer_tasks()
    print("tasks: " + String(len(tasks)))

    print("plan_quantization...")
    var bundle_opt = plan_quantization(tasks, headers)
    if not bundle_opt:
        print("planning failed")
        return
    var bundle = bundle_opt.take()
    print("entries: " + String(len(bundle.entries))
        + ", plans: " + String(len(bundle.plans)))

    print("build_header...")
    var header_bytes = build_header(bundle.entries)
    print("header bytes: " + String(len(header_bytes)))

    var corrupt = False
    for i in range(8, len(header_bytes) - 1):
        if header_bytes[i] == UInt8(ord('"')) and header_bytes[i + 1] == UInt8(ord('"')):
            if i + 2 < len(header_bytes) and header_bytes[i + 2] != UInt8(ord(":")):
                print("CORRUPT at byte " + String(i))
                corrupt = True
                break
    if not corrupt:
        print("OK — no corruption detected")
