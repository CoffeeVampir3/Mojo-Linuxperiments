from std.pathlib import Path

from modeling.gemma_4_moe_butterquant_tp import Gemma4ButterQuant
from modeling.loader import discover_shards
from notstdcollections import HeapMoveArray
from quant.butterquant import OutputEntry, plan_task
from safetensors.parser import parse_safetensors_header, SafetensorsHeader


def main():
    var source_dir = Path("checkpoints/gemma-4-26B-A4B")
    var tasks = Gemma4ButterQuant[1].build_quantizer_tasks()
    var shard_paths = discover_shards(source_dir)
    print("tasks: " + String(len(tasks)))
    print("shards: " + String(len(shard_paths)))

    var headers = HeapMoveArray[SafetensorsHeader](len(shard_paths))
    for i in range(len(shard_paths)):
        print("parse shard " + String(i) + ": " + String(shard_paths[i]))
        var h = parse_safetensors_header(shard_paths[i])
        if not h:
            print("parse failed: " + String(shard_paths[i]))
            return
        headers.push(h.take())
        print("parsed shard " + String(i))

    print("planning...")
    var entries = List[OutputEntry]()
    var offset = 0
    for i in range(len(tasks)):
        if i < 5 or i % 50 == 0 or i >= 640:
            print("plan task " + String(i))
        var task = tasks[i].copy()
        var result = plan_task(task, offset, headers, entries)
        if not result:
            print("planning FAILED at task " + String(i))
            return
        var tp = result.value()
        offset = tp[1]
        if i < 5 or i % 50 == 0 or i >= 640:
            print("planned task " + String(i))

    print("loop done")
    print("planning OK: " + String(len(entries)) + " entries")
