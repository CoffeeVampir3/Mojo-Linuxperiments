#!/usr/bin/env fish

pixi run mojo build -I . -debug-level=line-tables bench_hot_cold.mojo 2>&1 | head -20
./bench_hot_cold
