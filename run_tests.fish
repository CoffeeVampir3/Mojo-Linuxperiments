#!/usr/bin/env fish

pixi run mojo build -I . -debug-level=line-tables test_burst_stress.mojo 2>&1 | head -20
./test_burst_stress
