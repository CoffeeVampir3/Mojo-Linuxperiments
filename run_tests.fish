#!/usr/bin/env fish
env MOJO_ENABLE_RUNTIME=0 pixi run mojo -I . test_burst_stress.mojo
env MOJO_ENABLE_RUNTIME=0 pixi run mojo -I . test_loader.mojo
