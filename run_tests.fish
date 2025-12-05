#!/usr/bin/env fish

pixi run mojo build -I . -debug-level=line-tables test_linux_threading.mojo
./test_linux_threading
