
### Mojo Linuxperiments

Currently this repo has an initial design for x86-64 linux threading on mojo, without any C interop or binding. This will, naturally, only work on linux systems.

See the test_linux_threading.mojo file as it should contain fairly clear usage patterns. This is not intended to be a production level library but rather a proof of concept design.

### Notable bits:

Minimal NUMA library for querying basic numa topology.
Move-only heap array in nostdcollections
Numa aware arena allocator
Numa aware threading (But not automatic)

### Doing the thing:

Recommended to run tests with line tables to verify mojo doesn't throw memory errors. Should address sanitize at some point.

To run the code:
```
pixi run mojo build -I . -debug-level=line-tables test_linux_threading.mojo
./test_linux_threading
```

Or

```
./run_tests.fish
```