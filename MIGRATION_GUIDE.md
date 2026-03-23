# Mojo v0.26.3 Migration Guide

Migration notes for updating this codebase from v0.26.1 to v0.26.3 nightly.

## 1. `fn` → `def` (declarations only)

**All function and method declarations** use `def` instead of `fn`. This is the most widespread change.

```mojo
# Before
fn my_function(x: Int) -> Int:
    return x + 1

# After
def my_function(x: Int) -> Int:
    return x + 1
```

**This applies everywhere — including function type positions:**

```mojo
# Before
comptime KernelFn = fn(Int, Int, Int, Int, Int, Int)
func: fn[S: ShardStrategy, T: Encoding & Shaped] (String, Int) capturing -> None,
pred_scalar: fn(Byte) -> Bool,

# After
comptime KernelFn = def(Int, Int, Int, Int, Int, Int)
func: def[S: ShardStrategy, T: Encoding & Shaped] (String, Int) capturing -> None,
pred_scalar: def(Byte) -> Bool,
```

**Every `fn` becomes `def` — declarations, type expressions, comptime aliases, all of it.**


## 2. `constrained[COND, "MSG"]()` → `comptime assert COND, "MSG"`

```mojo
# Before
constrained[
    InT.COLS == W.COLS,
    "gemm: input K != weight K",
]()

# After
comptime assert InT.COLS == W.COLS, "gemm: input K != weight K"
```

**Be careful with sed:** The old syntax has `]()` at the end which is easy to accidentally strip from other `]()` patterns like `size_of[T]()` or `method[param]()`. Do NOT use a global sed for this — fix each `constrained` call individually or with a precise pattern that matches only `constrained[`.


## 3. `@parameter if` → `comptime if`

```mojo
# Before
@parameter
if count == 0:
    return x

# After
comptime if count == 0:
    return x
```

**`@parameter for` → `comptime for`:**
```mojo
# Before
@parameter
for i in range(N):
    body()

# After
comptime for i in range(N):
    body()
```

**`@parameter` before a nested `def`/`fn` is unchanged** — it marks the closure as a parametric closure:
```mojo
# This stays as-is
@parameter
def do_join(idx: Int, var fence: PoolFence) capturing:
    fence^.join()
```


## 4. Implicit standard library imports → `std.` prefix

```mojo
# Before
from memory import UnsafePointer, memcpy
from collections import InlineArray
from sys import inlined_assembly
from sys.info import size_of, simd_width_of
from pathlib import Path
from os.atomic import Atomic, Consistency
from math import sqrt, align_up
from bit import count_trailing_zeros

# After
from std.memory import UnsafePointer, memcpy
from std.collections import InlineArray
from std.sys import inlined_assembly
from std.sys.info import size_of, simd_width_of
from std.pathlib import Path
from std.os.atomic import Atomic, Consistency
from std.math import sqrt, align_up
from std.bit import count_trailing_zeros
```

**Package-internal imports (`.module`) are unchanged:**
```mojo
# These stay as-is
from .linux_sys import *
from .capabilities import ByteTransformCapability
```


## 5. `UnsafePointer` and `Span` require explicit origins

Bare `UnsafePointer[T]` and `Span[T]` in function signatures no longer infer the origin parameter. Add `_` to let the compiler infer it:

```mojo
# Before
def foo(ptr: UnsafePointer[UInt32]) -> Bool: ...
def bar(data: Span[Byte]) -> Int: ...

# After
def foo(ptr: UnsafePointer[UInt32, _]) -> Bool: ...
def bar(data: Span[Byte, _]) -> Int: ...
```

**Inside struct fields with `MutAnyOrigin`, no change needed** — the origin is already explicit:
```mojo
# Already fine
var pool: UnsafePointer[BurstPool[], MutAnyOrigin]
```

**`Span` in constructor calls also needs it:**
```mojo
# Before
return String(unsafe_from_utf8=Span[Byte](ptr=p, length=n))

# After
return String(unsafe_from_utf8=Span[Byte, _](ptr=p, length=n))
```


## 6. Parameterized types must be explicitly bound

Bare parameterized type names are no longer allowed where a concrete type is needed. Use `[]` to bind default parameters:

```mojo
# Before
var pool: BurstPool
def load_tokenizer(path: Path) -> Optional[BPETokenizer]:

# After — [] binds all defaults
var pool: BurstPool[]
def load_tokenizer(path: Path) -> Optional[BPETokenizer[]]:
```

**`[...]` UNBINDS parameters (opposite of what you want).** Use `[]` for defaults:
```mojo
BurstPool[]     # concrete type with default stack_size and mask_size
BurstPool[...]  # non-concrete, all parameters unbound — use in generic contexts only
```


## 7. String slicing requires `byte=` keyword

```mojo
# Before
var prefix = content[:newline_pos]
var suffix = content[start:]
var middle = content[a:b]

# After
var prefix = content[byte=:newline_pos]
var suffix = content[byte=start:]
var middle = content[byte=a:b]
```


## 8. `atol()` now raises

```mojo
# Before (in a def function)
var n = atol(some_string)

# After — caller must handle raises
def parse_thing() raises -> Int:
    return atol(some_string)

# Or wrap in try
def parse_thing() -> Int:
    try:
        return atol(some_string)
    except:
        return 0
```


## 9. `Optional` is no longer implicitly copyable

```mojo
# Before
var err = validate(...)
if err:
    return err

# After — transfer ownership
var err = validate(...)
if err:
    return err^
```


## Files to migrate (in dependency order)

1. `linux/linux_sys.mojo` — UAPI types, traits (fn→def, UnsafePointer origins)
2. `linux/x86_64_impl.mojo` — arch impl (fn→def, constrained, @parameter if, std imports)
3. `linux/sys.mojo` — dispatch (fn→def, constrained, std imports)
4. `numa/cpumask.mojo` — bitmask (fn→def, std imports)
5. `numa/numa.mojo` — topology (fn→def, string slicing, atol raises, std imports)
6. `numa/arena.mojo` — allocator (fn→def, @parameter if, std imports)
7. `notstdcollections/heap_move_array.mojo` — move array (fn→def, std imports)
8. `threading/burst_threading.mojo` — thread pool (fn→def, constrained, std imports, BurstPool[])
9. `jsontools/parser.mojo` — JSON parser (fn→def, std imports)
10. `safetensors/parser.mojo` — safetensors parser (fn→def, std imports, Span origins)
11. `safetensors/loader.mojo` — io_uring loader (fn→def, constrained, Optional^, std imports)
12. `tokenizer/capabilities.mojo` — traits (fn→def, Span origins)
13. `tokenizer/shared_capabilities.mojo` — utils (fn→def, Span/UnsafePointer origins)
14. `tokenizer/gpt2.mojo` — GPT2 (fn→def, Span/UnsafePointer origins)
15. `tokenizer/deepseek_v3.mojo` — DeepSeek (fn→def, Span/UnsafePointer origins)
16. `tokenizer/auto.mojo` — auto-detect (fn→def)
17. `tokenizer/tokenizer.mojo` — BPE core (fn→def, Span origins, BPETokenizer[])
18. `tokenizer/loader.mojo` — tokenizer loader (fn→def, BPETokenizer[])
19. `tokenizer/unicode_props.mojo` — generated ranges (fn→def)
20. `tokenizer/unicode_psm_props.mojo` — generated ranges (fn→def)
21. `experimental4/model_spec.mojo` — model types (fn→def)
22. `experimental4/kernel_ops.mojo` — kernels (fn→def, BurstPool[], constrained, Span origins)
23. `experimental4/profiler.mojo` — profiler (fn→def)
24. `experimental4/loader.mojo` — weight loader (fn→def, @parameter if → comptime if)
25. `experimental4/smollm2.mojo` — TP=1 model (fn→def, BurstPool[], @parameter for)
26. `experimental4/smollm2_tp.mojo` — parametric TP model (fn→def, BurstPool[], constrained)
27. `test_smollm2.mojo` — test harness (fn→def, std imports)
28. `test_rings.mojo` — ring broadcast tests (fn→def, std imports)
