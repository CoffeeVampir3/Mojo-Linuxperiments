"""bf16->f32 cast codegen comparison: compiler vs manual bit-shift."""

from memory import UnsafePointer, alloc
from sys.info import simd_width_of
from time import perf_counter_ns
from benchmark import keep


# --- Bad: compiler-generated cast (scalar extract/insert) ---
@no_inline
fn cast_bad(ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]) -> SIMD[DType.float32, 8]:
    return ptr.load[width=8]().cast[DType.float32]()


# --- Good: manual bit-shift (zero-extend + shift left 16) ---
@no_inline
fn cast_good(ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]) -> SIMD[DType.float32, 8]:
    var raw = ptr.bitcast[Scalar[DType.uint16]]().load[width=8]()
    var wide = raw.cast[DType.uint32]()
    var shifted = wide << 16
    var tmp = UnsafePointer(to=shifted)
    return tmp.bitcast[Scalar[DType.float32]]().load[width=8]()


# --- SIMD-in-register: compiler cast on a SIMD value, no pointer ---
@no_inline
fn cast_bad_simd(v: SIMD[DType.bfloat16, 8]) -> SIMD[DType.float32, 8]:
    return v.cast[DType.float32]()


# --- SIMD-in-register: manual shift on a SIMD value, no pointer ---
@no_inline
fn cast_good_simd(v: SIMD[DType.bfloat16, 8]) -> SIMD[DType.float32, 8]:
    var raw = UnsafePointer(to=v).bitcast[Scalar[DType.uint16]]().load[width=8]()
    var wide = raw.cast[DType.uint32]()
    var shifted = wide << 16
    var tmp = UnsafePointer(to=shifted)
    return tmp.bitcast[Scalar[DType.float32]]().load[width=8]()


fn main():
    # Allocate a small buffer with real bf16 values
    var buf = alloc[Scalar[DType.bfloat16]](8)
    for i in range(8):
        buf[i] = Scalar[DType.bfloat16](1.0 + Float64(i))

    var any_buf = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(buf)
    )

    # Correctness check
    var a = cast_bad(any_buf)
    var b = cast_good(any_buf)
    var v = any_buf.load[width=8]()
    var c = cast_bad_simd(v)
    var d = cast_good_simd(v)
    print("cast_bad       (ptr):", a)
    print("cast_good      (ptr):", b)
    print("cast_bad_simd  (reg):", c)
    print("cast_good_simd (reg):", d)

    comptime ITERS = 100_000_000

    # Warm up
    for _ in range(1000):
        a = cast_bad(any_buf)
        keep(a)
        b = cast_good(any_buf)
        keep(b)
        c = cast_bad_simd(v)
        keep(c)
        d = cast_good_simd(v)
        keep(d)

    # Bench cast_bad (ptr)
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        a = cast_bad(any_buf)
        keep(a)
    var t_bad = perf_counter_ns() - t0

    # Bench cast_good (ptr)
    var t1 = perf_counter_ns()
    for _ in range(ITERS):
        b = cast_good(any_buf)
        keep(b)
    var t_good = perf_counter_ns() - t1

    # Bench cast_bad_simd (register)
    var t2 = perf_counter_ns()
    for _ in range(ITERS):
        c = cast_bad_simd(v)
        keep(c)
    var t_bad_s = perf_counter_ns() - t2

    # Bench cast_good_simd (register)
    var t3 = perf_counter_ns()
    for _ in range(ITERS):
        d = cast_good_simd(v)
        keep(d)
    var t_good_s = perf_counter_ns() - t3

    print()
    print("---", ITERS, "iterations ---")
    print("cast_bad       (ptr, .cast):  ", Int(t_bad / 1_000_000), "ms")
    print("cast_good      (ptr, shift):  ", Int(t_good / 1_000_000), "ms")
    print("cast_bad_simd  (reg, .cast):  ", Int(t_bad_s / 1_000_000), "ms")
    print("cast_good_simd (reg, shift):  ", Int(t_good_s / 1_000_000), "ms")

    buf.free()
