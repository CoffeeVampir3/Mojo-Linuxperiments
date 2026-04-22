from std.sys import llvm_intrinsic
from std.memory import UnsafePointer


def set_subnormal_zeroing():
    """Flush subnormal floats to zero (FTZ+DAZ). Eliminates ~100-cycle
    microcode assists when SIMD operations produce or consume subnormals.
    Per-thread setting — call once at thread start."""
    var mxcsr = UInt32(0)
    var ptr = UnsafePointer(to=mxcsr)
    llvm_intrinsic["llvm.x86.sse.stmxcsr", NoneType](ptr)
    mxcsr = mxcsr | (1 << 15) | (1 << 6)
    llvm_intrinsic["llvm.x86.sse.ldmxcsr", NoneType](ptr)


def get_mxcsr() -> UInt32:
    """Read the current MXCSR register value."""
    var mxcsr = UInt32(0)
    var ptr = UnsafePointer(to=mxcsr)
    llvm_intrinsic["llvm.x86.sse.stmxcsr", NoneType](ptr)
    return mxcsr
