"""Test: can function pointer types take variadic args?"""

from std.memory import UnsafePointer

comptime PtrBF16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


def kernel_int(a: Int, b: Int, c: Int, d: Int, e: Int, f: Int):
    print("kernel a=" + String(a))

def kernel_ptr(a: PtrBF16, b: PtrBF16, c: PtrBF16, d: Int, e: Int, f: Int):
    print("kernel d=" + String(d))

def kernel_var(*args: Int):
    print("variadic kernel")


# Can a fn ptr type have variadic args?
def take_variadic_fn(f: def(*args: Int) -> None):
    print("got it")

# What about non-variadic fn ptr with concrete types?
def take_fn(f: def(Int, Int, Int, Int, Int, Int) -> None):
    var ptr = UnsafePointer(to=f).bitcast[Int]()[]
    print("ptr=" + String(ptr))


def main():
    take_fn(kernel_int)
    take_variadic_fn(kernel_var)
