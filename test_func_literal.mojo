"""Test: can function literals with UnsafePointer params be TrivialRegisterPassable?"""

from std.memory import UnsafePointer
from std.sys.info import size_of


def kernel_int_args(a: Int, b: Int, c: Int, d: Int, e: Int, f: Int):
    pass

def kernel_ptr_args(
    a: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    c: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    d: Int, e: Int, f: Int,
):
    pass

def kernel_4ptr_args(
    a: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    c: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    d: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    e: Int, f: Int,
):
    pass

def kernel_int_params[N: Int](a: Int, b: Int, c: Int, d: Int, e: Int, f: Int):
    pass

def kernel_ptr_params[N: Int](
    a: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    c: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    d: Int, e: Int, f: Int,
):
    pass


def check_trp[F: TrivialRegisterPassable](name: String, f: F):
    print(name, "size_of =", size_of[type_of(f)]())


def main():
    print("=== non-parameterized ===")
    check_trp("int_args", kernel_int_args)
    check_trp("ptr_args", kernel_ptr_args)
    check_trp("4ptr_args", kernel_4ptr_args)

    print("=== parameterized ===")
    check_trp("int_params[42]", kernel_int_params[42])
    check_trp("ptr_params[42]", kernel_ptr_params[42])
