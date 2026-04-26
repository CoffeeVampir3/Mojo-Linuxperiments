from std.collections import InlineArray
from std.memory import Span, UnsafePointer


comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


@fieldwise_init
struct F32Rows[origin: MutOrigin](Copyable, ImplicitlyCopyable):
    var rows: Span[Float32, Self.origin]

    @always_inline
    def ptr(self) -> F32Ptr:
        return self.rows.unsafe_ptr()


@always_inline
def first_value(ptr: F32Ptr) -> Float32:
    return ptr[]


def main():
    var values = InlineArray[Float32, 4](fill=Float32(0))
    values[0] = Float32(3)
    values[1] = Float32(1)
    values[2] = Float32(4)
    values[3] = Float32(1)
    var rows = F32Rows(Span(values))
    print(first_value(rows.ptr()))
