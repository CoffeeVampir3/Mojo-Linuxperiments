from std.memory import Pointer, UnsafePointer


trait IntPtrLowerable:
    def lower(self) -> UnsafePointer[Int, MutAnyOrigin]:
        ...


@fieldwise_init
struct StackInt[origin: MutOrigin](Copyable, ImplicitlyCopyable, IntPtrLowerable):
    var value: Pointer[Int, Self.origin]

    @always_inline
    def lower(self) -> UnsafePointer[Int, MutAnyOrigin]:
        return UnsafePointer(to=self.value[])


@always_inline
def read_any_origin(ptr: UnsafePointer[Int, MutAnyOrigin]) -> Int:
    return ptr[]


@always_inline
def read_lowered[T: IntPtrLowerable](x: T) -> Int:
    return read_any_origin(x.lower())


def main():
    var value = 42
    var lowered = StackInt(Pointer(to=value))
    print(read_lowered(lowered))
