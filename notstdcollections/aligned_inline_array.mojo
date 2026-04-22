from std.sys.info import size_of
from std.collections import InlineArray


struct AlignedInlineArray[ElementType: Copyable, count: Int, alignment: Int = 64]:
    comptime PAD = (Self.alignment - 1) // size_of[Self.ElementType]() + 1
    comptime TOTAL = Self.count + Self.PAD
    var storage: InlineArray[Self.ElementType, Self.TOTAL]

    @always_inline
    def __init__(out self, *, uninitialized: Bool):
        self.storage = InlineArray[Self.ElementType, Self.TOTAL](
            uninitialized=True)

    @always_inline
    def __init__(out self, *, fill: Self.ElementType):
        self.storage = InlineArray[Self.ElementType, Self.TOTAL](fill=fill)

    @always_inline
    def unsafe_ptr(ref self) -> UnsafePointer[Self.ElementType, MutAnyOrigin]:
        var raw = Int(UnsafePointer(to=self.storage[0]))
        var aligned = (raw + Self.alignment - 1) & ~(Self.alignment - 1)
        return UnsafePointer[Self.ElementType, MutAnyOrigin](
            unsafe_from_address=aligned)
