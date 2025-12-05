from memory import UnsafePointer, memcpy
from collections import InlineArray

struct CpuMask[size: Int = 128](Copyable, Movable):
    var bytes: InlineArray[UInt8, Self.size]

    fn __init__(out self):
        self.bytes = InlineArray[UInt8, Self.size](fill=0)

    fn set(mut self, cpu_id: Int):
        var byte_idx = cpu_id >> 3
        var bit_idx = cpu_id & 7
        if byte_idx < Self.size:
            self.bytes[byte_idx] |= (1 << bit_idx)

    fn clear(mut self, cpu_id: Int):
        var byte_idx = cpu_id >> 3
        var bit_idx = cpu_id & 7
        if byte_idx < Self.size:
            self.bytes[byte_idx] &= ~(1 << bit_idx)

    fn test(self, cpu_id: Int) -> Bool:
        var byte_idx = cpu_id >> 3
        var bit_idx = cpu_id & 7
        if byte_idx >= Self.size:
            return False
        return (self.bytes[byte_idx] & (1 << bit_idx)) != 0

    fn clear_all(mut self):
        for i in range(Self.size):
            self.bytes[i] = 0

    fn set_all(mut self):
        for i in range(Self.size):
            self.bytes[i] = 0xFF

    fn count(self) -> Int:
        var total = 0
        for i in range(Self.size):
            var b = self.bytes[i]
            while b != 0:
                total += Int(b & 1)
                b >>= 1
        return total

    fn ptr(self) -> UnsafePointer[UInt8, MutOrigin.external]:
        return UnsafePointer[UInt8, MutOrigin.external](unsafe_from_address=Int(UnsafePointer(to=self.bytes)))

    fn copy_to(self, dest: UnsafePointer[UInt8, MutOrigin.external]):
        memcpy(dest=dest, src=self.ptr(), count=Self.size)

    @staticmethod
    fn from_cpu_list(cpu_ids: List[Int]) -> Self:
        var mask = Self()
        for i in range(len(cpu_ids)):
            mask.set(cpu_ids[i])
        return mask^

    @staticmethod
    fn byte_size() -> Int:
        return Self.size
