from memory import UnsafePointer, alloc
from iter import Iterator

@fieldwise_init
struct HeapMoveArray[T: Movable](Movable, Indexer):
    var ptr: UnsafePointer[Self.T, MutAnyOrigin]
    var capacity: Int
    var len: Int

    fn __init__(out self, capacity: Int):
        self.ptr = alloc[Self.T](capacity)
        self.capacity = capacity
        self.len = 0

    fn __del__(deinit self):
        for i in range(self.len):
            (self.ptr + i).destroy_pointee()
        self.ptr.free()

    fn push(mut self, var value: Self.T):
        debug_assert(self.len < self.capacity, "Attempted to index outside of MoveArray's bounds: push")
        (self.ptr + self.len).init_pointee_move(value^)
        self.len += 1

    fn __getitem__(mut self, idx: Int) -> UnsafePointer[Self.T, MutAnyOrigin]:
        debug_assert(idx >= 0 and idx < self.len, "Attempted to index outside of MoveArray's bounds: __getitem__")
        return self.ptr + idx

    fn __len__(self) -> Int:
        return self.len

    fn __index__(self) -> Int:
        return self.capacity

    fn __mlir_index__(self) -> __mlir_type.index:
        return self.capacity.__mlir_index__()

    fn __iter__(mut self) -> MoveOnlyArrayIter[Self.T]:
        return MoveOnlyArrayIter[Self.T](self.ptr, self.len)

struct MoveOnlyArrayIter[T: Movable](Iterator):
    alias Element = UnsafePointer[Self.T, MutAnyOrigin]
    var ptr: UnsafePointer[Self.T, MutAnyOrigin]
    var index: Int
    var len: Int

    fn __init__(out self, ptr: UnsafePointer[Self.T, MutAnyOrigin], len: Int):
        self.ptr = ptr
        self.index = 0
        self.len = len

    fn __next__(mut self) -> Self.Element:
        var p = self.ptr + self.index
        self.index += 1
        return p

    fn __has_next__(self) -> Bool:
        return self.index < self.len
