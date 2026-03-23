from std.memory import UnsafePointer, alloc


struct HeapMoveArray[T: Movable & ImplicitlyDestructible](Movable, Sized):
    var ptr: UnsafePointer[Self.T, MutAnyOrigin]
    var capacity: Int
    var len: Int

    def __init__(out self, capacity: Int):
        self.ptr = alloc[Self.T](capacity)
        self.capacity = capacity
        self.len = 0

    def __del__(deinit self):
        for i in range(self.len):
            (self.ptr + i).destroy_pointee()
        self.ptr.free()

    def push(mut self, var value: Self.T):
        debug_assert(self.len < self.capacity, "push: out of bounds")
        (self.ptr + self.len).init_pointee_move(value^)
        self.len += 1

    def __getitem__(ref self, idx: Int) -> ref [self] Self.T:
        debug_assert(idx >= 0 and idx < self.len, "getitem: out of bounds")
        return (self.ptr + idx)[]

    def __len__(self) -> Int:
        return self.len
