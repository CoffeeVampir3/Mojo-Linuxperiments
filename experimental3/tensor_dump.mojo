"""Comptime-gated tensor dumper. Writes raw binary files: {dir}/{label}.bin

When Enabled=False, all tap calls compile to nothing.
"""

from std.pathlib import Path
from std.memory import UnsafePointer
from std.os import makedirs
from std.sys.info import size_of


struct Dumper[Enabled: Bool = False]:
    var dir: String

    def __init__(out self, dir: String):
        self.dir = dir
        comptime if Self.Enabled:
            try:
                makedirs(dir)
            except:
                pass

    def tap[T: AnyType](self, label: String, ptr: UnsafePointer[T, _], count: Int):
        comptime if Self.Enabled:
            var num_bytes = count * size_of[T]()
            var raw = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(ptr))
            var path = self.dir + "/" + label + ".bin"
            try:
                var fh = FileHandle(path, "w")
                fh.write_bytes(Span[UInt8, MutAnyOrigin](ptr=raw, length=num_bytes))
                fh.close()
                print("dump:", label, num_bytes, "bytes")
            except:
                print("Dumper: write failed for", label)
