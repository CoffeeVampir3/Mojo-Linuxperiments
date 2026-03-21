# Direct test: can reflection see comptime members at all?

from std.reflection import (
    struct_field_count, struct_field_names,
    get_type_name, struct_field_types,
)


struct HasComptime:
    comptime A = 42
    comptime B = "hello"
    comptime C = DType.bfloat16

struct HasVar:
    var a: Int
    var b: String

struct HasBoth:
    comptime X = 99
    var y: Int


fn inspect[T: AnyType]():
    comptime count = struct_field_count[T]()
    comptime names = struct_field_names[T]()
    print(get_type_name[T](), "-> field count:", count)

    @parameter
    for idx in range(count):
        comptime name = names[idx]
        print("  ", name)


fn main():
    inspect[HasComptime]()
    inspect[HasVar]()
    inspect[HasBoth]()
