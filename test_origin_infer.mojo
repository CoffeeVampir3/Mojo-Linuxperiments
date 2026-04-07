"""Test origin-polymorphic dispatch through a trait method."""

from std.memory import UnsafePointer


@fieldwise_init
struct MyArgs(Copyable, ImplicitlyCopyable):
    var a: Int
    var b: Int


def my_kernel(args: MyArgs):
    print("  kernel got a=" + String(args.a) + " b=" + String(args.b))


# --- Approach 1: origin: MutOrigin, no // ---

trait Pool1:
    def dispatch[Args: Copyable & ImplicitlyCopyable,
        kernel: def (Args) -> None, origin: MutOrigin](
        mut self, args: UnsafePointer[Args, origin], n: Int): ...

struct FakePool1(Pool1):
    def __init__(out self): pass
    def dispatch[Args: Copyable & ImplicitlyCopyable,
        kernel: def (Args) -> None, origin: MutOrigin](
        mut self, args: UnsafePointer[Args, origin], n: Int):
        for i in range(n):
            kernel((args + i)[])


# --- Approach 2: is_mutable: Bool before //, origin after ---

trait Pool2:
    def dispatch[is_mutable: Bool, //,
        Args: Copyable & ImplicitlyCopyable,
        kernel: def (Args) -> None,
        origin: Origin[mut=is_mutable]](
        mut self, args: UnsafePointer[Args, origin], n: Int): ...

struct FakePool2(Pool2):
    def __init__(out self): pass
    def dispatch[is_mutable: Bool, //,
        Args: Copyable & ImplicitlyCopyable,
        kernel: def (Args) -> None,
        origin: Origin[mut=is_mutable]](
        mut self, args: UnsafePointer[Args, origin], n: Int):
        for i in range(n):
            kernel((args + i)[])


def use_pool1[P: Pool1](mut pool: P):
    var jobs = InlineArray[MyArgs, 4](fill=MyArgs(0, 0))
    jobs[0] = MyArgs(10, 20)
    jobs[1] = MyArgs(30, 40)
    pool.dispatch[MyArgs, my_kernel](UnsafePointer(to=jobs[0]), 2)


def use_pool2[P: Pool2](mut pool: P):
    var jobs = InlineArray[MyArgs, 4](fill=MyArgs(0, 0))
    jobs[0] = MyArgs(10, 20)
    jobs[1] = MyArgs(30, 40)
    pool.dispatch[MyArgs, my_kernel](UnsafePointer(to=jobs[0]), 2)


def main():
    print("approach 1 (MutOrigin, no //):")
    var p1 = FakePool1()
    use_pool1(p1)

    print("approach 2 (Bool // Origin):")
    var p2 = FakePool2()
    use_pool2(p2)
