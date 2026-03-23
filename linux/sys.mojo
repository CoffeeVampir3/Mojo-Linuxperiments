from std.sys import CompilationTarget
from std.sys.info import is_64bit

from .linux_sys import *
from .x86_64_impl import X86_64LinuxSys

comptime IS_X86_64: Bool = is_64bit() and CompilationTarget.is_x86()


def linux_sys() -> X86_64LinuxSys:
    comptime assert IS_X86_64, "linux.sys: unsupported target (only x86_64 implemented)"
    return X86_64LinuxSys()
