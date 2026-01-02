"""System entrypoint - attempts conditional export."""

from sys import CompilationTarget
from .linux_sys import *
from .x86_64_impl import X86_64LinuxSys

# Attempt: define System inside @parameter if, hope it exports
@parameter
if CompilationTarget.is_x86():
    comptime System = X86_64LinuxSys
