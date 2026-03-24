"""Safetensors file loading via io_uring.

Defines ReadOp conforming to IoOp and provides checked queue processing
for bulk safetensors loading. Uses IoRing from linux.io_uring.
"""

from std.collections import Dict
from std.memory import UnsafePointer
from std.pathlib import Path
from linux.io_uring import (
    IoRing, ReadOp, Completion,
    IoRingError, RingError,
)


# =============================================================================
# Load error — wraps any IoRingError for consumers that just want to print
# =============================================================================


struct LoadError(Copyable, Writable):
    var msg: String

    def __init__(out self, msg: String):
        self.msg = msg

    @staticmethod
    def from_ring[E: IoRingError](err: E) -> Self:
        return Self("io_uring: op " + String(err.error_op_id()) + ": " + err.error_message())


# =============================================================================
# Checked queue processing — validates completions for read operations
# =============================================================================


def process_read_queue[
    on_complete: def(Completion) capturing -> None,
](mut ring: IoRing, ops: List[ReadOp[]]) -> Optional[LoadError]:
    """Submit all read ops, drain completions, validate results.
    Returns None on success, LoadError on failure."""
    var total = len(ops)
    if total == 0:
        return None

    if not ring:
        return LoadError("ring not initialized")

    # Build expected/seen maps for validation
    var expected_by_id = Dict[Int, Int]()
    var seen_by_id = Dict[Int, Int]()
    for i in range(total):
        var op = ops[i]
        var expected = op.expected_bytes()
        if expected <= 0:
            return LoadError("invalid op: id=" + String(op.op_id()) + " length=" + String(expected))
        var prior = expected_by_id.get(op.op_id())
        if prior:
            return LoadError("duplicate op id: " + String(op.op_id()))
        expected_by_id[op.op_id()] = expected
        seen_by_id[op.op_id()] = 0

    var submitted = 0
    var completed = 0

    while submitted < total:
        # Submit one op
        var submit_res: Int
        try:
            submit_res = ring.submit_one[ReadOp[]](ops[submitted])
        except err:
            return LoadError.from_ring(err)

        # Queue was full — drain completions first
        if submit_res == 0:
            var completions: List[Completion]
            try:
                completions = ring.wait(min_complete=1)
            except err:
                return LoadError.from_ring(err)
            if len(completions) == 0:
                return LoadError("wait returned no completions")
            var err = _validate_and_deliver[on_complete](completions, expected_by_id, seen_by_id)
            if err:
                return err^
            completed += len(completions)
            continue

        submitted += 1

        # Drain available completions
        var completions: List[Completion]
        try:
            completions = ring.wait(min_complete=1)
        except err:
            return LoadError.from_ring(err)
        if len(completions) == 0:
            return LoadError("wait returned no completions after submit")
        var err = _validate_and_deliver[on_complete](completions, expected_by_id, seen_by_id)
        if err:
            return err^
        completed += len(completions)

    # Drain remaining
    while ring.pending() > 0:
        var completions: List[Completion]
        try:
            completions = ring.wait(min_complete=1)
        except err:
            return LoadError.from_ring(err)
        if len(completions) == 0:
            return LoadError("wait returned no completions during drain")
        var err = _validate_and_deliver[on_complete](completions, expected_by_id, seen_by_id)
        if err:
            return err^
        completed += len(completions)

    if completed != total:
        return LoadError("incomplete: " + String(completed) + "/" + String(total) + " ops")

    return None


def _validate_and_deliver[
    on_complete: def(Completion) capturing -> None,
](
    completions: List[Completion],
    expected_by_id: Dict[Int, Int],
    mut seen_by_id: Dict[Int, Int],
) -> Optional[LoadError]:
    for c in completions:
        var expected_opt = expected_by_id.get(c.id)
        var seen_opt = seen_by_id.get(c.id)
        if not expected_opt or not seen_opt:
            return LoadError("unknown completion id: " + String(c.id))
        if seen_opt.value() != 0:
            return LoadError("duplicate completion id: " + String(c.id))
        seen_by_id[c.id] = 1
        if c.result < 0:
            return LoadError("CQE negative: op " + String(c.id) + " errno=" + String(Int(c.result)))
        var expected = expected_opt.value()
        var got = Int(c.result)
        if got != expected:
            return LoadError("short read: op " + String(c.id) + " got " + String(got) + "/" + String(expected))
        on_complete(c)
    return None


# =============================================================================
# Legacy aliases for existing consumers
# =============================================================================

