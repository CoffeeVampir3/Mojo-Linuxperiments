"""Model-level traits for inference.

Inference trait defines the minimal interface that any model must
satisfy for generation loops, samplers, and other consumer code.
Consumer code depends on the trait, not the concrete model struct.
"""

from experimental3.logits import LogitsView


trait Inference(Movable):
    """A loaded model that can run forward passes and produce logits.
    VOCAB is comptime so LogitsView and samplers can use SIMD-optimized
    access at compile-time-known bounds."""
    comptime VOCAB: Int

    fn forward(
        mut self, tokens_ptr: Int, seq_len: Int, pos: Int,
    ) -> LogitsView[Self.VOCAB]:
        """Run a forward pass over the given token IDs.
        tokens_ptr: address of seq_len int32 token IDs.
        pos: starting position in the KV cache (0 for prefill, current pos for decode).
        Returns a read-only view of logits — valid until the next forward call."""
        ...
