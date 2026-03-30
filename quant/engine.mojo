"""Quantization engine — I/O, buffer management, conforms_to-dispatched processing.

The engine handles safetensors I/O (io_uring), output header construction,
and buffer management. The top-level quantize function iterates
for_each_weight and dispatches via conforms_to on the weight's tag:

    Absorbed     → stash gamma (read norm from source, hold for next group)
    Gamma        → quantize with stashed gamma (absorb + FWHT + int8)
    Quantizable  → quantize without gamma (FWHT + int8)
    otherwise    → passthrough

Block size is a model-level constant, not per-weight.

Usage:
    from quant.engine import quantize
    return quantize[Self, DType.float32, DType.int8, 64](source, output)
"""

from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.pathlib import Path
from std.time import perf_counter_ns
from std.collections import InlineArray

from safetensors.parser import (
    parse_safetensors_header, SafetensorsHeader, TensorMeta,
    HEADER_LEN_BYTES,
)
from linux.io_uring import (
    IoRing, ReadOp, WriteOp, Completion, ReadWriteMode, RingError,
)

from modeling.model_spec import (
    Encoding, Shaped, Placed, Named,
    Quantizable, Gamma, Passthrough, Absorbed,
    WeightIterable, WeightDesc, weight_desc,
)
from .ops import (
    QuantContext,
    hadamard, hadamard_gamma,
)

comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]


# =============================================================================
# Safetensors writer utilities
# =============================================================================


def dtype_to_st_string(dt: DType) -> String:
    if dt == DType.int8: return "I8"
    if dt == DType.float32: return "F32"
    if dt == DType.bfloat16: return "BF16"
    if dt == DType.float16: return "F16"
    if dt == DType.uint8: return "U8"
    if dt == DType.int32: return "I32"
    if dt == DType.int64: return "I64"
    if dt == DType.float64: return "F64"
    return "UNKNOWN"


def dtype_bits(dt: DType) -> Int:
    if dt == DType.int8: return 8
    if dt == DType.float32: return 32
    if dt == DType.float64: return 64
    if dt == DType.bfloat16: return 16
    if dt == DType.float16: return 16
    return 32


@fieldwise_init
struct OutputTensor(Copyable, Movable):
    var name: String
    var dtype: DType
    var shape: List[Int]
    var data_start: Int
    var data_end: Int
    var src_file_offset: Int
    var src_length: Int
    var quantize: Bool
    var is_scale: Bool


def build_header_json(entries: List[OutputTensor]) -> String:
    var s = String("{")
    for i in range(len(entries)):
        if i > 0:
            s += ","
        var e = entries[i].copy()
        s += '"' + e.name + '":{"dtype":"'
        s += dtype_to_st_string(e.dtype) + '","shape":['
        for j in range(len(e.shape)):
            if j > 0: s += ","
            s += String(e.shape[j])
        s += '],"data_offsets":['
        s += String(e.data_start) + "," + String(e.data_end) + "]}"
    s += "}"
    return s^


def write_u64_le(value: UInt64) -> InlineArray[UInt8, 8]:
    var buf = InlineArray[UInt8, 8](fill=0)
    for i in range(8):
        buf[i] = UInt8((value >> UInt64(i * 8)) & 0xFF)
    return buf


# =============================================================================
# Output layout
# =============================================================================


def compute_output_layout(
    weights: List[WeightDesc],
    header: SafetensorsHeader,
    weight_dtype: DType, weight_bits: Int,
    scale_dtype: DType, scale_bits: Int,
) -> List[OutputTensor]:
    var entries = List[OutputTensor]()
    var offset = 0

    for i in range(len(weights)):
        var w = weights[i].copy()
        if w.absorbed:
            continue
        var meta_opt = header.tensors.get(w.name)
        if not meta_opt:
            continue
        var meta = meta_opt.value().copy()

        if w.quantizable:
            var rows = meta.shape[0]
            var cols = meta.shape[1] if len(meta.shape) > 1 else 1
            var weight_bytes = rows * cols * weight_bits // 8
            entries.append(OutputTensor(
                name=w.name,
                dtype=weight_dtype, shape=meta.shape.copy(),
                data_start=offset, data_end=offset + weight_bytes,
                src_file_offset=header.data_offset + meta.start,
                src_length=meta.end - meta.start,
                quantize=True, is_scale=False,
            ))
            offset += weight_bytes
            var sc_bytes = rows * scale_bits // 8
            var shape_list = List[Int]()
            shape_list.append(rows)
            shape_list.append(1)
            entries.append(OutputTensor(
                name=w.name + "_scale",
                dtype=scale_dtype, shape=shape_list^,
                data_start=offset, data_end=offset + sc_bytes,
                src_file_offset=-1, src_length=0,
                quantize=False, is_scale=True,
            ))
            offset += sc_bytes
        else:
            var byte_size = meta.end - meta.start
            entries.append(OutputTensor(
                name=w.name,
                dtype=meta.dtype, shape=meta.shape.copy(),
                data_start=offset, data_end=offset + byte_size,
                src_file_offset=header.data_offset + meta.start,
                src_length=byte_size,
                quantize=False, is_scale=False,
            ))
            offset += byte_size

    return entries^


# =============================================================================
# I/O helpers
# =============================================================================


def submit_write(mut ring: IoRing[], op: WriteOp) raises RingError:
    _ = ring.submit_one(op)
    var completions = ring.wait()
    if len(completions) == 0:
        raise RingError(op.id, -1, "write: no completion")
    if completions[0].result < 0:
        raise RingError(op.id, Int(completions[0].result), "write")


def submit_read(mut ring: IoRing[], op: ReadOp) raises RingError:
    _ = ring.submit_one(op)
    var completions = ring.wait()
    if len(completions) == 0:
        raise RingError(op.id, -1, "read: no completion")
    if completions[0].result < 0:
        raise RingError(op.id, Int(completions[0].result), "read")
    var got = Int(completions[0].result)
    if got != op.expected_bytes():
        raise RingError(op.id, -1, "short read")


def find_entry(entries: List[OutputTensor], name: String) -> Int:
    for i in range(len(entries)):
        if entries[i].name == name:
            return i
    return -1


# =============================================================================
# Quantization engine — buffers + I/O, no trait dispatch
# =============================================================================


@fieldwise_init
struct QuantEngine[P: DType, T: DType](Movable):
    var source_header: SafetensorsHeader
    var entries: List[OutputTensor]
    var data_start: Int
    var read_ring: IoRing[]
    var write_ring: IoRing[]
    var op_id: Int
    var read_buf: PtrU8
    var work: UnsafePointer[Scalar[Self.P], MutAnyOrigin]
    var qi: UnsafePointer[Scalar[Self.T], MutAnyOrigin]
    var scales: UnsafePointer[Scalar[Self.P], MutAnyOrigin]
    var gamma_buf: UnsafePointer[Scalar[Self.P], MutAnyOrigin]
    var max_rows: Int
    var max_cols: Int
    var quant_count: Int
    var quant_ns: Int
    var quant_bytes: Int
    var total_start: UInt

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------

    @staticmethod
    def open[M: WeightIterable](
        source_path: Path, output_path: Path,
    ) -> Optional[Self]:
        var weights = List[WeightDesc]()

        @parameter
        def collect[W: Encoding & Shaped & Placed & Named](
            prefix: String, base: Int,
        ):
            weights.append(weight_desc[W](prefix, base))

        M.for_each_weight[collect]()

        var header_opt = parse_safetensors_header(source_path)
        if not header_opt:
            print("engine: failed to parse source header")
            return None
        var source_header = header_opt.take()

        var target_bits = dtype_bits(Self.T)
        var precision_bits = dtype_bits(Self.P)
        var entries = compute_output_layout(
            weights, source_header,
            weight_dtype=Self.T, weight_bits=target_bits,
            scale_dtype=Self.P, scale_bits=precision_bits,
        )
        print("engine: " + String(len(source_header.tensors))
              + " source tensors -> " + String(len(entries)) + " output entries")

        var json = build_header_json(entries)
        var json_len = len(json)
        var header_prefix = write_u64_le(UInt64(json_len))
        var header_buf_size = HEADER_LEN_BYTES + json_len
        var header_buf = alloc[UInt8](header_buf_size)
        for i in range(HEADER_LEN_BYTES):
            header_buf[i] = header_prefix[i]
        var json_bytes = json.as_bytes()
        memcpy(
            dest=header_buf + HEADER_LEN_BYTES,
            src=json_bytes.unsafe_ptr(), count=json_len,
        )
        var data_start = HEADER_LEN_BYTES + json_len

        var read_ring = IoRing[]()
        var write_ring = IoRing[]()
        if not read_ring or not write_ring:
            print("engine: io_uring setup failed")
            header_buf.free()
            return None

        var paths = List[Path]()
        paths.append(source_path)
        paths.append(output_path)
        try:
            _ = read_ring.register_files[ReadWriteMode](paths)
            _ = write_ring.register_files[ReadWriteMode](paths)
        except:
            print("engine: register_files failed")
            header_buf.free()
            return None

        try:
            submit_write(write_ring, WriteOp(
                file_idx=1, offset=0, length=header_buf_size,
                src=header_buf, id=0,
            ))
        except:
            print("engine: header write failed")
            header_buf.free()
            return None
        header_buf.free()

        var max_src = 0
        var max_rows = 0
        var max_cols = 0
        for i in range(len(entries)):
            if entries[i].src_length > max_src:
                max_src = entries[i].src_length
            if entries[i].quantize:
                var shape = entries[i].shape.copy()
                if len(shape) >= 1 and shape[0] > max_rows: max_rows = shape[0]
                if len(shape) >= 2 and shape[1] > max_cols: max_cols = shape[1]

        print("engine: ready, max tensor "
              + String(max_rows) + "x" + String(max_cols))

        return Self(
            source_header=source_header^,
            entries=entries^,
            data_start=data_start,
            read_ring=read_ring^,
            write_ring=write_ring^,
            op_id=1,
            read_buf=alloc[UInt8](max(max_src, 1)),
            work=alloc[Scalar[Self.P]](max(max_rows * max_cols, 1)),
            qi=alloc[Scalar[Self.T]](max(max_rows * max_cols, 1)),
            scales=alloc[Scalar[Self.P]](max(max_rows, 1)),
            gamma_buf=alloc[Scalar[Self.P]](max(max_cols, 1)),
            max_rows=max_rows,
            max_cols=max_cols,
            quant_count=0,
            quant_ns=0,
            quant_bytes=0,
            total_start=perf_counter_ns(),
        )

    def source_has(self, name: String) -> Bool:
        return Bool(self.source_header.tensors.get(name))

    # -------------------------------------------------------------------------
    # Read source tensor into read_buf
    # -------------------------------------------------------------------------

    def read_source(mut self, name: String) -> Bool:
        var meta_opt = self.source_header.tensors.get(name)
        if not meta_opt:
            print("engine: tensor not found: " + name)
            return False
        var meta = meta_opt.value().copy()
        try:
            submit_read(self.read_ring, ReadOp(
                file_idx=0,
                offset=self.source_header.data_offset + meta.start,
                length=meta.end - meta.start,
                dest=self.read_buf, id=self.op_id,
            ))
            self.op_id += 1
            return True
        except:
            print("engine: read failed for " + name)
            return False

    # -------------------------------------------------------------------------
    # bf16 read_buf → f32 work buffer
    # -------------------------------------------------------------------------

    def load_work(mut self, rows: Int, cols: Int):
        var src = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(self.read_buf),
        )
        for r in range(rows):
            var src_row = src + r * cols
            var dst_row = self.work + r * cols
            var k = 0
            while k + 16 <= cols:
                (dst_row + k).store((src_row + k).load[width=16]().cast[Self.P]())
                k += 16
            while k < cols:
                dst_row[k] = src_row[k].cast[Self.P]()
                k += 1

    # -------------------------------------------------------------------------
    # Write qi + scales to output
    # -------------------------------------------------------------------------

    def write_result(mut self, name: String) -> Bool:
        var wi = find_entry(self.entries, name)
        if wi < 0:
            print("engine: entry not found: " + name)
            return False
        var entry = self.entries[wi].copy()
        try:
            submit_write(self.write_ring, WriteOp(
                file_idx=1,
                offset=self.data_start + entry.data_start,
                length=entry.data_end - entry.data_start,
                src=PtrU8(unsafe_from_address=Int(self.qi)),
                id=self.op_id,
            ))
            self.op_id += 1
        except:
            print("engine: weight write failed: " + name)
            return False
        if wi + 1 >= len(self.entries) or not self.entries[wi + 1].is_scale:
            print("engine: scale entry missing after " + name)
            return False
        var sc = self.entries[wi + 1].copy()
        try:
            submit_write(self.write_ring, WriteOp(
                file_idx=1,
                offset=self.data_start + sc.data_start,
                length=sc.data_end - sc.data_start,
                src=PtrU8(unsafe_from_address=Int(self.scales)),
                id=self.op_id,
            ))
            self.op_id += 1
        except:
            print("engine: scale write failed: " + name)
            return False
        return True

    # -------------------------------------------------------------------------
    # Write passthrough
    # -------------------------------------------------------------------------

    def write_passthrough(mut self, name: String) -> Bool:
        var ei = find_entry(self.entries, name)
        if ei < 0:
            print("engine: passthrough entry not found: " + name)
            return False
        var entry = self.entries[ei].copy()
        try:
            submit_write(self.write_ring, WriteOp(
                file_idx=1,
                offset=self.data_start + entry.data_start,
                length=entry.data_end - entry.data_start,
                src=self.read_buf, id=self.op_id,
            ))
            self.op_id += 1
        except:
            print("engine: passthrough write failed: " + name)
            return False
        return True

    # -------------------------------------------------------------------------
    # Stash gamma: read bf16 norm from source → gamma_buf in precision P
    # -------------------------------------------------------------------------

    def stash_gamma(mut self, name: String, count: Int) -> Bool:
        if not self.read_source(name): return False
        var src = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(self.read_buf),
        )
        for i in range(count):
            self.gamma_buf[i] = src[i].cast[Self.P]()
        print("  absorb     " + name)
        return True

    # -------------------------------------------------------------------------
    # Quantize: read + FWHT + int8, with or without gamma
    # -------------------------------------------------------------------------

    def quantize_weight[block: Int](
        mut self, name: String, rows: Int, cols: Int, use_gamma: Bool,
    ) -> Bool:
        var t0 = perf_counter_ns()
        if not self.read_source(name): return False
        self.load_work(rows, cols)
        var ctx = QuantContext[Self.P, Self.T](
            self.work, self.qi, self.scales, rows, cols,
        )
        if use_gamma:
            hadamard_gamma[Self.P, Self.T, block](ctx, self.gamma_buf)
        else:
            hadamard[Self.P, Self.T, block](ctx)
        if not self.write_result(name): return False
        var ns = Int(perf_counter_ns() - t0)
        self.quant_count += 1
        self.quant_ns += ns
        self.quant_bytes += rows * cols * 2
        var tag = " (gamma)" if use_gamma else ""
        print("  quantize   " + name
              + " [" + String(rows) + "x" + String(cols) + "]"
              + tag + " " + String(ns // 1_000) + "us")
        return True

    # -------------------------------------------------------------------------
    # Passthrough: read + write unchanged
    # -------------------------------------------------------------------------

    def passthrough(mut self, name: String, rows: Int, cols: Int) -> Bool:
        if not self.read_source(name): return False
        if not self.write_passthrough(name): return False
        print("  passthrough " + name)
        return True

    # -------------------------------------------------------------------------
    # Finalize
    # -------------------------------------------------------------------------

    def finalize(mut self) -> Bool:
        var total_ns = Int(perf_counter_ns() - self.total_start)
        var expected = 0
        for i in range(len(self.entries)):
            if self.entries[i].quantize: expected += 1
        if self.quant_count != expected:
            print("engine: WARNING " + String(self.quant_count)
                  + "/" + String(expected) + " tensors quantized")
        var avg_us = self.quant_ns // max(self.quant_count, 1) // 1_000
        var throughput = self.quant_bytes * 1_000 // max(self.quant_ns, 1)
        print("engine: " + String(self.quant_count) + " quantized"
              + ", avg " + String(avg_us) + "us"
              + ", " + String(throughput) + " MB/s"
              + ", total " + String(total_ns // 1_000_000) + "ms")
        return True

    def free(mut self):
        self.read_buf.free()
        self.work.free()
        self.qi.free()
        self.scales.free()
        self.gamma_buf.free()


# =============================================================================
# Top-level quantize — one-liner for the model
# =============================================================================


def quantize[M: WeightIterable, P: DType, T: DType, block: Int](
    source_path: Path, output_path: Path,
) -> Bool:
    """Quantize a model. Slot tags drive dispatch automatically.

    Absorbed  → stash gamma
    Gamma     → FWHT + gamma absorption + int8
    Quantizable → FWHT + int8
    otherwise → passthrough
    """
    var engine_opt = QuantEngine[P, T].open[M](source_path, output_path)
    if not engine_opt:
        return False
    var engine = engine_opt.take()
    var ok = True

    @parameter
    def each[W: Encoding & Shaped & Placed & Named](
        prefix: String, base: Int,
    ):
        if not ok: return
        var name = prefix + String(W.NAME)
        comptime if conforms_to(W, Absorbed):
            ok = engine.stash_gamma(name, W.ROWS * W.COLS)
        elif conforms_to(W, Gamma):
            ok = engine.quantize_weight[block](name, W.ROWS, W.COLS, True)
        elif conforms_to(W, Quantizable):
            ok = engine.quantize_weight[block](name, W.ROWS, W.COLS, False)
        else:
            if engine.source_has(name):
                ok = engine.passthrough(name, W.ROWS, W.COLS)

    M.for_each_weight[each]()
    if ok:
        ok = engine.finalize()
    engine.free()
    return ok
