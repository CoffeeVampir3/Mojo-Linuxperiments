comptime LAYER_NONE = -1
comptime AXIS_NONE = -1
comptime AXIS_HOST = -2


fn shape_1d(d0: Int) -> List[Int]:
    var out = List[Int]()
    out.append(d0)
    return out^


fn shape_2d(d0: Int, d1: Int) -> List[Int]:
    var out = List[Int]()
    out.append(d0)
    out.append(d1)
    return out^


@always_inline
fn align_up_int(value: Int, alignment: Int) -> Int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


@always_inline
fn ceil_div_int(n: Int, d: Int) -> Int:
    if d <= 0:
        return 0
    return (n + d - 1) // d


fn dtype_bytes(dtype: DType) -> Int:
    if dtype == DType.bfloat16:
        return 2
    if dtype == DType.float16:
        return 2
    if dtype == DType.float32:
        return 4
    if dtype == DType.float64:
        return 8
    if dtype == DType.int8 or dtype == DType.uint8 or dtype == DType.bool:
        return 1
    if dtype == DType.int16 or dtype == DType.uint16:
        return 2
    if dtype == DType.int32 or dtype == DType.uint32:
        return 4
    if dtype == DType.int64 or dtype == DType.uint64:
        return 8
    return 0


fn validate_dtype(dtype: DType, expected: DType) -> Bool:
    return dtype == expected


fn validate_shape_exact(expected: List[Int], got: List[Int]) -> Bool:
    if len(expected) != len(got):
        return False
    for i in range(len(expected)):
        if expected[i] != got[i]:
            return False
    return True


fn shape_numel(shape: List[Int]) -> Int:
    var total = 1
    for i in range(len(shape)):
        total *= shape[i]
    return total


fn shape_num_bytes(shape: List[Int], dtype: DType) -> Int:
    var elem_bytes = dtype_bytes(dtype)
    if elem_bytes <= 0:
        return 0
    return shape_numel(shape) * elem_bytes


@fieldwise_init
struct StaticTensorSpec(Copyable, Writable):
    var slot_id: UInt16
    var slot_name: String
    var layer_idx: Int
    var tensor_name: String
    var expected_shape: List[Int]
    var dtype: DType
    var shard_axis: Int


trait StaticModelDescriptor:
    comptime TP: Int
    comptime NODE_COUNT: Int
    comptime NUM_LAYERS: Int

    @staticmethod
    fn model_id() -> String:
        ...

    @staticmethod
    fn node_ids() -> List[Int]:
        ...

    @staticmethod
    fn host_node_index() -> Int:
        ...

    @staticmethod
    fn tensor_specs() -> List[StaticTensorSpec]:
        ...

    @staticmethod
    fn state_bytes_per_node(alignment: Int) -> Int:
        ...
