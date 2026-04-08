from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from modeling.deepseekv2_lite import (
    DeepSeekV2Lite, DeepSeekV2LiteConfig, ExpertWeights, MoELayer, DSV2Model,
)

comptime C = DeepSeekV2LiteConfig
comptime M = DSV2Model[1]
comptime E = MoELayer[1]
comptime EW = ExpertWeights[1]

def var_at(ptr: Int, n: Int) -> Float32:
    var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=ptr)
    comptime w = simd_width_of[DType.float32]()
    var acc = Float64(0)
    for i in range(0, n, w):
        var v = (p + i).load[width=w]().cast[DType.float64]()
        acc += (v * v).reduce_add()
    return Float32(acc / Float64(n))

def main():
    var model_opt = DeepSeekV2Lite[1].load(Path("checkpoints/deepseekv2-lite"))
    if not model_opt:
        return
    var model = model_opt.take()
    var host = model.rank(0)

    # Check expert weights for layer 1
    var ewb = host.expert_weight_base(1)
    print("expert_weight_base(1):", ewb)
    print("expected:", host.weight_base() + M.MOE_LAYERS_OFF + (1 - C.FIRST_K_DENSE) * M.MOE_STRIDE + E.EXPERTS_OFF)
    print()

    # Check that expert 0 gate_proj has reasonable variance
    var e0_gate = ewb
    var e0_up = ewb + EW.UP_PROJ.OFFSET
    var e0_down = ewb + EW.DOWN_PROJ.OFFSET
    var e1_gate = ewb + EW.STRIDE

    print("expert 0 gate var:", var_at(e0_gate, C.MOE_INTERMEDIATE * C.HIDDEN))
    print("expert 0 up var:", var_at(e0_up, C.MOE_INTERMEDIATE * C.HIDDEN))
    print("expert 0 down var:", var_at(e0_down, C.HIDDEN * C.MOE_INTERMEDIATE))
    print("expert 1 gate var:", var_at(e1_gate, C.MOE_INTERMEDIATE * C.HIDDEN))
    print()

    # Check router weights
    var router = host.moe_layer_weight[E.ROUTER](1)
    print("router var:", var_at(router.ptr, C.N_ROUTED_EXPERTS * C.HIDDEN))

    # Check shared expert weights
    var sg = host.moe_layer_weight[E.SHARED_GATE](1)
    var su = host.moe_layer_weight[E.SHARED_UP](1)
    var sd = host.moe_layer_weight[E.SHARED_DOWN](1)
    print("shared gate var:", var_at(sg.ptr, E.SHARED_GATE.ROWS * E.SHARED_GATE.COLS))
    print("shared up var:", var_at(su.ptr, E.SHARED_UP.ROWS * E.SHARED_UP.COLS))
    print("shared down var:", var_at(sd.ptr, E.SHARED_DOWN.ROWS * E.SHARED_DOWN.COLS))
    print()

    # Spot check: are expert weights non-zero and distinct?
    var p0 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=e0_gate)
    var p1 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=e1_gate)
    print("expert 0 gate[:5]:", Float32(p0[0]), Float32(p0[1]), Float32(p0[2]), Float32(p0[3]), Float32(p0[4]))
    print("expert 1 gate[:5]:", Float32(p1[0]), Float32(p1[1]), Float32(p1[2]), Float32(p1[3]), Float32(p1[4]))

    # Check shared expert dimensions after sharding
    print("shared gate shape: [", E.SHARED_GATE.ROWS, ",", E.SHARED_GATE.COLS, "]")
    print("shared gate global: [", E.SHARED_GATE.GLOBAL_ROWS, ",", E.SHARED_GATE.GLOBAL_COLS, "]")
    print("shared down shape: [", E.SHARED_DOWN.ROWS, ",", E.SHARED_DOWN.COLS, "]")
    print("shared down global: [", E.SHARED_DOWN.GLOBAL_ROWS, ",", E.SHARED_DOWN.GLOBAL_COLS, "]")

    # Check multiple layers' shared gate variance
    for layer in range(1, 5):
        var sg_l = host.moe_layer_weight[E.SHARED_GATE](layer)
        print("layer", layer, "shared gate var:", var_at(sg_l.ptr, E.SHARED_GATE.ROWS * E.SHARED_GATE.COLS))

    _ = model
