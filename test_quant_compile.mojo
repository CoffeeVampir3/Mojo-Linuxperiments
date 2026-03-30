"""Verify state sizes match between bf16 and hadquant models."""

from modeling.smollm2_hadquant_tp import HadQuantTPModel
from modeling.smollm2_tp import TPModel, BF16

def main():
    comptime M_HAD = HadQuantTPModel[1]
    comptime M_BF16 = TPModel[BF16, 1]

    print("State comparison (tp=1):")
    print("  bf16 state:     " + String(M_BF16.STATE_BYTES) + " bytes")
    print("  hadquant state: " + String(M_HAD.STATE_BYTES) + " bytes")
    print("  difference:     " + String(M_HAD.STATE_BYTES - M_BF16.STATE_BYTES) + " bytes")
    print("")
    print("  bf16 distributed:     " + String(M_BF16.DISTRIBUTED_BYTES) + " bytes")
    print("  hadquant distributed: " + String(M_HAD.DISTRIBUTED_BYTES) + " bytes")
    print("  weight savings:       " + String(M_BF16.DISTRIBUTED_BYTES - M_HAD.DISTRIBUTED_BYTES) + " bytes")
