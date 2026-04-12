"""Re-export shim — canonical source is now experimental3/kernels/rmsnorm.mojo."""

from experimental3.kernels.rmsnorm import (
    RMSNormNoScaleArgs,
    RMSNormPerHeadArgs,
    rmsnorm_no_scale_kernel,
    rmsnorm_per_head_kernel,
    rmsnorm_no_scale,
    rmsnorm_per_head,
)
