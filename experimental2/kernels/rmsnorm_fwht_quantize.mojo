"""Re-export shim — canonical sources are now:
  experimental3/kernels/fwht.mojo    (FWHT primitives)
  experimental3/kernels/rmsnorm.mojo (all RMSNorm variants)
"""

from experimental3.kernels.fwht import (
    fwht_width,
    butterfly_partner,
    butterfly_shuffle,
    fwht_apply,
    fwht_block,
)

from experimental3.kernels.rmsnorm import (
    rmsnorm_gamma_fwht_quantize_row,
    rmsnorm_dual_gamma_fwht_quantize_row,
    rmsnorm_fwht_quantize_row,
    rmsnorm_gamma_fwht_per_block_quantize_row,
    RmsNormFwhtArgs,
    RmsNormGammaFwhtPerBlockArgs,
    RmsNormGammaFwhtArgs,
    RmsNormDualGammaFwhtArgs,
    rmsnorm_gamma_fwht_per_block_quantize_worker,
    rmsnorm_gamma_fwht_quantize_worker,
    rmsnorm_dual_gamma_fwht_quantize_worker,
    rmsnorm_fwht_quantize_worker,
    rmsnorm_gamma_fwht_quantize,
    rmsnorm_gamma_fwht_per_block_quantize,
    rmsnorm_dual_gamma_fwht_quantize,
    rmsnorm_fwht_quantize,
)
