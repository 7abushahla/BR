"""
BR: Bin Regularization for Weight and Activation Quantization

This package implements Bin Regularization (BR) for full W/A quantization, based on:
"Improving Low-Precision Network Quantization via Bin Regularization" (ICCV 2021)

Key components:
- LSQ_WeightQuantizer: Learned Step-size Quantization for weights (signed)
- LSQ_ActivationQuantizer: Learned Step-size Quantization for activations (unsigned)
- QuantizedConv2d, QuantizedLinear: Quantized layers
- QuantizedClippedReLU: Quantized activation function
- BinRegularizer: BR loss for weights or activations (supports both signed/unsigned)
- ActivationHookManager: Captures activations for BR loss

Features:
- Full W/A quantization with LSQ
- BR applied to both weights (signed) and activations (unsigned)
- Separate λ control for weights vs activations
- Paper-faithful LSQ implementation (Qp-based scaling)
- Always-on quantization during eval (no FP32 fallback)
"""

from .lsq_quantizer import (
    LSQ_WeightQuantizer,
    LSQ_ActivationQuantizer,
    QuantizedConv2d,
    QuantizedLinear,
    QuantizedClippedReLU,
    grad_scale,
    round_pass
)

from .regularizer_binreg import BinRegularizer
from .hooks import ActivationHookManager

__all__ = [
    'LSQ_WeightQuantizer',
    'LSQ_ActivationQuantizer',
    'QuantizedConv2d',
    'QuantizedLinear',
    'QuantizedClippedReLU',
    'BinRegularizer',
    'ActivationHookManager',
    'grad_scale',
    'round_pass'
]
