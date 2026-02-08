"""
LSQ (Learned Step-size Quantization) for Weights and Activations

Implements symmetric uniform quantization with learnable scale parameters
for full W/A quantization-aware training (QAT).

Based on: "Learned Step Size Quantization" (Esser et al., ICLR 2020)
Adapted from: LSQuantization-master/lsq.py (official implementation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============ Helper Functions (from official LSQ implementation) ============

def grad_scale(x, scale):
    """
    Scale gradient by a factor.
    
    Forward: pass through unchanged
    Backward: multiply gradient by scale
    """
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    """
    Straight-Through Estimator (STE) for rounding.
    
    Forward: round(x)
    Backward: pass gradient through unchanged (as if no rounding)
    """
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


# ============ LSQ Weight Quantizer ============

class LSQ_WeightQuantizer(nn.Module):
    """
    Learned Step-size Quantization (LSQ) for signed weights.
    
    Based on official LSQ implementation (ConvLSQ, LinearLSQ) with adaptations for:
    - Signed symmetric quantization (weights can be positive or negative)
    - Data-driven initialization
    - Compatible with standard Conv2d and Linear layers
    
    Args:
        num_bits: Number of bits for quantization (e.g., 2 for INT2)
        weight_shape: Shape of the weight tensor (for initialization)
    """
    
    def __init__(self, num_bits=2):
        super().__init__()
        self.num_bits = num_bits
        
        # SIGNED symmetric quantization: Qn=-(2^(b-1)), Qp=2^(b-1)-1
        # Example: 2-bit → Qn=-2, Qp=1 → levels: [-2s, -1s, 0, 1s]
        # Special case: 1-bit binary weights use {-1, +1} (not {-1, 0})
        if num_bits == 1:
            # Binary weights: {-α, +α}
            self.Qn = -1
            self.Qp = 1
            self.num_levels = 2
        else:
            self.Qn = -(2 ** (num_bits - 1))
            self.Qp = 2 ** (num_bits - 1) - 1
            self.num_levels = 2 ** num_bits
        
        # Learnable scale parameter (alpha in LSQ paper, s in BR paper)
        # Initialize to a reasonable default
        # For weights, typical range is [-0.1, 0.1] initially
        init_alpha = 0.1 / max(abs(self.Qn), abs(self.Qp))
        self.alpha = nn.Parameter(torch.tensor([init_alpha]))
        
        # Initialize flag (will be set on first forward pass)
        self.register_buffer('init_state', torch.zeros(1))
        
    def forward(self, x):
        """
        Quantize weights with learnable scale.
        
        Forward: x_q = round(x / alpha).clamp(Qn, Qp) * alpha
        Backward: STE for rounding, scaled gradient for alpha
        """
        # Use a safe LSQ scaling factor for edge cases (e.g., signed 1-bit has Qp=0).
        # Standard LSQ uses Qp, but that would cause divide-by-zero for W1.
        q_scale = max(self.Qp, 1)

        # Data-driven initialization (on first pass)
        if self.training and self.init_state == 0:
            # Initialize alpha based on weight statistics (LSQ paper Eq. 8)
            # alpha = 2 * mean(|x|) / sqrt(Qp)
            # NOTE: for W1, Qp=0 from signed range [-1, 0], so we use q_scale>=1.
            init_alpha = 2 * x.abs().mean() / math.sqrt(q_scale)
            self.alpha.data.copy_(init_alpha)
            self.init_state.fill_(1)
            print(f"  [LSQ Weight Init] num_bits={self.num_bits}, Qn={self.Qn}, Qp={self.Qp}, "
                  f"x.mean={x.abs().mean().item():.4f}, alpha={init_alpha.item():.6f}")
        
        # Gradient scale factor for alpha (LSQ paper Eq. 7)
        # g = 1 / sqrt(numel * Qp)
        # NOTE: for W1, use q_scale to avoid division by zero.
        g = 1.0 / math.sqrt(x.numel() * q_scale)
        
        # Scale alpha gradient
        alpha = grad_scale(self.alpha, g)
        
        # Quantize with STE (exactly as in original LSQ)
        # x_q = round(x / alpha).clamp(Qn, Qp) * alpha
        x_q = round_pass((x / alpha).clamp(self.Qn, self.Qp)) * alpha
        
        return x_q
    
    def get_quantization_levels(self):
        """Get current quantization levels."""
        levels = torch.arange(self.Qn, self.Qp + 1, dtype=self.alpha.dtype, device=self.alpha.device)
        return levels * self.alpha.data
    
    def extra_repr(self):
        return f'num_bits={self.num_bits}, Qn={self.Qn}, Qp={self.Qp}, alpha={self.alpha.item():.4f}'


# ============ LSQ Activation Quantizer ============

class LSQ_ActivationQuantizer(nn.Module):
    """
    Learned Step-size Quantization (LSQ) for unsigned activations [0, clip_value].
    
    Based on official LSQ implementation (ActLSQ) with adaptations for:
    - Unsigned activations (ReLU outputs)
    - Configurable clipping range
    - Data-driven initialization
    
    Args:
        num_bits: Number of bits for quantization (e.g., 2 for 4 levels)
        clip_value: Maximum activation value (e.g., 6.0 for ReLU6, None for standard ReLU)
    """
    
    def __init__(self, num_bits=2, clip_value=None):
        super().__init__()
        self.num_bits = num_bits
        self.clip_value = clip_value
        
        # Unsigned quantization: Qn=0, Qp=2^nbits - 1
        self.Qn = 0
        self.Qp = 2 ** num_bits - 1
        
        # Learnable scale parameter (alpha in LSQ paper)
        # Initialize to a reasonable default based on expected activation range
        if clip_value is None:
            init_alpha = 4.0 / self.Qp  # Assume ~[0, 4] for standard ReLU
        else:
            init_alpha = clip_value / self.Qp
        self.alpha = nn.Parameter(torch.tensor([init_alpha]))
        
        # Initialize flag (will be set on first forward pass)
        self.register_buffer('init_state', torch.zeros(1))
        
    def forward(self, x):
        """
        Quantize activations with learnable scale.
        
        Forward: x_q = round(x / alpha).clamp(Qn, Qp) * alpha
        Backward: STE for rounding, scaled gradient for alpha
        """
        # Data-driven initialization (on first batch)
        if self.training and self.init_state == 0:
            # Initialize alpha based on data statistics (LSQ paper Eq. 8)
            # alpha = 2 * mean(|x|) / sqrt(Qp)
            init_alpha = 2 * x.abs().mean() / math.sqrt(self.Qp)
            self.alpha.data.copy_(init_alpha)
            self.init_state.fill_(1)
            print(f"  [LSQ Act Init] num_bits={self.num_bits}, Qp={self.Qp}, "
                  f"x.mean={x.abs().mean().item():.4f}, alpha={init_alpha.item():.6f}")
        
        # Gradient scale factor for alpha (LSQ paper Eq. 7)
        # g = 1 / sqrt(numel * Qp)
        g = 1.0 / math.sqrt(x.numel() * self.Qp)
        
        # Scale alpha gradient
        alpha = grad_scale(self.alpha, g)
        
        # Quantize with STE (exactly as in original LSQ)
        # x_q = round(x / alpha).clamp(Qn, Qp) * alpha
        x_q = round_pass((x / alpha).clamp(self.Qn, self.Qp)) * alpha
        
        return x_q
    
    def get_quantization_levels(self):
        """Get current quantization levels."""
        levels = torch.arange(self.Qn, self.Qp + 1, dtype=self.alpha.dtype, device=self.alpha.device)
        return levels * self.alpha.data
    
    def extra_repr(self):
        return f'num_bits={self.num_bits}, Qp={self.Qp}, alpha={self.alpha.item():.4f}'


class QuantizedClippedReLU(nn.Module):
    """
    Clipped ReLU with LSQ quantization.
    
    Combines:
    1. ReLU activation
    2. Optional clipping to [0, clip_value]
    3. LSQ quantization with learnable scale
    
    This is used for QAT with Bin Regularization on both weights and activations.
    
    IMPORTANT: For BR to work correctly, it needs access to PRE-quantization
    activations (continuous values after ReLU/clip, before round_pass).
    We store these in self.pre_quant_activation for BR loss computation.
    """
    
    def __init__(self, clip_value=None, num_bits=2):
        super().__init__()
        self.clip_value = clip_value
        self.num_bits = num_bits
        
        # LSQ quantizer
        self.quantizer = LSQ_ActivationQuantizer(
            num_bits=num_bits,
            clip_value=clip_value
        )
        
        # Store pre-quantization activations for BR
        self.pre_quant_activation = None
    
    def forward(self, x):
        # Step 1: ReLU + clip to [0, clip_value] - CONTINUOUS VALUES
        if self.clip_value is not None:
            x_continuous = torch.clamp(F.relu(x), max=self.clip_value)
        else:
            x_continuous = F.relu(x)  # Standard ReLU, no clipping
        
        # Store pre-quantization activations (for BR loss)
        self.pre_quant_activation = x_continuous
        
        # Step 2: Quantize with learnable scale - DISCRETE VALUES
        x_quantized = self.quantizer(x_continuous)
        
        return x_quantized
    
    def extra_repr(self):
        alpha = self.quantizer.alpha.item() if hasattr(self.quantizer, 'alpha') else 0
        return f'clip_value={self.clip_value}, num_bits={self.num_bits}, alpha={alpha:.4f}'


class QuantizedConv2d(nn.Conv2d):
    """
    Conv2d with LSQ weight quantization.
    
    Drop-in replacement for nn.Conv2d with quantization-aware training.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, num_bits=2):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias)
        
        self.num_bits = num_bits
        # LSQ quantizer for weights
        self.weight_quantizer = LSQ_WeightQuantizer(num_bits=num_bits)
        
    def forward(self, x):
        # Always quantize weights (training AND eval)
        # This is critical for measuring true quantized performance
        w_q = self.weight_quantizer(self.weight)
        
        # Standard conv2d forward
        return self._conv_forward(x, w_q, self.bias)
    
    def extra_repr(self):
        s = super().extra_repr()
        return s + f', num_bits={self.num_bits}'


class QuantizedLinear(nn.Linear):
    """
    Linear with LSQ weight quantization.
    
    Drop-in replacement for nn.Linear with quantization-aware training.
    """
    
    def __init__(self, in_features, out_features, bias=True, num_bits=2):
        super().__init__(in_features, out_features, bias)
        
        self.num_bits = num_bits
        # LSQ quantizer for weights
        self.weight_quantizer = LSQ_WeightQuantizer(num_bits=num_bits)
        
    def forward(self, x):
        # Always quantize weights (training AND eval)
        # This is critical for measuring true quantized performance
        w_q = self.weight_quantizer(self.weight)
        
        # Standard linear forward
        return nn.functional.linear(x, w_q, self.bias)
    
    def extra_repr(self):
        s = super().extra_repr()
        return s + f', num_bits={self.num_bits}'
