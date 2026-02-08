"""
Bin Regularization (BR) for Quantized Tensors

Encourages values to cluster tightly around quantization bin centers,
creating sharp (Dirac-like) distributions for each bin.

Based on: "Improving Low-Precision Network Quantization via Bin Regularization" (ICCV 2021)

Key Insight from BR Paper:
- LSQ learns WHERE the quantization grid should be (via learned step size s)
- BR makes values STICK to that grid (minimize within-bin variance)
- They are NOT redundant: LSQ defines optimal grid, BR shapes distribution

Paper-Faithful Implementation Details:
- BR uses the SAME levels as LSQ: [Qn·s, ..., -s, 0, s, ..., Qp·s]
- Bin assignment: round(clip(x/s, Qn, Qp)) - LSQ's integer code (Eq. 1-2)
- Loss per bin: L_mse(mean, target) + L_var (Eq. 5)
- Single-element bins: MSE only, variance=0 (V_i ≤ 1)
- Multi-element bins: Both MSE and variance (var with unbiased=False)
- Empty bins: Skipped
- Levels are DYNAMIC (based on LSQ's current learned s)

Two-Stage Training (BR Paper S2 Strategy):
1. Warmup (~30 epochs): LSQ learns optimal s from data (no BR)
2. Joint training: Add BR while continuing to optimize s (co-evolution)

Signed vs Unsigned:
- For weights (signed): Qn = -2^(b-1), Qp = 2^(b-1)-1
- For activations (unsigned): Qn = 0, Qp = 2^b-1
"""

import torch
import torch.nn as nn


class BinRegularizer(nn.Module):
    """
    Bin Regularization (BR) for quantized tensors.
    
    Implements the BR paper's per-bin loss formulation:
    L_BR = Σ (L_mse(⟨w_i⟩, ŵ_i) + L_var(w_i))
           i=1 to 2^b
    
    Where:
    - ŵ_i = i·s (LSQ's quantization levels, NOT fixed linspace!)
    - Bins determined by LSQ integer code: round(clip(w/s, Qn, Qp))
    - L_mse: Push bin mean toward that bin's LSQ target
    - L_var: { 0             if V_i ≤ 1
             { var(w_i)      if V_i ≥ 2  (unbiased=False)
    
    Implementation notes:
    - BR uses the SAME levels as LSQ (dynamic, based on learned s/alpha)
    - We sum loss across layers (paper formulation)
    - Can be applied to weights (signed) or activations (unsigned)
    
    The lambda weighting is applied in the training loop: L = L_CE + λ · L_BR
    
    Args:
        num_bits: Target bit-width for quantization (e.g., 2 for INT2)
        signed: Whether to use signed quantization bins (weights=True, activations=False)
        name: Optional label for logging (e.g., "Weights", "Activations")
    """
    
    def __init__(self, num_bits=2, signed=True, name=None):
        super().__init__()
        self.num_bits = num_bits
        self.signed = signed
        self.name = name or ("Weights" if signed else "Activations")
        
        if signed:
            # SIGNED symmetric quantization bounds (for weights)
            # Qn = -(2^(b-1)), Qp = 2^(b-1) - 1
            # Example: 2-bit → Qn=-2, Qp=1 → 4 levels: [-2s, -1s, 0, 1s]
            self.Qn = -(2 ** (num_bits - 1))
            self.Qp = 2 ** (num_bits - 1) - 1
        else:
            # UNSIGNED quantization bounds (for activations)
            # Qn = 0, Qp = 2^b - 1
            # Example: 2-bit → Qn=0, Qp=3 → 4 levels: [0, 1s, 2s, 3s]
            self.Qn = 0
            self.Qp = 2 ** num_bits - 1
        self.num_levels = 2 ** num_bits
        
        if signed:
            levels_str = f"[{self.Qn}α, ..., -α, 0, α, ..., {self.Qp}α]"
            quant_str = "SIGNED symmetric"
        else:
            levels_str = f"[{self.Qn}α, ..., {self.Qp}α]"
            quant_str = "UNSIGNED"
        print(f"BinRegularizer ({self.name}): {num_bits}-bit ({self.num_levels} levels)")
        print(f"  Levels are DYNAMIC: {levels_str} (tied to LSQ)")
        print(f"  ✓ {quant_str} quantization bins")
    
    def compute_bin_loss(self, weights: torch.Tensor, alpha: float, backprop_to_alpha: bool = False) -> tuple:
        """
        Compute bin regularization loss for a single weight tensor.
        
        CRITICAL SEMANTIC CLARITY:
        - 'alpha' here is the STEP SIZE (s in BR paper notation)
        - NOT the weight range or clip value!
        - LSQ quantization: w_q = round(w / s) * s
        - Quantization levels: [Qn·s, ..., -s, 0, s, ..., Qp·s]
        - BR targets are these SAME levels (using LSQ's learned s)
        
        This function fetches the CURRENT alpha/s from LSQ each forward pass,
        ensuring BR and LSQ always use the same grid (no semantic mismatch).
        
        Args:
            weights: Weight tensor (any shape)
            alpha: Current learned step size (s) from LSQ quantizer for this layer
            backprop_to_alpha: If True, allow BR gradients to flow to alpha (moving target)
                              If False, detach alpha (stable target, recommended)
            
        Returns:
            (total_loss, mse_loss, var_loss, bin_info)
        """
        # Flatten weights
        weights_flat = weights.flatten()
        
        # Detach alpha if we don't want BR gradients flowing to it
        # This prevents "moving target" issues during BR phase
        if backprop_to_alpha:
            # Allow gradients to flow: loss → levels → alpha
            # Paper's "simultaneous update" approach (can be unstable)
            alpha_for_levels = alpha
        else:
            # Detach: BR loss doesn't affect alpha (only CE loss does)
            # Stable approach: levels don't move while clustering
            alpha_for_levels = alpha.item() if torch.is_tensor(alpha) else alpha
        
        # Compute quantization levels dynamically from LSQ's current alpha
        # levels = [Qn·α, ..., -α, 0, α, ..., Qp·α]
        level_indices = torch.arange(self.Qn, self.Qp + 1, device=weights_flat.device, dtype=weights_flat.dtype)
        levels = level_indices * alpha_for_levels
        
        # For bin assignment, always use the actual alpha value (not detached)
        # We want to assign to bins based on current alpha, just don't let BR gradients flow back
        alpha_for_assignment = alpha.item() if torch.is_tensor(alpha) else alpha
        bin_assignments = torch.round(torch.clamp(weights_flat / alpha_for_assignment, self.Qn, self.Qp)).long()
        # NOTE: round() is non-differentiable (returns integer indices)
        # This means bin membership is automatically stop-grad (stable)
        # Only the target levels (bin_center = bin_idx * alpha) can receive gradients
        
        # Map bin assignments from [Qn, Qp] to [0, num_levels-1] for indexing
        bin_assignments_idx = bin_assignments - self.Qn
        
        # Compute loss for each bin
        total_mse = 0.0
        total_var = 0.0
        bins_used = 0
        
        for bin_idx in range(self.num_levels):
            # Get weights assigned to this bin
            mask = (bin_assignments_idx == bin_idx)
            bin_values = weights_flat[mask]
            
            if len(bin_values) == 0:
                # Empty bin, skip
                continue
            
            bins_used += 1
            bin_center = levels[bin_idx]
            
            # L_mse: Push bin mean toward target (paper Eq. 5)
            # Use .mean() consistently for all non-empty bins (even size 1)
            mse_loss = ((bin_values.mean() - bin_center) ** 2)
            total_mse += mse_loss
            
            # L_var: Minimize variance (make sharp)
            # Paper: variance term becomes 0 when V_i ≤ 1
            var_loss = 0.0
            if len(bin_values) >= 2:
                # Use unbiased=False explicitly (paper just says "var")
                var_loss = bin_values.var(unbiased=False)
                total_var += var_loss
            # else: var_loss = 0 (implicit, V_i = 1)
        
        # Total BR loss for this layer: Σ(L_mse + L_var) across bins
        # Paper: L_BR = Σ(L_mse(⟨w_i⟩, ŵ_i) + L_var(w_i))
        loss = total_mse + total_var
        
        # ========== Compute BR Effectiveness Metrics ==========
        # Metrics computed without gradients (logging only)
        with torch.no_grad():
            # Use ACTUAL LSQ quantization (same as bin assignment) for metrics
            # This ensures metrics reflect true quantization error, not nearest-level
            weights_quantized = bin_assignments.float() * alpha
            
            # 1. Mean distance to quantized value (actual LSQ error)
            mean_distance = (weights_flat - weights_quantized).abs().mean()
        
            # 2. BR Effectiveness Score: 0-100%
            # Perfect clustering (Dirac deltas) = 100%
            # Uniform spread = 0%
            # Tensor-safe: ensure max_dist is tensor with correct device/dtype
            max_dist = (alpha * 0.5) if torch.is_tensor(alpha) else torch.tensor(
                alpha * 0.5, device=weights_flat.device, dtype=weights_flat.dtype
            )
            effectiveness = 100.0 * (1.0 - mean_distance / (max_dist + 1e-12))
            effectiveness = effectiveness.clamp(0.0, 100.0)
        
            # 3. Percentage of weights "at" quantization levels (within 1% of alpha)
            threshold = (0.01 * alpha) if torch.is_tensor(alpha) else torch.tensor(
                0.01 * alpha, device=weights_flat.device, dtype=weights_flat.dtype
            )
            near_levels = ((weights_flat - weights_quantized).abs() < threshold).float().mean() * 100.0
        
            # 4. Actual Quantization MSE (using LSQ assignment, not nearest-level)
            quantization_mse = ((weights_flat - weights_quantized) ** 2).mean()
        
        # Info for logging
        bin_info = {
            'bins_used': bins_used,
            'br_mse_loss': total_mse.item() if isinstance(total_mse, torch.Tensor) else total_mse,  # BR loss component
            'br_var_loss': total_var.item() if isinstance(total_var, torch.Tensor) else total_var,  # BR loss component
            'quantization_mse': quantization_mse.item(),  # ACTUAL quantization error!
            'mean_distance': mean_distance.item(),
            'effectiveness': effectiveness.item(),
            'pct_near_levels': near_levels.item(),
        }
        
        return loss, total_mse, total_var, bin_info
    
    def compute_total_loss(self, weights_dict: dict, alphas_dict: dict, backprop_to_alpha: bool = False) -> tuple:
        """
        Compute bin regularization loss across all layers.
        
        Args:
            weights_dict: Dictionary of {layer_name: weight_tensor}
            alphas_dict: Dictionary of {layer_name: current_alpha_value}
                         These are the learned step sizes from LSQ quantizers
            backprop_to_alpha: If True, allow BR gradients to flow to alpha
                              If False, detach alpha (recommended, avoids moving target)
            
        Returns:
            (total_loss, info_dict)
        """
        total_loss = 0.0
        total_mse = 0.0
        total_var = 0.0
        total_quant_mse = 0.0  # Actual quantization MSE
        total_mean_distance = 0.0
        total_effectiveness = 0.0
        total_pct_near = 0.0
        layer_losses = {}
        
        for layer_name, weights in weights_dict.items():
            # Get the current alpha for this layer from LSQ
            if layer_name not in alphas_dict:
                continue  # Skip if alpha not provided
            alpha = alphas_dict[layer_name]
            
            loss, mse, var, bin_info = self.compute_bin_loss(weights, alpha, backprop_to_alpha=backprop_to_alpha)
            
            total_loss += loss
            total_mse += mse if isinstance(mse, torch.Tensor) else torch.tensor(mse)
            total_var += var if isinstance(var, torch.Tensor) else torch.tensor(var)
            total_quant_mse += bin_info['quantization_mse']
            total_mean_distance += bin_info['mean_distance']
            total_effectiveness += bin_info['effectiveness']
            total_pct_near += bin_info['pct_near_levels']
            
            layer_losses[layer_name] = {
                'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
                'mse': mse.item() if isinstance(mse, torch.Tensor) else mse,
                'var': var.item() if isinstance(var, torch.Tensor) else var,
                'quantization_mse': bin_info['quantization_mse'],
                'bins_used': bin_info['bins_used'],
                'mean_distance': bin_info['mean_distance'],
                'effectiveness': bin_info['effectiveness'],
                'pct_near_levels': bin_info['pct_near_levels'],
            }
        
        num_layers = len(weights_dict)
        # PAPER-FAITHFUL: Sum across layers (no averaging)
        # Paper Eq. (5): L_BR = Σ(L_mse + L_var) over bins
        # Paper Eq. (7): L = L_CE + λ·L_BR
        # → L_BR is a SUM over all bins (across all layers), not an average
        total_loss_final = total_loss  # Sum, not average
        total_mse_final = total_mse
        total_var_final = total_var
        # Metrics: still average for interpretability
        avg_quant_mse = total_quant_mse / num_layers if num_layers > 0 else 0.0
        avg_mean_distance = total_mean_distance / num_layers if num_layers > 0 else 0.0
        avg_effectiveness = total_effectiveness / num_layers if num_layers > 0 else 0.0
        avg_pct_near = total_pct_near / num_layers if num_layers > 0 else 0.0
        
        info_dict = {
            'avg_loss': total_loss_final.item() if isinstance(total_loss_final, torch.Tensor) else total_loss_final,
            'avg_mse': total_mse_final.item() if isinstance(total_mse_final, torch.Tensor) else total_mse_final,
            'avg_var': total_var_final.item() if isinstance(total_var_final, torch.Tensor) else total_var_final,
            'avg_quantization_mse': avg_quant_mse,  # Metrics: averaged for interpretability
            'avg_mean_distance': avg_mean_distance,
            'avg_effectiveness': avg_effectiveness,
            'avg_pct_near': avg_pct_near,
            'layer_losses': layer_losses
        }
        
        return total_loss_final, info_dict
    
    def get_bin_statistics(self, weights_dict: dict, alphas_dict: dict) -> dict:
        """
        Get detailed statistics about bin assignments (for visualization/debugging).
        
        Args:
            weights_dict: Dictionary of {layer_name: weight_tensor}
            alphas_dict: Dictionary of {layer_name: current_alpha_value}
        
        Returns dictionary with per-layer bin counts and statistics.
        """
        stats = {}
        
        for layer_name, weights in weights_dict.items():
            if layer_name not in alphas_dict:
                continue
            
            weights_flat = weights.flatten()
            alpha = alphas_dict[layer_name]
            
            # Compute levels dynamically
            level_indices = torch.arange(self.Qn, self.Qp + 1, device=weights_flat.device, dtype=weights_flat.dtype)
            levels = level_indices * alpha
            
            # Bin assignment (paper-faithful: LSQ integer code)
            bin_assignments = torch.round(torch.clamp(weights_flat / alpha, self.Qn, self.Qp)).long()
            bin_assignments_idx = bin_assignments - self.Qn
            
            # Count per bin
            bin_counts = []
            bin_means = []
            bin_stds = []
            
            for bin_idx in range(self.num_levels):
                mask = (bin_assignments_idx == bin_idx)
                bin_values = weights_flat[mask]
                
                bin_counts.append(len(bin_values))
                if len(bin_values) > 0:
                    bin_means.append(bin_values.mean().item())
                    bin_stds.append(bin_values.std().item() if len(bin_values) > 1 else 0.0)
                else:
                    bin_means.append(0.0)
                    bin_stds.append(0.0)
            
            stats[layer_name] = {
                'bin_counts': bin_counts,
                'bin_means': bin_means,
                'bin_stds': bin_stds,
                'total_values': len(weights_flat),
                'levels': levels.cpu().tolist()
            }
        
        return stats
