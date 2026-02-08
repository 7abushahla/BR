# W-BR vs A-BR: Separate Weight and Activation Regularization

This guide explains the clean separation between Weight BR (W-BR) and Activation BR (A-BR) in our implementation.

---

## 🎯 **Design Philosophy**

### **Two Separate BR Components:**

#### **1. W-BR (Weight Bin Regularization)**
- **Purpose:** Original paper's approach for weights
- **Behavior:** Simple, paper-faithful
- **Stability:** Weights are inherently stable (not batch-dependent)
- **Gradient Flow:** Always allows BR gradients to alpha (paper's "co-evolution")
- **Control:** Enable/disable via `--lambda-br` (set to 0 to disable)

#### **2. A-BR (Activation Bin Regularization)**
- **Purpose:** Our extension to activations
- **Behavior:** Sophisticated, with stability controls
- **Stability:** Activations are batch-dependent (need careful handling)
- **Gradient Flow:** User-controlled via `--br-backprop-to-alpha-act` (default: False)
- **Control:** Enable/disable via `--lambda-br-act` (set to 0 to disable)

---

## 🔄 **Why Separate Them?**

### **Weights are Stable:**
```python
# Weights: Fixed per layer, same for all batches
conv1.weight.shape = [64, 3, 3, 3]  # Same values every iteration

# BR on weights can safely allow gradient flow to alpha
# The clustering target (weights) doesn't change batch-to-batch
```

### **Activations are Batch-Dependent:**
```python
# Activations: Different for each batch
batch1_activations = [0.1, 0.5, 0.2, ...]  # Batch 1
batch2_activations = [0.3, 0.4, 0.6, ...]  # Batch 2 (different!)

# BR on activations + gradient flow = "moving target"
# Target changes every batch, levels chase a moving distribution
```

**Solution:** Keep W-BR simple (paper's approach), add controls for A-BR (stability).

---

## 🎛️ **Control Mechanisms**

### **For W-BR (Weights):**
- **No special controls** (paper-faithful)
- **Always** allows BR gradients to alpha
- **Only** controlled by `--lambda-br` and `--freeze-alpha`

### **For A-BR (Activations):**
- **Has `--br-backprop-to-alpha-act` flag**
- **Default: False** (detached, stable)
- **Also** controlled by `--freeze-alpha` (global freeze)

---

## 📊 **Usage Scenarios**

### **Scenario 1: W-only BR (Original Paper)**
```bash
python cifar10_qat_br.py \
    --num-bits 2 \
    --lambda-br 1.0 \         # W-BR enabled
    --lambda-br-act 0.0 \     # A-BR disabled
    --freeze-alpha \
    --output-dir results/w_only_br/
```

**Result:**
- ✅ BR applied to **weights only**
- ✅ Paper-faithful approach
- ✅ W-BR gradients flow to weight alpha
- ❌ No BR on activations

**Use case:** Reproduce original paper results

---

### **Scenario 2: W+A BR with Stable A-BR (Recommended)**
```bash
python cifar10_qat_br.py \
    --num-bits 2 \
    --lambda-br 1.0 \         # W-BR enabled
    --lambda-br-act 1.0 \     # A-BR enabled
    --freeze-alpha \
    --output-dir results/w_a_br_stable/
    # Default: br-backprop-to-alpha-act=False (stable)
```

**Result:**
- ✅ BR applied to **both weights and activations**
- ✅ W-BR: Gradients flow (stable for weights)
- ✅ A-BR: Gradients detached (stable for activations)
- ✅ Best performance

**Use case:** Full quantization with stability (our approach)

---

### **Scenario 3: W+A BR with Moving Target A-BR**
```bash
python cifar10_qat_br.py \
    --num-bits 2 \
    --lambda-br 1.0 \         # W-BR enabled
    --lambda-br-act 1.0 \     # A-BR enabled
    --br-backprop-to-alpha-act \  # A-BR gradients flow (moving target!)
    --freeze-alpha \
    --output-dir results/w_a_br_moving_target/
```

**Result:**
- ✅ BR applied to both W and A
- ✅ W-BR: Gradients flow (stable)
- ⚠️ A-BR: Gradients flow (unstable, moving target)
- ❌ Can diverge or oscillate

**Use case:** Research on co-evolution (not recommended for production)

---

### **Scenario 4: A-only BR (Research)**
```bash
python cifar10_qat_br.py \
    --num-bits 2 \
    --lambda-br 0.0 \         # W-BR disabled
    --lambda-br-act 1.0 \     # A-BR enabled
    --freeze-alpha \
    --output-dir results/a_only_br/
```

**Result:**
- ❌ No BR on weights
- ✅ BR applied to **activations only**
- ✅ A-BR: Stable (gradients detached)

**Use case:** Isolate activation regularization effects

---

## 🔬 **Implementation Details**

### **In `train_epoch()`:**

```python
# 1. W-BR for WEIGHTS
if regularizer_w is not None and lambda_br > 0:
    weights_dict = {}
    alphas_w_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight_quantizer'):
            weights_dict[name] = module.weight
            alphas_w_dict[name] = module.weight_quantizer.alpha
    
    # ALWAYS use backprop_to_alpha=True for weights (paper-faithful)
    br_w_loss, _ = regularizer_w.compute_total_loss(
        weights_dict, alphas_w_dict, 
        backprop_to_alpha=True  # ← Always True for weights
    )

# 2. A-BR for ACTIVATIONS
if regularizer_a is not None and lambda_br_act > 0:
    activations_dict = hook_manager.get_pre_quant_activations()
    alphas_a_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'quantizer') and hasattr(module.quantizer, 'alpha'):
            alphas_a_dict[name] = module.quantizer.alpha
    
    # Use br_backprop_to_alpha_act flag (user-controlled for stability)
    br_a_loss, _ = regularizer_a.compute_total_loss(
        activations_dict, alphas_a_dict,
        backprop_to_alpha=br_backprop_to_alpha_act  # ← User flag
    )

# Total loss
loss = ce_loss + lambda_br * br_w_loss + lambda_br_act * br_a_loss
```

---

## 📊 **Comparison Matrix**

| Configuration | W-BR | A-BR | W-BR→α | A-BR→α | Stability | Paper-Faithful |
|---------------|------|------|--------|--------|-----------|----------------|
| **W-only** | ✅ | ❌ | ✅ | N/A | ★★★★★ | ✅ Yes |
| **W+A (stable)** | ✅ | ✅ | ✅ | ❌ | ★★★★★ | Partial |
| **W+A (moving)** | ✅ | ✅ | ✅ | ✅ | ★★☆☆☆ | Partial |
| **A-only** | ❌ | ✅ | N/A | ❌ | ★★★★☆ | ❌ No |

**Legend:**
- W-BR→α: W-BR gradients flow to alpha
- A-BR→α: A-BR gradients flow to alpha

---

## 🎓 **Why This Design?**

### **1. Paper Fidelity for Weights**
```python
# Original BR paper quantizes W+A but BR is described for weights
# Weights are stable → no need for gradient detachment
# Keep it simple and paper-faithful
```

### **2. Stability for Activations**
```python
# Activations vary batch-to-batch → need stability controls
# A-BR is our extension → add sophistication where needed
# Separate control allows best of both worlds
```

### **3. Clean Toggle**
```python
# Want W-only (paper)? → --lambda-br 1.0 --lambda-br-act 0.0
# Want W+A (full)?     → --lambda-br 1.0 --lambda-br-act 1.0
# Want A-only (research)? → --lambda-br 0.0 --lambda-br-act 1.0
```

---

## 💡 **Best Practices**

### **For Reproducing Paper:**
```bash
# Use W-only BR with freeze-alpha
python cifar10_qat_br.py \
    --lambda-br 1.0 \
    --lambda-br-act 0.0 \
    --freeze-alpha
```

### **For Best Performance:**
```bash
# Use W+A BR with stable A-BR
python cifar10_qat_br.py \
    --lambda-br 1.0 \
    --lambda-br-act 1.0 \
    --freeze-alpha
    # Default: br-backprop-to-alpha-act=False
```

### **For Ablation Studies:**
```bash
# Try different combinations
# W-only, A-only, W+A
# Different lambda values
# With/without freeze-alpha
```

---

## 🔍 **How to Tell What's Running**

When you run training, you'll see:

```
================================================================================
Starting 2-Stage QAT+BR Training
================================================================================
Stage 1 (Warmup): Epochs 1-30 (LSQ only)
Stage 2 (BR): Epochs 31-100

BR Configuration:
  - W-BR (weights): λ=1.0 [ENABLED]
    └─> Gradients to alpha: Always enabled (paper-faithful, weights are stable)
  - A-BR (activations): λ=1.0 [ENABLED]
    └─> Gradients to alpha: False (default: False, recommended)

Alpha Control:
  - Freeze alpha after warmup: True
    └─> If True: NO updates to alpha from ANY loss (CE or BR)
    └─> If False: CE can update alpha, A-BR respects --br-backprop-to-alpha-act flag
================================================================================
```

This clearly shows:
- Which BR components are active
- Gradient flow settings
- Alpha freeze status

---

## 📝 **Arguments Summary**

### **Core BR Arguments:**
```bash
--lambda-br 1.0           # W-BR weight (0 = disable)
--lambda-br-act 1.0       # A-BR weight (0 = disable)
```

### **Alpha Control:**
```bash
--freeze-alpha            # Freeze alpha after warmup (applies to both W and A)
--br-backprop-to-alpha-act  # Allow A-BR gradients to activation alpha (default: False)
```

**Note:** No `--br-backprop-to-alpha` for weights - always enabled for W-BR.

---

## 🎯 **Quick Reference**

| Goal | Command |
|------|---------|
| **Original paper (W-only)** | `--lambda-br 1.0 --lambda-br-act 0.0 --freeze-alpha` |
| **Full W+A (stable)** | `--lambda-br 1.0 --lambda-br-act 1.0 --freeze-alpha` |
| **W+A (moving target)** | `--lambda-br 1.0 --lambda-br-act 1.0 --br-backprop-to-alpha-act --freeze-alpha` |
| **A-only (research)** | `--lambda-br 0.0 --lambda-br-act 1.0 --freeze-alpha` |
| **No BR (LSQ baseline)** | `--lambda-br 0.0 --lambda-br-act 0.0` |

---

## ✅ **Summary**

### **W-BR (Weights):**
- ✅ Paper-faithful
- ✅ Always allows gradient flow to alpha
- ✅ Stable (weights don't change batch-to-batch)
- ✅ Simple (no extra flags)

### **A-BR (Activations):**
- ✅ Our extension
- ✅ User-controlled gradient flow
- ✅ Default: detached (stable)
- ✅ Sophisticated (has --br-backprop-to-alpha-act flag)

### **Together:**
- ✅ Toggle independently with lambda values
- ✅ W-only reproduces paper
- ✅ W+A gives best performance
- ✅ Clean separation of concerns

---

*Design finalized: February 8, 2026*  
*Based on: Clean separation between stable weights and batch-dependent activations*
