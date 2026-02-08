# Alpha Control Guide: Freeze vs. BR Backprop

This guide explains the two separate mechanisms for controlling how LSQ alpha parameters are updated during training.

---

## 🎯 **Two Independent Controls**

### **1. `--freeze-alpha`** (Stop ALL alpha updates)
- **What it does:** Sets `alpha.requires_grad = False` after warmup
- **Effect:** Alpha receives **NO gradients** from any loss (CE or BR)
- **When to use:** When you want completely fixed quantization levels during BR phase

### **2. `--br-backprop-to-alpha`** (Control BR gradients to alpha)
- **What it does:** Controls whether BR loss gradients flow to alpha
- **Effect:** 
  - `False` (default): Only CE loss updates alpha, BR doesn't
  - `True`: Both CE and BR update alpha
- **When to use:** When alpha is NOT frozen, but you want stable levels during BR

---

## 🔄 **How They Work Together**

### **Scenario 1: No Flags (Default, Recommended)**
```bash
python cifar10_qat_br.py \
    --warmup-epochs 30 \
    --qat-epochs 100
```

**Behavior:**
- **Warmup (1-30):** Alpha updated by CE loss ✅
- **BR Phase (31-100):** 
  - Alpha updated by CE loss ✅
  - Alpha NOT updated by BR loss ❌ (detached)

**Result:** Stable quantization levels during BR, but can still adapt via CE loss

---

### **Scenario 2: Freeze Alpha**
```bash
python cifar10_qat_br.py \
    --freeze-alpha \
    --warmup-epochs 30 \
    --qat-epochs 100
```

**Behavior:**
- **Warmup (1-30):** Alpha updated by CE loss ✅
- **BR Phase (31-100):** 
  - Alpha is **frozen** (requires_grad=False)
  - No updates from CE ❌
  - No updates from BR ❌

**Result:** Completely fixed quantization levels during BR phase

---

### **Scenario 3: BR Backprop Enabled (Moving Target)**
```bash
python cifar10_qat_br.py \
    --br-backprop-to-alpha \
    --warmup-epochs 30 \
    --qat-epochs 100
```

**Behavior:**
- **Warmup (1-30):** Alpha updated by CE loss ✅
- **BR Phase (31-100):** 
  - Alpha updated by CE loss ✅
  - Alpha updated by BR loss ✅

**Result:** "Moving target" - levels shift as BR tries to cluster values (can be unstable)

---

### **Scenario 4: Freeze + Backprop (Contradiction)**
```bash
python cifar10_qat_br.py \
    --freeze-alpha \
    --br-backprop-to-alpha \  # ← This has no effect when frozen!
    --warmup-epochs 30
```

**Behavior:**
- Same as Scenario 2 (freeze dominates)
- `--br-backprop-to-alpha` is ignored because alpha is frozen

**Result:** Completely fixed levels (--freeze-alpha wins)

---

## 📊 **Decision Matrix**

| `--freeze-alpha` | `--br-backprop-to-alpha` | CE → α | BR → α | Use Case |
|------------------|--------------------------|--------|--------|----------|
| ❌ No | ❌ No (default) | ✅ Yes | ❌ No | **Recommended:** Stable BR, adaptive via CE |
| ✅ Yes | ❌ No | ❌ No | ❌ No | **Stable:** Fixed levels during BR |
| ❌ No | ✅ Yes | ✅ Yes | ✅ Yes | **Unstable:** Moving target (not recommended) |
| ✅ Yes | ✅ Yes | ❌ No | ❌ No | Same as row 2 (freeze wins) |

---

## 🎓 **The "Moving Target" Problem**

### **What is it?**

When BR gradients flow to alpha, the quantization levels move while BR tries to cluster values around them:

```
Iteration 1: levels = [0, 0.5, 1.0, 1.5]  → BR: "cluster around these!"
Iteration 2: levels = [0, 0.52, 1.05, 1.58] → BR: "wait, new targets!"
Iteration 3: levels = [0, 0.48, 0.98, 1.42] → BR: "now these!"
```

**Problem:** The target keeps moving, making it hard to converge.

### **Solution:**

**Detach alpha from BR loss** (default behavior):
```python
# In compute_bin_loss():
if backprop_to_alpha:
    alpha_for_levels = alpha  # Gradients flow
else:
    alpha_for_levels = alpha.item()  # Detached, no gradients
```

**Result:** BR clusters values around **fixed** levels (from previous iteration)

---

## 💡 **Recommendations**

### **For Most Cases: Use Default (No Flags)**
```bash
python cifar10_qat_br.py \
    --lambda-br 1.0 \
    --lambda-br-act 1.0
```

**Why:**
- ✅ Stable BR (levels don't move during clustering)
- ✅ Adaptive (CE loss can still adjust alpha)
- ✅ Best of both worlds

### **For Maximum Stability: Freeze Alpha**
```bash
python cifar10_qat_br.py \
    --freeze-alpha \
    --lambda-br 1.0 \
    --lambda-br-act 1.0
```

**Why:**
- ✅ Completely fixed levels
- ✅ BR has stable targets
- ✅ Good for debugging
- ⚠️ Less adaptive (can't fix bad alpha values)

### **Avoid: BR Backprop Without Freeze**
```bash
# NOT RECOMMENDED
python cifar10_qat_br.py \
    --br-backprop-to-alpha \  # ← Moving target!
    --lambda-br 1.0
```

**Why:**
- ❌ Unstable (moving target problem)
- ❌ Can diverge
- ❌ Harder to tune

---

## 🔬 **Experimental Comparison**

### **Expected Behavior:**

| Configuration | Convergence | Final Acc | Stability |
|---------------|-------------|-----------|-----------|
| **Default** (no flags) | Fast | ~91.5% | ✅ Stable |
| **Freeze alpha** | Fast | ~91.0% | ✅ Very stable |
| **BR backprop** | Slow | ~90.0%? | ⚠️ Unstable |

### **Training Curves:**

**Default (Stable):**
```
Epoch 31: BR_W=0.012, BR_A=0.023
Epoch 40: BR_W=0.008, BR_A=0.015  ← Decreasing (good)
Epoch 60: BR_W=0.003, BR_A=0.006
Epoch 100: BR_W=0.001, BR_A=0.002 ← Converged
```

**BR Backprop (Unstable):**
```
Epoch 31: BR_W=0.012, BR_A=0.023
Epoch 40: BR_W=0.015, BR_A=0.028  ← Increasing? (bad)
Epoch 60: BR_W=0.010, BR_A=0.020  ← Oscillating
Epoch 100: BR_W=0.008, BR_A=0.015 ← Never converges
```

---

## 📝 **Summary**

### **Default Behavior (Recommended):**
- ✅ BR gradients **do NOT** flow to alpha
- ✅ CE gradients **do** flow to alpha
- ✅ Stable targets for BR
- ✅ Adaptive via CE

### **With `--freeze-alpha`:**
- ✅ **Nothing** updates alpha after warmup
- ✅ Maximum stability
- ⚠️ Less adaptive

### **With `--br-backprop-to-alpha`:**
- ⚠️ BR gradients flow to alpha (moving target)
- ⚠️ Can be unstable
- ❌ Not recommended unless experimenting

---

## 🎯 **Recommended Usage**

### **Standard Training:**
```bash
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/baseline/checkpoints/best.pth \
    --num-bits 2 \
    --lambda-br 1.0 \
    --lambda-br-act 1.0 \
    --warmup-epochs 30 \
    --qat-epochs 100 \
    --output-dir results/qat_br_2bit/
    # Default: br-backprop-to-alpha=False (stable)
```

### **Maximum Stability:**
```bash
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/baseline/checkpoints/best.pth \
    --num-bits 2 \
    --lambda-br 1.0 \
    --lambda-br-act 1.0 \
    --warmup-epochs 30 \
    --qat-epochs 100 \
    --freeze-alpha \  # ← Add this
    --output-dir results/qat_br_2bit_frozen/
```

---

## 🔧 **Implementation Details**

### **In `BinRegularizer.compute_bin_loss()`:**

```python
# Detach alpha if we don't want BR gradients flowing to it
if backprop_to_alpha:
    # Allow gradients to flow: loss → levels → alpha
    alpha_for_levels = alpha
else:
    # Detach: BR loss doesn't affect alpha (only CE loss does)
    alpha_for_levels = alpha.item() if torch.is_tensor(alpha) else alpha

# Compute levels (gradients flow only if backprop_to_alpha=True)
levels = level_indices * alpha_for_levels

# Bin assignment always uses detached alpha (stable assignment)
alpha_for_assignment = alpha.item() if torch.is_tensor(alpha) else alpha
bin_assignments = round(clamp(weights / alpha_for_assignment, Qn, Qp))
```

**Key Insight:** Even when detached, BR still clusters values around the **current** levels - it just doesn't try to move the levels via gradients.

---

## ❓ **FAQ**

**Q: Why is default `br-backprop-to-alpha=False`?**  
A: Avoids moving target problem. BR clusters around stable levels.

**Q: When would I use `--br-backprop-to-alpha`?**  
A: Rarely. Only for research/experiments on co-evolution of grid + clustering.

**Q: What if I freeze alpha AND enable backprop?**  
A: Freeze wins. Backprop has no effect because `requires_grad=False`.

**Q: Can I use different settings for weights vs activations?**  
A: Currently no - both use the same `br-backprop-to-alpha` flag. Could be extended if needed.

**Q: What did the BR paper use?**  
A: Paper mentions both approaches (S1=freeze, S2=continue). S2 is "co-evolution" but can be unstable.

---

## ✅ **Bottom Line**

**Use the default (no flags) for best results:**
- Stable BR clustering
- Adaptive quantization scales
- Fast convergence
- Best accuracy

Only use `--freeze-alpha` if you need maximum stability and don't mind less adaptability.

Avoid `--br-backprop-to-alpha` unless you're specifically researching the moving target problem.

---

*Guide created: February 8, 2026*  
*Implementation: BR package with full W/A quantization*
