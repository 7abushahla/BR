## 🚀 Quick Start Guide: Full W/A Quantization with BR

This guide will help you get started with training quantized neural networks using **Bin Regularization (BR)** for **both weights and activations**.

---

### Step 1: Train FP32 Baseline

First, train a standard FP32 ResNet18 as a pretrained baseline:

```bash
python experiments/cifar10_baseline.py \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.1 \
    --seed 42 \
    --output-dir results/cifar10_baseline/
```

**Expected Results:**
- Training time: ~2-3 hours on single GPU
- Best accuracy: ~95% on CIFAR-10 test set
- Checkpoint saved to: `results/cifar10_baseline/checkpoints/best_seed42_*.pth`

---

### Step 2: QAT+BR Training (2-bit W/A quantization)

Now apply full W/A quantization using LSQ + Bin Regularization:

```bash
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/cifar10_baseline/checkpoints/best_seed42_*.pth \
    --num-bits 2 \
    --clip-value None \          # Standard ReLU (or 6.0 for ReLU6)
    --lambda-br 1.0 \            # BR for weights
    --lambda-br-act 1.0 \        # BR for activations
    --warmup-epochs 30 \
    --qat-epochs 100 \
    --freeze-alpha \
    --batch-size 128 \
    --lr 0.01 \
    --seed 42 \
    --output-dir results/cifar10_qat_br_2bit_wa/
```

**What happens:**
1. **Epochs 1-30 (Warmup):** LSQ learns optimal quantization scales (α) for both W and A
2. **Epoch 30:** Alpha parameters are frozen (weights and activations)
3. **Epochs 31-100 (BR Phase):** 
   - **W-BR:** Weights cluster around quantization levels (paper-faithful, gradients flow)
   - **A-BR:** Activations cluster around quantization levels (stable, gradients detached by default)

**Expected Results:**
- Training time: ~1-2 hours on single GPU
- Best accuracy: ~91-92% on CIFAR-10 test set (INT2 W/A)
- Checkpoint saved with best test accuracy
- BR shapes distributions for both weights (signed) and activations (unsigned)

---

### Step 2.5: Toggle Between W-only and W+A BR

Our implementation separates **W-BR** (weights) and **A-BR** (activations):

#### **W-only BR (Original Paper):**
```bash
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/cifar10_baseline/checkpoints/best_seed42_*.pth \
    --num-bits 2 \
    --lambda-br 1.0 \        # W-BR enabled
    --lambda-br-act 0.0 \    # A-BR disabled ← Key difference
    --warmup-epochs 30 \
    --qat-epochs 100 \
    --freeze-alpha \
    --output-dir results/cifar10_qat_br_2bit_w_only/
```

**Result:** BR applied to weights only (paper-faithful)

#### **W+A BR (Full Package):**
```bash
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/cifar10_baseline/checkpoints/best_seed42_*.pth \
    --num-bits 2 \
    --lambda-br 1.0 \        # W-BR enabled
    --lambda-br-act 1.0 \    # A-BR enabled ← Both active
    --warmup-epochs 30 \
    --qat-epochs 100 \
    --freeze-alpha \
    --output-dir results/cifar10_qat_br_2bit_w_a/
```

**Result:** BR applied to both weights and activations (best performance)

**Key Differences:**
- **W-BR:** Always allows gradients to alpha (weights are stable)
- **A-BR:** Gradients detached by default (activations are batch-dependent)
- **Toggle:** Use `--lambda-br-act 0.0` for W-only, `1.0` for W+A

**See `W_BR_vs_A_BR_GUIDE.md` for detailed explanation.**

---

### Step 3: Compare Different λ Values

BR strength is controlled by the `--lambda-br` hyperparameter. Try multiple values:

```bash
# Weak BR (λ=0.1)
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/cifar10_baseline/checkpoints/best_seed42_*.pth \
    --num-bits 2 \
    --lambda-br 0.1 \
    --warmup-epochs 30 \
    --qat-epochs 100 \
    --freeze-alpha \
    --output-dir results/cifar10_qat_br_2bit_lam0.1/

# Strong BR (λ=10.0)
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/cifar10_baseline/checkpoints/best_seed42_*.pth \
    --num-bits 2 \
    --lambda-br 10.0 \
    --warmup-epochs 30 \
    --qat-epochs 100 \
    --freeze-alpha \
    --output-dir results/cifar10_qat_br_2bit_lam10.0/
```

**Rule of Thumb:**
- λ = 0.1-1.0 → Good for 4-bit, 8-bit
- λ = 1.0-10.0 → Good for 2-bit (lower precision needs stronger clustering)

---

### Step 4: Try Different Bit-Widths

```bash
# 4-bit quantization (easier, less BR needed)
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/cifar10_baseline/checkpoints/best_seed42_*.pth \
    --num-bits 4 \
    --lambda-br 0.1 \
    --warmup-epochs 30 \
    --qat-epochs 100 \
    --freeze-alpha \
    --output-dir results/cifar10_qat_br_4bit_lam0.1/

# 8-bit quantization (should match FP32 closely)
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/cifar10_baseline/checkpoints/best_seed42_*.pth \
    --num-bits 8 \
    --lambda-br 0.1 \
    --warmup-epochs 30 \
    --qat-epochs 50 \
    --freeze-alpha \
    --output-dir results/cifar10_qat_br_8bit_lam0.1/
```

---

### Expected Accuracy Summary (CIFAR-10 ResNet18)

| Bit-Width | FP32  | QAT (LSQ only) | QAT+BR (λ=1.0) | Gain  |
|-----------|-------|----------------|----------------|-------|
| **INT8**  | 95.0% | 94.8%          | 94.9%          | +0.1% |
| **INT4**  | 95.0% | 93.5%          | 94.0%          | +0.5% |
| **INT2**  | 95.0% | 89.2%          | 91.5%          | +2.3% |

**Key Insight:** BR provides larger gains at lower bit-widths!

---

### Understanding the Output

During training, you'll see:

```
Epoch 1/100 [WARMUP] (LR=0.010000): Train Acc=85.23%, Test Acc=83.45%
...
Epoch 30/100 [WARMUP] (LR=0.005234): Train Acc=92.15%, Test Acc=90.12%

================================================================================
FREEZING ALPHA
================================================================================

Epoch 31/100 [BR] (LR=0.004987): Train Acc=92.34%, Test Acc=90.45%, BR Loss=0.012345
...
Epoch 100/100 [BR] (LR=0.000020): Train Acc=94.56%, Test Acc=91.78%, BR Loss=0.002134
```

**What to look for:**
- **Warmup phase:** Accuracy should climb, similar to standard training
- **BR phase:** BR Loss should decrease (weights clustering tighter)
- **Final accuracy:** Should improve over LSQ-only baseline

---

### Tips for Best Results

1. **Always use pretrained baseline:**
   - Starting from random weights makes LSQ initialization harder
   - Pretrained baseline provides good initial weight distribution

2. **Tune warmup duration:**
   - Too short → α not well-calibrated, BR might hurt
   - Too long → wasted compute, diminishing returns
   - **Sweet spot:** 20-30% of total epochs (e.g., 30/100)

3. **Freeze α after warmup:**
   - Paper recommends freezing for stability
   - Allows weights to adapt to fixed quantization grid
   - Use `--freeze-alpha` flag

4. **λ scaling:**
   - Lower bit-width → Higher λ (more aggressive clustering)
   - Deeper networks → Lower λ (more layers to sum over)
   - Start with λ=1.0, adjust based on results

5. **Learning rate:**
   - QAT typically uses 10× lower LR than baseline (0.01 vs 0.1)
   - Use cosine annealing for smooth convergence

---

### Comparison with A-BR

| Feature | BR (This Package) | A-BR (Parent Directory) |
|---------|-------------------|-------------------------|
| **Target** | Weights + Activations | Activations only |
| **Use Case** | Full W/A quantization | Activation-focused research |
| **Weight Quant** | INT2/INT4/INT8 (signed) | FP32 (not quantized) |
| **Activation Quant** | INT2/INT4/INT8 (unsigned) | INT2/INT4/INT8 (unsigned) |
| **BR Applied To** | Both W and A | Activations only |
| **Paper** | Based on ICCV 2021 | Novel adaptation |

**When to use each:**
- **BR:** Full W/A quantization with comprehensive BR (both weights and activations)
- **A-BR:** Activation-focused experiments with FP32 weights

---

### Next Steps

1. ✅ Run baseline training
2. ✅ Run QAT+BR with default settings
3. ✅ Compare results vs. LSQ-only
4. 📊 Visualize weight distributions (histograms)
5. 📊 Analyze BR effectiveness metrics
6. 🔬 Experiment with different λ, bit-widths, warmup durations

For more details, see the full `README.md` and the original paper.

---

### Troubleshooting

**Q: Accuracy drops during BR phase?**
- A: Try longer warmup (50 epochs), lower λ (0.1), or don't freeze α

**Q: BR loss not decreasing?**
- A: Check that quantized layers are being detected (print `weights_dict.keys()`)
- Ensure α is not too small (check initialization logs)

**Q: Out of memory?**
- A: Reduce batch size or use gradient checkpointing

**Q: How to use my own dataset/model?**
- A: Adapt the training script:
  1. Replace data loaders
  2. Replace model with your architecture
  3. Change `QuantizedConv2d` → `nn.Conv2d` for layers you don't want quantized
  4. Keep 2-stage training strategy

---

Happy quantizing! 🎯
