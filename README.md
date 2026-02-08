# BR: Bin Regularization for Weight and Activation Quantization

This directory implements **Bin Regularization (BR)** for **full W/A quantization**, based on:

> **"Improving Low-Precision Network Quantization via Bin Regularization"**  
> Yixing Xu, Kai Han, Yehui Tang, Yunhe Wang, Chao Xu, Dacheng Tao  
> *ICCV 2021*

---

## 📋 Overview

### What is Bin Regularization?

**Bin Regularization (BR)** is a training technique that encourages network parameters to cluster tightly around quantization levels, reducing quantization error and improving low-bit performance.

**Key Idea:**
- **LSQ** (Learned Step-size Quantization) learns **where** the quantization grid should be
- **BR** makes parameters **stick** to that grid (minimize within-bin variance)
- They work **together**: LSQ defines optimal grid placement, BR shapes the distribution

**This Implementation:**
- Full **W/A quantization** with LSQ (weights AND activations)
- **Asymmetric quantization** support (e.g., W4A8, W2A4, W1A2)
- BR applied to **both weights (signed) and activations (unsigned)**
- **Independent control:** Separate bit widths, lambdas, and freeze flags for W and A
- Paper-faithful LSQ formulas with correct Qp-based scaling

---

## 🔬 BR vs. A-BR

| Aspect | **BR (This Implementation)** | **A-BR (Parent Directory)** |
|--------|------------------------------|----------------------------|
| **Target** | **Weights AND Activations** | **Activations only** |
| **Quantization** | Signed (weights) + Unsigned (activations) | Unsigned (ReLU outputs) |
| **Asymmetric Quant** | ✅ Supported (W4A8, W2A4, W1A2, etc.) | ❌ Symmetric only |
| **Weight Quant** | `[Qn·s, ..., -s, 0, s, ..., Qp·s]` | None (uses FP32 weights) |
| **Activation Quant** | `[0, s, 2s, ..., Qp·s]` | `[0, s, 2s, ..., Qp·s]` |
| **BR Applied To** | Both W and A (separate regularizers) | Activations only |
| **Alpha Control** | Separate freeze for W and A | Single freeze for all |
| **Paper** | Based on BR (ICCV 2021) + extensions | Novel activation-focused adaptation |

**Summary:**
- **BR** = Full W/A quantization + BR on both (comprehensive)
- **A-BR** = Activation quantization + BR on activations only (focused research)

---

## 🏗️ Architecture

```
BR/
├── br/                              # Core package
│   ├── __init__.py
│   ├── lsq_quantizer.py            # LSQ for weights (Conv2d, Linear)
│   └── regularizer_binreg.py       # BR loss for weights
├── experiments/                     # Training scripts
│   ├── cifar10_baseline.py         # Baseline FP32 training
│   ├── cifar10_qat_br.py           # QAT+BR for weights
│   └── mnist_qat_br.py             # MNIST example
├── results/                         # Checkpoints and logs
├── scripts/                         # Helper scripts
├── README.md                        # This file
└── QUICKSTART.md                    # Getting started guide
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision numpy matplotlib
```

### 2. Train a Baseline Model

```bash
python experiments/cifar10_baseline.py \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.1 \
    --output-dir results/cifar10_baseline/
```

### 3. Train with QAT+BR (Full W/A Quantization + BR)

#### **Symmetric Quantization (W=A):**
```bash
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/cifar10_baseline/best.pth \
    --num-bits 2 \               # Same bits for W and A (W2A2)
    --clip-value None \          # or 6.0 for ReLU6, 1.0 for ReLU1
    --lambda-br 1.0 \            # BR weight for weights
    --lambda-br-act 1.0 \        # BR weight for activations
    --warmup-epochs 30 \
    --qat-epochs 100 \
    --freeze-act-alpha \         # Freeze activation alpha (W alpha adapts)
    --output-dir results/cifar10_qat_br_w2a2/
```

#### **Asymmetric Quantization (W≠A):**
```bash
# W4A8: 4-bit weights, 8-bit activations
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/cifar10_baseline/best.pth \
    --num-bits-weight 4 \        # 4-bit weights (16 levels)
    --num-bits-act 8 \           # 8-bit activations (256 levels)
    --lambda-br 0.1 \            # Weaker BR for higher precision
    --lambda-br-act 0.01 \
    --freeze-act-alpha \
    --output-dir results/cifar10_qat_br_w4a8/
```

---

## 🎛️ Command-Line Arguments

### **Quantization Control:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-bits` | int | None | Symmetric quantization (same bits for W and A). Overrides separate settings. |
| `--num-bits-weight` | int | 2 | Bit width for weights (1, 2, 4, 8, 32=FP32) |
| `--num-bits-act` | int | 2 | Bit width for activations (1, 2, 4, 8, 32=FP32) |
| `--clip-value` | float | None | **Activation range:** None=ReLU [0,∞), 6.0=ReLU6 [0,6], 1.0=ReLU1 [0,1] |

**Examples:**
- `--num-bits 2` → W2A2 (symmetric)
- `--num-bits-weight 4 --num-bits-act 8` → W4A8 (asymmetric)
- `--num-bits-weight 2 --num-bits-act 4` → W2A4 (asymmetric)

### **BR Control:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lambda-br` | float | 1.0 | W-BR loss weight (0=disable W-BR) |
| `--lambda-br-act` | float | 1.0 | A-BR loss weight (0=disable A-BR) |
| `--warmup-epochs` | int | 30 | LSQ warmup epochs (no BR) |

**Examples:**
- `--lambda-br 1.0 --lambda-br-act 0.0` → W-only BR (paper-faithful)
- `--lambda-br 1.0 --lambda-br-act 1.0` → Full W+A BR
- `--lambda-br 0.1 --lambda-br-act 10.0` → Weak W-BR, strong A-BR

### **Alpha Control (Separate W/A):**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--freeze-weight-alpha` | flag | False | Freeze weight alpha after warmup |
| `--freeze-act-alpha` | flag | False | Freeze activation alpha after warmup |
| `--br-backprop-to-alpha-act` | flag | False | Allow A-BR gradients to flow to activation alpha |

**Key Differences:**
- **W-BR:** Always allows gradients to weight alpha (paper-faithful, weights are stable)
- **A-BR:** Gradients detached by default (activations are batch-dependent)

**Examples:**
- No flags → W alpha adapts (CE + W-BR), A alpha adapts (CE only) - **Recommended**
- `--freeze-act-alpha` → W adapts, A frozen (stable activations)
- `--freeze-weight-alpha --freeze-act-alpha` → Both frozen (maximum stability)

### **Training Control:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--qat-epochs` | int | 100 | Total training epochs |
| `--batch-size` | int | 128 | Training batch size |
| `--lr` | float | 0.01 | Learning rate |
| `--pretrained-baseline` | str | None | Path to FP32 baseline checkpoint |
| `--tensorboard` | flag | False | Enable TensorBoard logging (distributions, losses, alphas) |

---

## 📊 Training Strategy (Paper S2)

BR uses a **2-stage training** approach:

### Stage 1: Warmup (LSQ Only)
- **Duration:** 30-50 epochs
- **Objective:** Learn optimal quantization scales (α) from data
- **Loss:** `L = L_CE` (cross-entropy only)
- **Purpose:** Let LSQ find good grid positions before clustering

### Stage 2: Joint Training (LSQ + BR)
- **Duration:** 50-100 epochs
- **Objective:** Cluster weights around learned levels
- **Loss:** `L = L_CE + λ · L_BR`
- **α:** Can be frozen or continue learning
- **Purpose:** Reduce quantization error via tight clustering

**Typical λ values:** 0.1, 1.0, 10.0 (depends on network depth and bit-width)

---

## 📊 TensorBoard Visualization

Enable TensorBoard logging with `--tensorboard` to visualize:

### **What Gets Logged:**

1. **Training Metrics** (every epoch):
   - Total loss, CE loss, BR losses (W and A separately)
   - Train accuracy, test accuracy

2. **Weight Distributions** (every 5 epochs):
   - FP32 weight histograms per layer
   - Quantized weight histograms per layer
   - See clustering effect during BR phase

3. **Activation Distributions** (every 5 epochs):
   - Pre-quantization activation histograms
   - Post-quantization activation histograms
   - Visualize BR's effect on activations

4. **LSQ Alpha Values** (every 5 epochs):
   - Track alpha (scale) evolution over time
   - Monitor weight alphas vs activation alphas
   - See impact of freezing

### **Usage Example:**

```bash
# Train with TensorBoard enabled
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/baseline/best.pth \
    --num-bits 2 \
    --lambda-br 1.0 \
    --lambda-br-act 1.0 \
    --freeze-act-alpha \
    --tensorboard \
    --output-dir results/w2a2_tensorboard/

# In another terminal, start TensorBoard
tensorboard --logdir results/w2a2_tensorboard/tensorboard

# Open browser to http://localhost:6006
```

### **What to Look For:**

- **Histograms Tab:**
  - Weights should cluster tightly around quantization levels during BR phase
  - Activations should show reduced variance within bins
  - Compare FP32 vs quantized distributions

- **Scalars Tab:**
  - BR losses should decrease over epochs
  - Test accuracy should improve or stabilize
  - Alpha values should converge (if not frozen)

- **Expected Patterns:**
  - **Warmup:** Alphas adjust, distributions spread out
  - **BR Phase:** Distributions sharpen, cluster around levels
  - **After Freeze:** Alphas constant, distributions continue to cluster

**Note:** Logging every 5 epochs (for distributions) avoids performance overhead while still showing trends.

---

## 🔧 Activation Clipping Options

Control the activation range with `--clip-value`:

### **Option 1: Standard ReLU (Unbounded)**
```bash
python experiments/cifar10_qat_br.py \
    --clip-value None \  # or omit (default)
    --num-bits 2 \
    --output-dir results/relu/
```
**Range:** [0, ∞)  
**Quantization:** [0, α, 2α, ..., (2^b-1)α] (e.g., [0, α, 2α, 3α] for 2-bit)  
**Use case:** Standard networks, no activation clipping

### **Option 2: ReLU6 (Clip at 6)**
```bash
python experiments/cifar10_qat_br.py \
    --clip-value 6.0 \
    --num-bits 2 \
    --output-dir results/relu6/
```
**Range:** [0, 6]  
**Quantization:** [0, 2, 4, 6] for 2-bit, [0, 0.75, 1.5, ..., 6] for 8-bit  
**Use case:** MobileNets, efficient networks (better for low-bit quantization)

### **Option 3: ReLU1 (Clip at 1)**
```bash
python experiments/cifar10_qat_br.py \
    --clip-value 1.0 \
    --num-bits 2 \
    --output-dir results/relu1/
```
**Range:** [0, 1]  
**Quantization:** [0, 0.33, 0.67, 1.0] for 2-bit, [0, 0.125, 0.25, ..., 1] for 8-bit  
**Use case:** Normalized activations, ultra low-bit (1-bit, 2-bit)

### **Custom Clipping:**
```bash
--clip-value 3.0  # Custom: [0, 3]
--clip-value 10.0 # Custom: [0, 10]
```

**Note:** The baseline model should be trained with the same clip value for best results.

---

## 💡 Common Use Cases

### **1. Paper-Faithful W-BR (Original)**
```bash
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/baseline/best.pth \
    --num-bits 2 \
    --clip-value None \      # Standard ReLU
    --lambda-br 1.0 \
    --lambda-br-act 0.0 \
    --warmup-epochs 30 \
    --qat-epochs 100 \
    --output-dir results/paper_w_br/
# W alpha co-evolves with W-BR (paper approach), no A-BR
```

### **2. Full W+A BR (Recommended)**
```bash
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/baseline/best.pth \
    --num-bits 2 \
    --lambda-br 1.0 \
    --lambda-br-act 1.0 \
    --warmup-epochs 30 \
    --qat-epochs 100 \
    --freeze-act-alpha \
    --tensorboard \
    --output-dir results/full_w_a_br/
# W alpha adapts, A alpha frozen (stable)
# TensorBoard: tensorboard --logdir results/full_w_a_br/tensorboard
```

### **3. W4A8 Deployment Configuration**
```bash
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/baseline/best.pth \
    --num-bits-weight 4 \
    --num-bits-act 8 \
    --lambda-br 0.1 \
    --lambda-br-act 0.01 \
    --freeze-act-alpha \
    --output-dir results/w4a8_deployment/
# Common for mobile/edge deployment
```

### **4. Ultra Low-Precision (W1A2)**
```bash
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/baseline/best.pth \
    --num-bits-weight 1 \
    --num-bits-act 2 \
    --lambda-br 10.0 \
    --lambda-br-act 1.0 \
    --warmup-epochs 50 \
    --qat-epochs 150 \
    --freeze-weight-alpha \
    --freeze-act-alpha \
    --output-dir results/w1a2_ultra_low/
# Binary weights need strong BR and more training
```

### **5. ReLU6 for Better Low-Bit Quantization**
```bash
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/baseline/best_relu6.pth \
    --num-bits 2 \
    --clip-value 6.0 \       # ReLU6: [0, 6]
    --lambda-br 1.0 \
    --lambda-br-act 1.0 \
    --freeze-act-alpha \
    --output-dir results/relu6_w2a2/
# Bounded activations work better for 2-bit quantization
```

### **6. ReLU1 for Ultra Low-Bit (1-bit, 2-bit)**
```bash
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/baseline/best_relu1.pth \
    --num-bits 2 \
    --clip-value 1.0 \       # ReLU1: [0, 1]
    --lambda-br 1.0 \
    --lambda-br-act 1.0 \
    --freeze-weight-alpha \
    --freeze-act-alpha \
    --output-dir results/relu1_w2a2/
# Normalized range helps with extreme quantization
```

### **7. Ablation: A-BR Only**
```bash
python experiments/cifar10_qat_br.py \
    --pretrained-baseline results/baseline/best.pth \
    --num-bits 2 \
    --clip-value None \
    --lambda-br 0.0 \
    --lambda-br-act 1.0 \
    --freeze-act-alpha \
    --output-dir results/a_br_only/
# Isolate activation BR effects
```

---

## 🧮 BR Loss Formulation

For a layer with weights `W` and LSQ scale `s`:

**1. Bin Assignment (LSQ Integer Code):**
```
bin_i = round(clip(w / s, Qn, Qp))
```

**2. Quantization Levels (Dynamic, tied to LSQ):**
```
levels = [Qn·s, ..., -s, 0, s, ..., Qp·s]
```

**3. Per-Bin Loss (Paper Eq. 5):**
```
L_BR = Σ [MSE(mean(bin_i), target_i) + Var(bin_i)]
       i=Qn to Qp
```

Where:
- `MSE`: Pushes bin mean toward target level
- `Var`: Reduces spread within bin (only if bin size ≥ 2)
- Empty bins are skipped

**4. Total Loss:**
```
L = L_CE + λ · L_BR
```

---

## 📈 Expected Results (CIFAR-10 ResNet18)

### Symmetric Quantization (W=A):

| Config | FP32 Acc | QAT (LSQ Only) | QAT+BR (λ=1.0) | Improvement |
|--------|----------|----------------|----------------|-------------|
| **W8A8**  | 95.0%    | 94.8%          | 94.9%          | +0.1%       |
| **W4A4**  | 95.0%    | 93.5%          | 94.0%          | +0.5%       |
| **W2A2**  | 95.0%    | 89.2%          | 91.5%          | +2.3%       |

### Asymmetric Quantization (W≠A):

| Config | FP32 Acc | QAT (LSQ Only) | QAT+BR | Notes |
|--------|----------|----------------|--------|-------|
| **W4A8** | 95.0% | ~94.5% | ~94.7% | Common for deployment |
| **W2A4** | 95.0% | ~91.0% | ~92.5% | Strong compression |
| **W1A2** | 95.0% | ~85.0% | ~87.5% | Ultra low-precision |

**Key Observations:**
- BR provides **larger gains at lower bit-widths** (2-bit, 4-bit)
- **Asymmetric quantization** allows trading off weight vs activation precision
- **W4A8** is common for mobile/edge (memory savings with minimal accuracy loss)
- Weights cluster tighter around quantization levels
- Quantization MSE decreases during BR phase
- **Effectiveness score** (BR metric) correlates with accuracy

---

## 📦 Core Components

### 1. LSQ Quantizers

```python
from br import LSQ_WeightQuantizer, LSQ_ActivationQuantizer

# Weight quantizer (signed symmetric) - can use different bits than activations
weight_quantizer = LSQ_WeightQuantizer(num_bits=4)  # W4: 16 levels
weights_quantized = weight_quantizer(weights_fp32)

# Activation quantizer (unsigned) - independent bit width
act_quantizer = LSQ_ActivationQuantizer(num_bits=8, clip_value=None)  # A8: 256 levels
activations_quantized = act_quantizer(activations_fp32)
```

### 2. Quantized Layers (Drop-in Replacements)

```python
from br import QuantizedConv2d, QuantizedLinear, QuantizedClippedReLU

# W4A8 configuration example
num_bits_weight = 4  # 4-bit weights
num_bits_act = 8     # 8-bit activations

# Replace standard layers (quantize weights)
conv = QuantizedConv2d(in_channels=64, out_channels=128, kernel_size=3, 
                       num_bits=num_bits_weight)
fc = QuantizedLinear(in_features=512, out_features=10, 
                     num_bits=num_bits_weight)

# Replace ReLU with quantized version (quantize activations)
relu = QuantizedClippedReLU(clip_value=None, num_bits=num_bits_act)  # or 6.0 for ReLU6
```

### 3. Bin Regularizers (Weights + Activations)

```python
from br import BinRegularizer, ActivationHookManager

# Separate regularizers for weights and activations (can use different bit widths)
regularizer_w = BinRegularizer(num_bits=4, signed=True, name="Weights")       # W4
regularizer_a = BinRegularizer(num_bits=8, signed=False, name="Activations")  # A8

# Setup hook manager to capture activations
hook_manager = ActivationHookManager(model, target_modules=[QuantizedClippedReLU])

# Forward pass
outputs = model(inputs)

# Collect weights and alphas
weights_dict = {}
alphas_w_dict = {}
for name, module in model.named_modules():
    if hasattr(module, 'weight_quantizer'):
        weights_dict[name] = module.weight
        alphas_w_dict[name] = module.weight_quantizer.alpha

# Collect activations and alphas
activations_dict = hook_manager.get_pre_quant_activations()
alphas_a_dict = {}
for name, module in model.named_modules():
    if hasattr(module, 'quantizer') and hasattr(module.quantizer, 'alpha'):
        alphas_a_dict[name] = module.quantizer.alpha

# Compute BR losses
br_w_loss, _ = regularizer_w.compute_total_loss(weights_dict, alphas_w_dict)
br_a_loss, _ = regularizer_a.compute_total_loss(activations_dict, alphas_a_dict)

# Total loss
loss = ce_loss + lambda_br * br_w_loss + lambda_br_act * br_a_loss
```

---

## 🔧 Key Differences from Paper

### ✅ Faithful to Paper:
- Same bin assignment (LSQ integer code)
- Same loss formulation (MSE + Variance)
- Same 2-stage training (warmup, then joint)
- Dynamic levels tied to LSQ scales

### 🔄 Minor Adaptations:
- **PyTorch implementation** (paper used different framework)
- **ResNet18 on CIFAR-10/MNIST** (paper used ImageNet)
- **Unsigned activations** optional (paper focused on weights)

---

## 📖 Paper Reference

```bibtex
@inproceedings{xu2021binreg,
  title={Improving Low-Precision Network Quantization via Bin Regularization},
  author={Xu, Yixing and Han, Kai and Tang, Yehui and Wang, Yunhe and Xu, Chao and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

---

## 🤝 Relation to A-BR

This **BR** implementation serves as the **foundation** for understanding how the original method works on weights. 

Our **A-BR** (Activation Bin Regularization) in the parent directory is a **novel adaptation** that:
1. Applies the same BR principle to **activations** instead of weights
2. Uses **hook-based collection** since activations are batch-dependent
3. Handles **unsigned quantization** for post-ReLU values
4. Maintains **semantic alignment** between LSQ and BR scales

Both implementations are **complementary**:
- **BR (weights)** → Static, signed, direct parameter access
- **A-BR (activations)** → Dynamic, unsigned, hook-based

---

## 📧 Contact

For questions or issues related to this implementation, please refer to the main project documentation in the `ActReg/` parent directory.

---

## 📄 License

This implementation is for research purposes. Please cite the original BR paper if you use this code.
