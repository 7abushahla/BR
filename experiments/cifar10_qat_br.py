"""
CIFAR-10 ResNet18 with W/A Quantization using LSQ + Bin Regularization

This script implements the ORIGINAL Bin Regularization method from the paper
"Improving Low-Precision Network Quantization via Bin Regularization" (ICCV 2021).

Key features:
- Weight + activation quantization (W/A)
- LSQ for learnable quantization scales
- 2-stage training: Warmup (LSQ only) → Joint training (LSQ + BR)
- BR can be applied to weights and/or activations

Usage:
    python experiments/cifar10_qat_br.py \
        --pretrained-baseline results/baseline/best.pth \
        --num-bits 2 \
        --lambda-br 1.0 \
        --warmup-epochs 30 \
        --qat-epochs 100 \
        --output-dir results/cifar10_qat_br_2bit/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import os
import sys
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path to import br package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from br import QuantizedConv2d, QuantizedLinear, QuantizedClippedReLU, BinRegularizer, ActivationHookManager


# ============================================================================
# ResNet18 with Quantized Weights
# ============================================================================

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, num_bits_weight=32):
    """3x3 convolution with padding - quantized version"""
    if num_bits_weight < 32:
        return QuantizedConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False,
                               dilation=dilation, num_bits=num_bits_weight)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, num_bits_weight=32):
    """1x1 convolution - quantized version"""
    if num_bits_weight < 32:
        return QuantizedConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               bias=False, num_bits=num_bits_weight)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_bits_weight=32, num_bits_act=32, clip_value=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, num_bits_weight=num_bits_weight)
        self.bn1 = nn.BatchNorm2d(planes)
        # Use quantized ReLU if num_bits_act < 32
        if num_bits_act < 32:
            self.relu = QuantizedClippedReLU(clip_value=clip_value, num_bits=num_bits_act)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, num_bits_weight=num_bits_weight)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18_Quantized(nn.Module):
    def __init__(self, num_classes=10, num_bits_weight=2, num_bits_act=2, clip_value=None):
        super(ResNet18_Quantized, self).__init__()
        self.num_bits_weight = num_bits_weight
        self.num_bits_act = num_bits_act
        self.clip_value = clip_value
        self.inplanes = 64

        # First layer: quantized if num_bits_weight < 32
        if num_bits_weight < 32:
            self.conv1 = QuantizedConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, num_bits=num_bits_weight)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Use quantized ReLU if num_bits_act < 32
        if num_bits_act < 32:
            self.relu = QuantizedClippedReLU(clip_value=clip_value, num_bits=num_bits_act)
        else:
            self.relu = nn.ReLU(inplace=True)

        # ResNet blocks with quantized weights AND activations
        self.layer1 = self._make_layer(64, 2, stride=1, num_bits_weight=num_bits_weight, num_bits_act=num_bits_act, clip_value=clip_value)
        self.layer2 = self._make_layer(128, 2, stride=2, num_bits_weight=num_bits_weight, num_bits_act=num_bits_act, clip_value=clip_value)
        self.layer3 = self._make_layer(256, 2, stride=2, num_bits_weight=num_bits_weight, num_bits_act=num_bits_act, clip_value=clip_value)
        self.layer4 = self._make_layer(512, 2, stride=2, num_bits_weight=num_bits_weight, num_bits_act=num_bits_act, clip_value=clip_value)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final FC layer: quantized weights
        if num_bits_weight < 32:
            self.fc = QuantizedLinear(512, num_classes, num_bits=num_bits_weight)
        else:
            self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1, num_bits_weight=32, num_bits_act=32, clip_value=None):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride, num_bits_weight=num_bits_weight),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample, 
                                 num_bits_weight=num_bits_weight, num_bits_act=num_bits_act, clip_value=clip_value))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, 
                                     num_bits_weight=num_bits_weight, num_bits_act=num_bits_act, clip_value=clip_value))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device,
                regularizer_w=None, lambda_br=0.0,
                regularizer_a=None, lambda_br_act=0.0,
                hook_manager=None, br_backprop_to_alpha_act=False, epoch=0, writer=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_ce_loss = 0.0
    running_br_w_loss = 0.0
    running_br_a_loss = 0.0
    correct = 0
    total = 0

    # Set hook manager to training mode if provided
    if hook_manager is not None:
        hook_manager.set_training_mode(True)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        
        # Clear previous activations
        if hook_manager is not None:
            hook_manager.clear_activations()
        
        # Forward pass (activations captured by hooks)
        outputs = model(inputs)
        ce_loss = criterion(outputs, targets)

        # Add BR loss if enabled (for BOTH weights and activations)
        br_w_loss = torch.tensor(0.0, device=device)
        br_a_loss = torch.tensor(0.0, device=device)
        
        if regularizer_w is not None and lambda_br > 0:
            # 1. BR for WEIGHTS
            weights_dict = {}
            alphas_w_dict = {}
            for name, module in model.named_modules():
                if hasattr(module, 'weight_quantizer'):
                    weights_dict[name] = module.weight
                    alphas_w_dict[name] = module.weight_quantizer.alpha

            if len(weights_dict) > 0:
                # Weight BR: Always allow gradients (paper-faithful, stable for weights)
                br_w_loss, _ = regularizer_w.compute_total_loss(weights_dict, alphas_w_dict, 
                                                                backprop_to_alpha=True)
                running_br_w_loss += br_w_loss.item()
        
        if regularizer_a is not None and lambda_br_act > 0 and hook_manager is not None:
            # 2. BR for ACTIVATIONS (pre-quantization)
            # Activation BR: Use br_backprop_to_alpha_act flag (user-controlled, for stability)
            activations_dict = hook_manager.get_pre_quant_activations()
            alphas_a_dict = {}
            for name, module in model.named_modules():
                if hasattr(module, 'quantizer') and hasattr(module.quantizer, 'alpha'):
                    alphas_a_dict[name] = module.quantizer.alpha

            if len(activations_dict) > 0:
                br_a_loss, _ = regularizer_a.compute_total_loss(activations_dict, alphas_a_dict,
                                                                backprop_to_alpha=br_backprop_to_alpha_act)
                running_br_a_loss += br_a_loss.item()
        
        # Total loss
        loss = ce_loss + lambda_br * br_w_loss + lambda_br_act * br_a_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_ce_loss += ce_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(train_loader)
    avg_ce_loss = running_ce_loss / len(train_loader)
    avg_br_w_loss = running_br_w_loss / len(train_loader) if regularizer_w is not None else 0.0
    avg_br_a_loss = running_br_a_loss / len(train_loader) if regularizer_a is not None else 0.0
    accuracy = 100. * correct / total

    # TensorBoard logging (if enabled)
    if writer is not None:
        # 1. Log training metrics
        writer.add_scalar('Train/Loss_Total', avg_loss, epoch)
        writer.add_scalar('Train/Loss_CE', avg_ce_loss, epoch)
        writer.add_scalar('Train/Accuracy', accuracy, epoch)
        
        if regularizer_w is not None and lambda_br > 0:
            writer.add_scalar('Train/Loss_BR_Weight', avg_br_w_loss, epoch)
        
        if regularizer_a is not None and lambda_br_act > 0:
            writer.add_scalar('Train/Loss_BR_Activation', avg_br_a_loss, epoch)
        
        # 2. Log weight distributions (every 5 epochs to avoid overhead)
        if epoch % 5 == 0:
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    writer.add_histogram(f'Weights/{name}', module.weight.data.cpu(), epoch)
                    
                    # Log quantized weights if available
                    if hasattr(module, 'weight_quantizer'):
                        try:
                            weight_q = module.weight_quantizer(module.weight)
                            writer.add_histogram(f'Weights_Quantized/{name}', weight_q.data.cpu(), epoch)
                        except:
                            pass
        
        # 3. Log activation distributions (from hook manager if available)
        if hook_manager is not None and epoch % 5 == 0:
            activations_dict = hook_manager.get_pre_quant_activations()
            for name, acts in activations_dict.items():
                if acts is not None and acts.numel() > 0:
                    writer.add_histogram(f'Activations/{name}', acts.cpu(), epoch)
        
        # 4. Log alpha values (LSQ scales)
        if epoch % 5 == 0:
            for name, param in model.named_parameters():
                if 'alpha' in name:
                    writer.add_scalar(f'Alpha/{name}', param.item(), epoch)

    return avg_loss, avg_ce_loss, avg_br_w_loss, avg_br_a_loss, accuracy


def test(model, test_loader, criterion, device, epoch=0, writer=None):
    """Evaluate on test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total

    # TensorBoard logging (if enabled)
    if writer is not None:
        writer.add_scalar('Test/Loss', avg_loss, epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)

    return avg_loss, accuracy


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 QAT+BR (Weight Quantization)')
    
    # Model arguments
    parser.add_argument('--pretrained-baseline', type=str, default=None,
                       help='Path to pretrained FP32 baseline checkpoint')
    parser.add_argument('--num-bits', type=int, default=None,
                       help='Number of bits for BOTH W and A quantization (default: None). '
                            'If specified, overrides --num-bits-weight and --num-bits-act. '
                            'Use this for symmetric W/A quantization (e.g., W2A2).')
    parser.add_argument('--num-bits-weight', type=int, default=2,
                       help='Number of bits for weight quantization only (default: 2). '
                            'Ignored if --num-bits is set. Use for asymmetric quantization (e.g., W4A8).')
    parser.add_argument('--num-bits-act', type=int, default=2,
                       help='Number of bits for activation quantization only (default: 2). '
                            'Ignored if --num-bits is set. Use for asymmetric quantization (e.g., W4A8).')
    parser.add_argument('--clip-value', type=float, default=None,
                       help='Activation clipping range (default: None). '
                            'Options: None (standard ReLU, [0, inf)), 6.0 (ReLU6, [0, 6]), 1.0 (ReLU1, [0, 1])')
    
    # BR arguments
    parser.add_argument('--lambda-br', type=float, default=1.0,
                       help='BR loss weight for weights (default: 1.0, set to 0 to disable W-BR)')
    parser.add_argument('--lambda-br-act', type=float, default=1.0,
                       help='BR loss weight for activations (default: 1.0, set to 0 to disable A-BR)')
    parser.add_argument('--warmup-epochs', type=int, default=30,
                       help='Warmup epochs (LSQ only, no BR) (default: 30)')
    parser.add_argument('--freeze-weight-alpha', action='store_true',
                       help='Freeze weight alpha after warmup (stops W alpha updates from CE and W-BR)')
    parser.add_argument('--freeze-act-alpha', action='store_true',
                       help='Freeze activation alpha after warmup (stops A alpha updates from CE and A-BR)')
    parser.add_argument('--br-backprop-to-alpha-act', action='store_true',
                       help='Allow A-BR gradients to flow to activation alpha (moving target for activations). '
                            'Default: False (stable, recommended). '
                            'Note: W-BR always allows gradients (paper-faithful, weights are stable)')
    
    # Training arguments
    parser.add_argument('--qat-epochs', type=int, default=100,
                       help='Total QAT epochs (including warmup) (default: 100)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay (default: 5e-4)')
    
    # Other arguments
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Enable TensorBoard logging (weight/activation distributions, losses, etc.)')
    
    args = parser.parse_args()

    # Handle bit width arguments
    if args.num_bits is not None:
        # If --num-bits is specified, use it for both W and A
        args.num_bits_weight = args.num_bits
        args.num_bits_act = args.num_bits
        print(f"Using symmetric quantization: W{args.num_bits}A{args.num_bits}")
    else:
        # Use separate bit widths for W and A
        print(f"Using asymmetric quantization: W{args.num_bits_weight}A{args.num_bits_act}")

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # TensorBoard writer (optional)
    writer = None
    if args.tensorboard:
        log_dir = os.path.join(args.output_dir, 'tensorboard')
        writer = SummaryWriter(log_dir)
        print(f"✓ TensorBoard logging enabled: {log_dir}")
        print(f"  Run: tensorboard --logdir {log_dir}")
    else:
        print("✗ TensorBoard logging disabled (use --tensorboard to enable)")

    # Data loaders
    print("Loading CIFAR-10 dataset...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                 download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    clip_str = f", clip={args.clip_value}" if args.clip_value else ""
    print(f"\nCreating ResNet18 with W{args.num_bits_weight}A{args.num_bits_act} quantization{clip_str}...")
    model = ResNet18_Quantized(num_classes=10, 
                               num_bits_weight=args.num_bits_weight, 
                               num_bits_act=args.num_bits_act,
                               clip_value=args.clip_value).to(device)

    # Load pretrained baseline if provided
    if args.pretrained_baseline is not None:
        print(f"Loading pretrained baseline from: {args.pretrained_baseline}")
        checkpoint = torch.load(args.pretrained_baseline, map_location='cpu')
        
        # Load state dict (handle keys carefully)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load (ignore quantizer-specific keys that don't exist in FP32)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"✓ Loaded {len(pretrained_dict)} parameter tensors from baseline")

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Optimizer: Include all parameters (weights, BN, alpha)
    # NOTE: LSQ uses gradient scaling in forward pass (g = 1/sqrt(numel*Qp))
    # This automatically gives alpha an effective ~10x higher learning rate
    # So we can use a single optimizer for all parameters (paper-faithful)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay)

    # Learning rate scheduler (cosine annealing ONLY during BR phase)
    # T_max should be the BR phase duration, not total epochs
    br_phase_epochs = args.qat_epochs - args.warmup_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=br_phase_epochs)
    
    # Collect alpha parameters for optional freezing after warmup
    # Separate weight alphas from activation alphas
    weight_alpha_params = []
    act_alpha_params = []
    
    for name, p in model.named_parameters():
        if 'alpha' in name:
            # Weight quantizers are in QuantizedConv2d/QuantizedLinear modules
            # Activation quantizers are in QuantizedClippedReLU modules
            if 'weight_quantizer.alpha' in name:
                weight_alpha_params.append(p)
            elif 'quantizer.alpha' in name:  # activation quantizer
                act_alpha_params.append(p)
            else:
                # Fallback: if unclear, add to both lists (shouldn't happen)
                weight_alpha_params.append(p)
                act_alpha_params.append(p)
    
    print(f"\nFound {len(weight_alpha_params)} weight alpha parameters (LSQ scales for weights)")
    print(f"Found {len(act_alpha_params)} activation alpha parameters (LSQ scales for activations)")

    # Bin Regularizers (use separate bit widths for W and A)
    regularizer_w = BinRegularizer(num_bits=args.num_bits_weight, signed=True, name="Weights")
    regularizer_a = BinRegularizer(num_bits=args.num_bits_act, signed=False, name="Activations")

    # Setup activation hook manager (for BR on activations)
    from br.lsq_quantizer import QuantizedClippedReLU
    hook_manager = ActivationHookManager(
        model,
        target_modules=[QuantizedClippedReLU],
        exclude_first_last=False
    )

    # Training loop
    print(f"\n{'='*80}")
    print(f"Starting 2-Stage QAT+BR Training")
    print(f"{'='*80}")
    print(f"Stage 1 (Warmup): Epochs 1-{args.warmup_epochs} (LSQ only)")
    print(f"Stage 2 (BR): Epochs {args.warmup_epochs+1}-{args.qat_epochs}")
    print(f"\nBR Configuration:")
    print(f"  - W-BR (weights): λ={args.lambda_br} {'[ENABLED]' if args.lambda_br > 0 else '[DISABLED]'}")
    print(f"    └─> Gradients to alpha: Always enabled (paper-faithful, weights are stable)")
    print(f"  - A-BR (activations): λ={args.lambda_br_act} {'[ENABLED]' if args.lambda_br_act > 0 else '[DISABLED]'}")
    print(f"    └─> Gradients to alpha: {args.br_backprop_to_alpha_act} (default: False, recommended)")
    print(f"\nAlpha Control:")
    print(f"  - Freeze weight alpha after warmup: {args.freeze_weight_alpha}")
    print(f"    └─> If True: NO updates to W alpha from ANY loss (CE or W-BR)")
    print(f"    └─> If False: W alpha updated by CE + W-BR (paper-faithful co-evolution)")
    print(f"  - Freeze activation alpha after warmup: {args.freeze_act_alpha}")
    print(f"    └─> If True: NO updates to A alpha from ANY loss (CE or A-BR)")
    print(f"    └─> If False: A alpha updated by CE, A-BR respects --br-backprop-to-alpha-act flag")
    print(f"{'='*80}\n")

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.qat_epochs + 1):
        # Stage 1: Warmup (LSQ only)
        if epoch <= args.warmup_epochs:
            stage = "WARMUP"
            
            # Train
            train_loss, ce_loss, br_w_loss, br_a_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device,
                regularizer_w=None, lambda_br=0.0,
                regularizer_a=None, lambda_br_act=0.0,
                hook_manager=None, br_backprop_to_alpha_act=False, epoch=epoch, writer=writer
            )
            
        # Stage 2: Joint training (LSQ + BR)
        else:
            stage = "BR"
            
            # Freeze alpha parameters if requested (separate control for W and A)
            if epoch == args.warmup_epochs + 1:
                if args.freeze_weight_alpha or args.freeze_act_alpha:
                    print(f"\n{'='*80}")
                    print("FREEZING ALPHA PARAMETERS")
                    print(f"{'='*80}")
                
                if args.freeze_weight_alpha:
                    print(f"✓ Freezing {len(weight_alpha_params)} weight alpha parameters")
                    for param in weight_alpha_params:
                        param.requires_grad = False
                
                if args.freeze_act_alpha:
                    print(f"✓ Freezing {len(act_alpha_params)} activation alpha parameters")
                    for param in act_alpha_params:
                        param.requires_grad = False
                
                if args.freeze_weight_alpha or args.freeze_act_alpha:
                    print(f"{'='*80}\n")
            
            # Train with BR (W-BR and/or A-BR depending on lambda values)
            train_loss, ce_loss, br_w_loss, br_a_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device,
                regularizer_w=regularizer_w, lambda_br=args.lambda_br,
                regularizer_a=regularizer_a, lambda_br_act=args.lambda_br_act,
                hook_manager=hook_manager, br_backprop_to_alpha_act=args.br_backprop_to_alpha_act, epoch=epoch, writer=writer
            )

        # Step LR scheduler (ONLY during BR phase, not warmup)
        if epoch > args.warmup_epochs:
            scheduler.step()

        # Test
        test_loss, test_acc = test(model, test_loader, criterion, device, epoch=epoch, writer=writer)

        # Print progress (get current LR)
        current_lr = optimizer.param_groups[0]['lr']
        if epoch <= args.warmup_epochs:
            print(f"Epoch {epoch}/{args.qat_epochs} [{stage}] (LR={current_lr:.6f}): "
                  f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
        else:
            print(f"Epoch {epoch}/{args.qat_epochs} [{stage}] (LR={current_lr:.6f}): "
                  f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, "
                  f"BR_W={br_w_loss:.6f}, BR_A={br_a_loss:.6f}")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            checkpoint_path = os.path.join(checkpoint_dir, 'best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'args': args,
            }, checkpoint_path)
            print(f"  → Saved best model: {checkpoint_path}")

    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"✓ TensorBoard logs saved")

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    if args.tensorboard:
        log_dir = os.path.join(args.output_dir, 'tensorboard')
        print(f"TensorBoard logs: {log_dir}")
        print(f"  View with: tensorboard --logdir {log_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
