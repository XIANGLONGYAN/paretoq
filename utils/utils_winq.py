"""
WinQ: Accelerating QAT of Language Models Around Saddle Points
ICML 2026 - https://github.com/facebookresearch/WinQ

WinQ is a training accelerator — NOT a quantization method.
It wraps any existing quantized Linear layer and applies:
  1. Noise Injection:  W += N(0,σ²) before forward, then revert
  2. Weight Re-init:    periodically reset W ← lerp(W, Q(W), α)

WinQ is orthogonal to all quantization methods (baseline / QuEST / Hestia)
and can be combined freely: --use_quest True --use_winq True, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Type
from transformers import TrainerCallback


# ============================================================================
# WinQLinear — wraps any quantized nn.Linear subclass
# ============================================================================

class WinQLinear(nn.Module):
    """
    Wraps a quantized Linear layer with WinQ acceleration.

    Noise Injection (every step):
      W += U  (U ~ N(0, σ²)), forward through inner layer, W -= U

    Weight Re-initialization (periodic, via callback):
      Standard:   W ← (1-α)·W + α·Q(W)
      Hadamard:   W ← Hᵀ((1-α)·HW + α·Q(HW))   [QuEST-compatible, Eq.3]
    """

    def __init__(self, inner_linear: nn.Module, sigma: float = 1e-3, alpha: float = 0.3):
        super().__init__()
        self.inner = inner_linear
        self.sigma = sigma
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Forward-delegated attributes (make WinQLinear transparent)
    # ------------------------------------------------------------------
    @property
    def weight(self):
        return self.inner.weight

    @property
    def bias(self):
        return self.inner.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.sigma > 0:
            noise = torch.randn_like(self.inner.weight) * self.sigma
            self.inner.weight.data.add_(noise)
            try:
                out = self.inner(x)
            finally:
                self.inner.weight.data.sub_(noise)
            return out
        return self.inner(x)

    # ------------------------------------------------------------------
    # Compute Q(W) — method-specific quantized weight
    # ------------------------------------------------------------------
    def _compute_qw(self) -> torch.Tensor:
        """Return the quantized version of the inner layer's weight."""
        w = self.inner.weight
        cls_name = self.inner.__class__.__name__

        if cls_name == "QuantizeLinear":
            # Baseline QAT quantizer
            from utils.utils_quant import DynamicQuantize
            sine_amp = getattr(self.inner, "sine_amplitude", None)
            qw = DynamicQuantize.apply(
                w, self.inner.w_bits, self.inner.weight_asymmetric, sine_amp
            )
        elif cls_name == "QuestQuantizeLinear":
            # QuEST: quantize in Hadamard domain, then IHT back for weight reset
            from utils.utils_quest import (
                block_hadamard_transform,
                inverse_block_hadamard_transform,
                quest_quantize,
            )
            bs = self.inner.hadamard_block_size
            w_had = block_hadamard_transform(w, bs)
            w_had_q = quest_quantize(w_had, self.inner.w_bits, self.inner.trust_scale_weight)
            qw = inverse_block_hadamard_transform(w_had_q, bs)
        elif cls_name == "HestiaLinear":
            # Hestia: run quantizer in hard mode
            qw = self.inner.quantizer(w, pressure=1.0, temp=0.0, is_training=False)
        else:
            raise TypeError(f"WinQ: unsupported inner layer type '{cls_name}'")

        return qw

    # ------------------------------------------------------------------
    # Weight re-initialization  (W ← lerp(W, Q(W), α))
    # For Hadamard methods, follows Eq.3 of WinQ paper.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset_weights(self, alpha: Optional[float] = None):
        """
        Periodic weight re-initialization.

        Standard:   W ← (1-α)·W + α·Q(W)
        Hadamard:   W ← Hᵀ((1-α)·HW + α·Q(HW))
        """
        if alpha is None:
            alpha = self.alpha

        cls_name = self.inner.__class__.__name__
        w = self.inner.weight
        qw = self._compute_qw()

        if cls_name == "QuestQuantizeLinear":
            # For Hadamard methods: lerp in Hadamard domain, then IHT
            from utils.utils_quest import (
                block_hadamard_transform,
                inverse_block_hadamard_transform,
            )
            bs = self.inner.hadamard_block_size
            w_had = block_hadamard_transform(w, bs)
            qw_had = block_hadamard_transform(qw, bs)
            w_had_new = w_had.lerp(qw_had, alpha)
            w_new = inverse_block_hadamard_transform(w_had_new, bs)
            w.copy_(w_new)
        else:
            w.lerp_(qw, alpha)

    def extra_repr(self) -> str:
        return f"sigma={self.sigma}, alpha={self.alpha}, inner={self.inner.__class__.__name__}"


# ============================================================================
# WinQ Callback — triggers periodic weight re-initialization
# ============================================================================

class WinQCallback(TrainerCallback):
    """
    HuggingFace Trainer callback: periodically triggers weight reset
    on all WinQLinear layers.

    Args:
        winq_layers:  list of WinQLinear instances
        interval:     number of steps between resets (K in the paper)
    """

    def __init__(self, winq_layers: List[WinQLinear], interval: int = 40000):
        self.winq_layers = winq_layers
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step % self.interval == 0:
            for layer in self.winq_layers:
                layer.reset_weights()
            # Optional: log the event
            # print(f"[WinQ] Weight reset at step {step}")


# ============================================================================
# Model wrapping utility
# ============================================================================

def apply_winq_to_model(
    model: nn.Module,
    sigma: float = 1e-3,
    alpha: float = 0.2,
) -> List[WinQLinear]:
    """
    Wrap all quantized Linear layers in the model with WinQLinear.

    The model is scanned once; each quantized layer (QuantizeLinear,
    QuestQuantizeLinear, or HestiaLinear) is replaced by a WinQLinear
    wrapper that delegates to it.

    Pure nn.Linear layers are left untouched (WinQ needs a quantizer).

    Args:
        model:  model (modified in-place)
        sigma:  noise standard deviation
        alpha:  interpolation weight for periodic reset

    Returns:
        List of all created WinQLinear wrappers (for callback registration).
    """
    winq_layers: List[WinQLinear] = []

    try:
        from utils.utils_myquant import MyQuantizeLinear
    except ImportError:
        MyQuantizeLinear = type(None)

    try:
        from utils.utils_quant import QuantizeLinear
    except ImportError:
        QuantizeLinear = type(None)

    try:
        from utils.utils_quest import QuestQuantizeLinear
    except ImportError:
        QuestQuantizeLinear = type(None)

    try:
        from utils.utils_hestia import HestiaLinear
    except ImportError:
        HestiaLinear = type(None)

    target_types = tuple(
        t for t in (MyQuantizeLinear, QuantizeLinear, QuestQuantizeLinear, HestiaLinear)
        if t is not None and t is not type(None)
    )

    def _wrap(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, target_types) and not isinstance(child, WinQLinear):
                wrapper = WinQLinear(child, sigma=sigma, alpha=alpha)
                setattr(module, name, wrapper)
                winq_layers.append(wrapper)
            elif isinstance(child, WinQLinear):
                winq_layers.append(child)
            else:
                _wrap(child, full_path)

    _wrap(model)

    print(
        f"[WinQ] Wrapped {len(winq_layers)} quantized layers "
        f"(sigma={sigma}, alpha={alpha})"
    )
    return winq_layers
