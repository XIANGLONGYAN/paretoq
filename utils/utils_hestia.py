"""
HESTIA: Hessian-Guided Differentiable QAT for Extremely Low-Bit LLMs

Core module containing:
  1. HestiaQuantizer  — softmax-based differentiable quantization
  2. HestiaScheduler  — three-phase pressure/temperature schedule
  3. HestiaLinear     — drop-in nn.Linear replacement with integrated scheduling
  4. HestiaStepCallback — global-step updater for HuggingFace Trainer integration

The scheduler runs three phases:
  - Compress:  pressure p rises from 0 to 1 (fp → quantized convex combination)
  - Anneal:    temperature τ decays (soft → hard assignments)
  - Solid:     pure discrete quantization (τ = 0, p = 1)

Global state (module-level, shared across all HestiaLinear instances):
  - global_cur_step:    current training step, updated by HestiaStepCallback
  - global_total_steps: total planned training steps, set before training begins
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from transformers import TrainerCallback


# ============================================================================
# Global step tracking
# ============================================================================

global_cur_step: int = 0
global_total_steps: int = 0


class HestiaStepCallback(TrainerCallback):
    """HuggingFace Trainer callback: updates global_cur_step each step."""

    def on_step_end(self, args, state, control, **kwargs):
        global global_cur_step
        global_cur_step = state.global_step




# ============================================================================
# HestiaQuantizer: Softmax-based differentiable quantization
# ============================================================================

class HestiaQuantizer(nn.Module):
    """
    Differentiable quantizer using temperature-controlled softmax.

    Forward (τ > 0):
        logits = -(x_norm - codebook)^2
        prob   = softmax(logits / τ)
        qx     = Σ prob * codebook     ← soft expectation over codebook

    Forward (τ = 0):
        qx = round(clip(x_norm, -1, 1))   ← hard discrete (STE on backward)

    The output is lerp'd with the full-precision input according to pressure.
    """

    def __init__(self, codebook: torch.Tensor, group_size: int = 0):
        super().__init__()
        self.register_buffer("codebook", codebook)
        self.group_size = group_size
        self._cached_codebook = None
        self._cached_device = None
        self._cached_dtype = None

    def forward(
        self,
        x: torch.Tensor,
        pressure: float,
        temp: float,
        is_training: bool = True,
    ) -> torch.Tensor:
        # Refresh cached codebook only on device/dtype change
        if (
            self._cached_codebook is None
            or self._cached_device != x.device
            or self._cached_dtype != x.dtype
        ):
            self._cached_codebook = self.codebook.to(device=x.device, dtype=x.dtype)
            self._cached_device = x.device
            self._cached_dtype = x.dtype

        codebook = self._cached_codebook

        reshaped_x, org_shape = self._reshape_for_grouping(x, self.group_size)
        # scale α = mean(|w|), per-group
        scales = 1.0 / reshaped_x.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        x_norm = reshaped_x * scales

        if temp >= 0.0:
            # Soft quantization via Gibbs distribution
            logits = -(x_norm.unsqueeze(-1) - codebook).pow(2)
            prob = F.softmax(logits / (temp + 1e-6), dim=-1)
            qx_norm = torch.sum(prob * codebook, dim=-1)
            qx = (qx_norm / scales).to(dtype=x.dtype)
        else:
            # Hard quantization (temp = 0)
            qx = x_norm.round().clamp(-1, 1) / scales
            if is_training:
                qx = reshaped_x + (qx - reshaped_x).detach()

        # Convex interpolation: W_eff = (1-p) * W + p * W_quantized
        if pressure >= 1.0:
            return qx.reshape(org_shape)
        elif pressure <= 0.0:
            return x
        else:
            return torch.lerp(x, qx.reshape(org_shape), pressure)

    def _reshape_for_grouping(self, x: torch.Tensor, group_size: int):
        """
        Reshape tensor for group-wise quantization.
        - group_size > 0:  block-wise, last dim split into groups
        - group_size == -1: per-channel (keep last dim)
        - group_size == 0:  per-tensor (flatten all)
        """
        org_shape = x.shape
        if group_size > 0:
            assert org_shape[-1] % group_size == 0
            x = x.reshape(-1, group_size)
        elif group_size == -1:
            x = x.reshape(-1, org_shape[-1])
        elif group_size == 0:
            x = x.reshape(1, -1)
        else:
            raise ValueError(f"Invalid group_size: {group_size}")
        return x, org_shape

def make_ternary_codebook():
    """Codebook for ternary quantization: {-1, 0, +1}."""
    return torch.tensor([-1.0, 0.0, 1.0])


def make_bitwidth_codebook(bits: int, symmetric: bool = True):
    """Codebook for general symmetric/asymmetric quantization."""
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax
        return torch.arange(qmin, qmax + 1, dtype=torch.float32)
    else:
        return torch.arange(0, 2 ** bits, dtype=torch.float32)


# ============================================================================
# HestiaScheduler: pressure + temperature schedule
# ============================================================================

class HestiaScheduler:
    """
    Three-phase scheduler for Hestia training.
    Reads global_cur_step and global_total_steps from module-level state.

    Phase 1 (Compress):  pressure ramps 0 → 1, temperature = init_temp
    Phase 2 (Anneal):    pressure = 1, temperature decays init_temp → end_temp
    Phase 3 (Solid):     pressure = 1, temperature = 0 (hard quantization)

    With temp_scale != 1.0 (from Hessian calibration):
      eff_temp = base_temp * temp_scale
    Higher temp_scale → slower cooling → softer quantization for longer.
    """

    def __init__(
        self,
        compress_ratio: float = 0.2,
        init_temp: float = 1.0,
        end_temp: float = 0.0,
        temp_decay_style: str = "cosine",
        anneal_ratio: float = 0.8,
        temp_scale: Optional[float] = None,
    ):
        if end_temp != 0.0:
            raise NotImplementedError(f"end_temp != 0.0 has not been implemented")
        self.compress_ratio = compress_ratio
        self.init_temp = init_temp
        self.end_temp = end_temp
        self.temp_decay_style = temp_decay_style
        self.anneal_ratio = anneal_ratio
        self.temp_scale = temp_scale

    def get_pressure(self) -> float:
        assert global_total_steps > 0, "global_total_steps must be set before calling get_pressure()"
        ratio = global_cur_step / global_total_steps
        if ratio < self.compress_ratio:
            return ratio / self.compress_ratio
        return 1.0

    def get_temp(self) -> float:
        assert global_total_steps > 0, "global_total_steps must be set before calling get_temp()"

        cur_ratio = global_cur_step / global_total_steps
        const_temp_ratio = 1.0 - self.anneal_ratio

        if cur_ratio <= const_temp_ratio:
            eff_temp = self.init_temp

        elif self.temp_decay_style == "linear":
            eff_temp = self.init_temp * (1 - cur_ratio) / self.anneal_ratio

        elif self.temp_decay_style == "cosine":
            eff_temp = self.init_temp * 0.5 * (1.0 + math.cos(math.pi * (cur_ratio - const_temp_ratio) / self.anneal_ratio))

        if self.temp_scale is not None:
            eff_temp *= self.temp_scale

        return eff_temp


# ============================================================================
# HestiaLinear: nn.Linear with integrated Hestia quantization
# ============================================================================

class HestiaLinear(nn.Linear):
    """
    Linear layer with Hestia thermal quantization.

    Training phases (automatic, driven by global_cur_step):
      1. Compress:  W_eff = lerp(W_fp, W_quant, pressure)
      2. Anneal:    W_eff = softmax_quant(W, τ)   (τ → 0)
      3. Solid:     W_eff = hard_quant(W)

    At inference (eval mode): always uses hard quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: int = 0,
        symmetric: bool = True,
        group_size: int = -1,
        compress_ratio: float = 0.2,
        init_temp: float = 1.0,
        end_temp: float = 0.0,
        anneal_ratio: float = 0.8,
        temp_scale: Optional[float] = None,
        layer_id: Optional[str] = None,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.bits = bits
        self.symmetric = symmetric
        self.layer_id = layer_id

        # For bits >= 16 no quantization happens; skip huge codebook alloc
        if bits >= 16:
            self.quantizer = None
            self.scheduler = None
        else:
            # Build codebook
            if bits == 0 and symmetric:
                codebook = make_ternary_codebook()
            else:
                codebook = make_bitwidth_codebook(bits, symmetric)

            self.quantizer = HestiaQuantizer(codebook, group_size)
            self.scheduler = HestiaScheduler(
                compress_ratio=compress_ratio,
                init_temp=init_temp,
                end_temp=end_temp,
                anneal_ratio=anneal_ratio,
                temp_scale=temp_scale,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bits >= 16:
            return F.linear(input, self.weight, self.bias)

        if self.training and global_total_steps > 0:
            pressure = self.scheduler.get_pressure()
            temp = self.scheduler.get_temp()

            q_weight = self.quantizer(
                self.weight,
                pressure=pressure,
                temp=temp,
                is_training=True,
            )
            weight = q_weight  # quantizer already handles lerp
        else:
            # Eval: hard ternary quantization
            q_weight = self.quantizer(
                self.weight,
                pressure=1.0,
                temp=0.0,
                is_training=False,
            )
            weight = q_weight

        return F.linear(input, weight, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 1,
        symmetric: bool = True,
        group_size: int = 0,
        compress_ratio: float = 0.2,
        init_temp: float = 1.0,
        end_temp: float = 0.0,
        anneal_ratio: float = 0.8,
        temp_scale: Optional[float] = None,
        layer_id: Optional[str] = None,
    ):
        """Create HestiaLinear from existing nn.Linear, copying weights."""
        hestia_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
            symmetric=symmetric,
            group_size=group_size,
            compress_ratio=compress_ratio,
            init_temp=init_temp,
            end_temp=end_temp,
            anneal_ratio=anneal_ratio,
            temp_scale=temp_scale,
            layer_id=layer_id,
        )
        hestia_linear.weight = linear.weight
        if linear.bias is not None:
            hestia_linear.bias = linear.bias
        return hestia_linear


# ============================================================================
# Model replacement utility
# ============================================================================

def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def replace_linear_with_hestia(
    model: nn.Module,
    bits: int = 1,
    symmetric: bool = True,
    group_size: int = 0,
    compress_ratio: float = 0.2,
    init_temp: float = 1.0,
    end_temp: float = 0.0,
    anneal_ratio: float = 0.8,
    temp_scales_dict: Optional[Dict[str, float]] = None,
    skip_keywords: Optional[List[str]] = ["lm_head", "embed"],
):
    """
    Recursively replace nn.Linear with HestiaLinear.

    Args:
        model:              model to modify in-place
        bits:               bit-width (1 = ternary, 2 = 2-bit, etc.)
        symmetric:          symmetric quantization
        group_size:         0 = per-tensor, -1 = per-channel
        compress_ratio:     fraction of training for compress phase
        init_temp:          initial temperature
        end_temp:           final temperature
        anneal_ratio:       fraction of training for anneal phase
        temp_scales_dict:   {layer_id: temp_scale} from Hessian calibration
        skip_keywords:      layer names containing these are skipped

    Returns:
        model (modified in-place)
    """
    layer_counter = [0]
    replace_count = [0]

    def _convert(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            if name in skip_keywords:
                continue
            full_path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not isinstance(child, HestiaLinear):
                layer_id = f"layer_{layer_counter[0]}_{full_path}"
                layer_counter[0] += 1
                temp_scale = temp_scales_dict.get(layer_id) if temp_scales_dict else None
                q_layer = HestiaLinear.from_linear(
                    child,
                    bits=bits,
                    symmetric=symmetric,
                    group_size=group_size,
                    compress_ratio=compress_ratio,
                    init_temp=init_temp,
                    end_temp=end_temp,
                    anneal_ratio=anneal_ratio,
                    temp_scale=temp_scale,
                    layer_id=layer_id,
                )
                setattr(module, name, q_layer)
                replace_count[0] += 1
            else:
                _convert(child, full_path)

    _convert(model)


    print(
        f"[Hestia] Replaced {replace_count[0]} nn.Linear -> HestiaLinear "
        f"(bits={bits}, group_size={group_size}, compress_ratio={compress_ratio})"
    )
    return model
