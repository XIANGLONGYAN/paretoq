"""
QuEST: Stable Training of LLMs with 1-Bit Weights and Activations
ICML 2025 - https://github.com/IST-DASLab/QuEST

This module implements the two core innovations of QuEST:
  1. Distribution Fitting: RMS normalization + Hadamard pre-processing
     + MSE-optimal Gaussian grid projection (closed-form, no backprop)
  2. Trust Gradient Estimation: mask-based gradient filtering in the
     Hadamard domain, bounding gradient bias from outlier entries

Pure PyTorch implementation — no external dependencies beyond torch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Precomputed optimal Gaussian scales for MSE-minimizing quantization
# alpha* = argmin_alpha E[(eps - alpha * round(clip(eps/alpha)))^2]
# for eps ~ N(0,1), computed numerically via scipy.integrate
# ============================================================================
OPTIMAL_GAUSSIAN_SCALES = {
    1: 0.7978845587140913,
    1.585: 1.2240089519030855,
    2: 1.4935346200015913,
    3: 2.051068354131873,
    4: 2.513930578568423,
    5: 2.9160938834961225,
    6: 3.276597282593217,
    7: 3.6010497188221655,
    8: 3.884938678807525,
}


# ============================================================================
# Hadamard Transform (block-diagonal, pure PyTorch)
# ============================================================================

def _build_hadamard_128():
    """Build a 128x128 orthonormal Hadamard matrix via Sylvester construction."""
    H = torch.tensor([[1.0]])
    for _ in range(7):  # 2^7 = 128
        H = torch.cat([torch.cat([H, H], dim=1),
                        torch.cat([H, -H], dim=1)], dim=0)
    return H / (128 ** 0.5)


# Global cache — built once, moved to target device via .to()
_H_128 = None


def _get_H_128(device, dtype):
    """Lazy-init the 128x128 Hadamard matrix on the target device."""
    global _H_128
    if _H_128 is None:
        _H_128 = _build_hadamard_128()
    return _H_128.to(device=device, dtype=dtype)


def block_hadamard_transform(x, block_size=128):
    """
    Apply block-diagonal Hadamard transform along the last dimension.

    Partitions the last dim into blocks of `block_size` and multiplies
    each block by the orthonormal Hadamard matrix H_{block_size}.
    Memory-efficient: does NOT materialize the full block-diagonal matrix.
    """
    n = x.shape[-1]
    assert n % block_size == 0, \
        f"Last dim {n} must be divisible by {block_size}, got {n}"
    H = _get_H_128(x.device, x.dtype)
    orig_shape = x.shape
    # reshape: [..., n // block_size, block_size] @ [block_size, block_size]^T
    x_reshaped = x.reshape(-1, n // block_size, block_size)
    x_had = x_reshaped @ H.T
    return x_had.reshape(orig_shape)


def inverse_block_hadamard_transform(x, block_size=128):
    """Inverse block-diagonal Hadamard transform.  H^{-1} = H^T = H (orthonormal)."""
    n = x.shape[-1]
    assert n % block_size == 0
    H = _get_H_128(x.device, x.dtype)
    orig_shape = x.shape
    x_reshaped = x.reshape(-1, n // block_size, block_size)
    x_inv = x_reshaped @ H  # H is symmetric orthonormal: H^{-1} = H
    return x_inv.reshape(orig_shape)


# ============================================================================
# QuEST Quantization — custom autograd.Function for Trust Gradient Estimation
# ============================================================================

class QuestQuantizeFn(torch.autograd.Function):
    """
    QuEST quantization with Trust Gradient Estimation.

    Forward:
      RMS-normalize the tensor -> clamp to [-alpha*, alpha*] ->
      round to uniform grid -> dequantize to floating-point approximation.

    Backward:
      Only propagate gradient through entries whose quantization error
      falls within the "trust threshold" (half a quantization bin width).
      Outlier entries receive zero gradient, bounding the gradient bias.
    """

    @staticmethod
    def forward(ctx, x, bits, trust_scale):
        if bits >= 16:
            return x

        n_levels = 2 ** bits
        alpha_star = OPTIMAL_GAUSSIAN_SCALES[bits]

        # RMS-based per-channel scale
        std = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) + 1e-8
        scale = alpha_star * std

        # Symmetric uniform quantization
        step = 2 * scale / (n_levels - 1)
        x_clipped = torch.clamp(x, -scale, scale)
        x_quantized = torch.round(x_clipped / step + 0.5) * step - step / 2

        # Trust threshold = trust_scale * half-bin-width
        trust_threshold = std * trust_scale * alpha_star / (n_levels - 1)
        mask = (torch.abs(x_quantized - x) <= trust_threshold).to(dtype=x.dtype)

        ctx.save_for_backward(mask)
        ctx.bits = bits

        return x_quantized

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        if ctx.bits >= 16:
            return grad_output, None, None
        # Trust Gradient: zero out gradient for untrusted (high-error) entries
        grad_input = grad_output * mask
        return grad_input, None, None


def quest_quantize(x, bits, trust_scale=1.0):
    """Functional interface to QuEST quantization."""
    if bits >= 16:
        return x
    return QuestQuantizeFn.apply(x, bits, trust_scale)


# ============================================================================
# QuestQuantizeLinear — drop-in replacement for nn.Linear
# ============================================================================

class QuestQuantizeLinear(nn.Linear):
    """
    Linear layer with QuEST quantization-aware training.

    Forward computation (per Algorithm 1 of the paper):
        x_h  = Hadamard(x)          # block-diag Hadamard -> near-Gaussian
        w_h  = Hadamard(W)
        x_hq = quest_quantize(x_h)   # Gaussian-optimal grid + trust mask
        w_hq = quest_quantize(w_h)
        y    = x_hq @ w_hq^T         # inner product preserved by orthogonality

    The Hadamard transform is applied along the matrix-multiplication
    dimension, so the matmul result is in the original domain (up to
    quantization error) without any inverse transform.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        w_bits=4,
        a_bits=4,
        hadamard_block_size=128,
        trust_scale_weight=1.0,
        trust_scale_act=1.0,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.hadamard_block_size = hadamard_block_size
        self.trust_scale_weight = trust_scale_weight
        self.trust_scale_act = trust_scale_act

        assert in_features % hadamard_block_size == 0, (
            f"in_features ({in_features}) must be divisible by "
            f"block_size ({hadamard_block_size})"
        )

    def _hadamard(self, x):
        return block_hadamard_transform(x, self.hadamard_block_size)
    
    def _inverse_hadmard(self, x):
        return inverse_block_hadamard_transform(x, self.hadamard_block_size)

    def forward(self, input):
        x_had = self._hadamard(input)

        x_had_q = quest_quantize(x_had, self.a_bits, self.trust_scale_act)

        x_had_q_inv = self._inverse_hadmard(x_had_q)

        w_had = self._hadamard(self.weight)

        w_had_q = quest_quantize(w_had, self.w_bits, self.trust_scale_weight)

        w_had_q_inv = self._inverse_hadmard(w_had_q)

        output = F.linear(x_had_q_inv, w_had_q_inv, self.bias)

        return output

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        w_bits=4,
        a_bits=4,
        hadamard_block_size=128,
        trust_scale_weight=1.0,
        trust_scale_act=1.0,
    ):
        """Create a QuestQuantizeLinear from an existing nn.Linear, copying weights."""
        quest_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            w_bits=w_bits,
            a_bits=a_bits,
            hadamard_block_size=hadamard_block_size,
            trust_scale_weight=trust_scale_weight,
            trust_scale_act=trust_scale_act,
        )
        quest_linear.weight = linear.weight
        if linear.bias is not None:
            quest_linear.bias = linear.bias
        return quest_linear


# ============================================================================
# Model replacement utility
# ============================================================================

def _set_module_by_name(model, name, new_module):
    """Set a submodule by dotted name, handling integer indices for ModuleList."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def replace_linear_with_quest(
    model: nn.Module,
    w_bits: int = 4,
    a_bits: int = 4,
    hadamard_block_size: int = 128,
    trust_scale_weight: float = 1.0,
    trust_scale_act: float = 1.0,
    skip_keywords: list = None,
):
    """
    Recursively replace nn.Linear layers with QuestQuantizeLinear.

    Layers whose name contains any keyword in `skip_keywords` are left unchanged.
    Layers whose in_features is not divisible by `hadamard_block_size` are skipped
    with a warning (e.g. the post-attention projection in some architectures).

    Args:
        model:                 nn.Module to modify in-place.
        w_bits:                weight quantization bit-width.
        a_bits:                activation quantization bit-width.
        hadamard_block_size:   block size for the Hadamard transform (power of 2).
        trust_scale_weight:    multiplier for weight trust threshold (default 1.0).
        trust_scale_act:       multiplier for activation trust threshold (default 1.0).
        skip_keywords:         list of substrings; matching layer names are skipped.

    Returns:
        model (modified in-place).
    """
    if w_bits >= 16 and a_bits >= 16:
        return model

    if skip_keywords is None:
        skip_keywords = ["lm_head", "embed"]

    # Collect replacements first to avoid mutating during iteration
    replace_list = []
    skipped_blocks = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, QuestQuantizeLinear):
            continue
        if any(kw in name for kw in skip_keywords):
            continue
        if module.in_features % hadamard_block_size != 0:
            skipped_blocks.append((name, module.in_features))
            continue
        replace_list.append((name, module))

    for name, module in replace_list:
        quest_linear = QuestQuantizeLinear.from_linear(
            module,
            w_bits=w_bits,
            a_bits=a_bits,
            hadamard_block_size=hadamard_block_size,
            trust_scale_weight=trust_scale_weight,
            trust_scale_act=trust_scale_act,
        )
        _set_module_by_name(model, name, quest_linear)

    if skipped_blocks:
        print(
            f"[QuEST] Skipped {len(skipped_blocks)} layers (in_features not divisible by "
            f"{hadamard_block_size}): {skipped_blocks[:5]}"
            f"{'...' if len(skipped_blocks) > 5 else ''}"
        )

    print(
        f"[QuEST] Replaced {len(replace_list)} nn.Linear -> QuestQuantizeLinear "
        f"(w_bits={w_bits}, a_bits={a_bits})"
    )
    return model
