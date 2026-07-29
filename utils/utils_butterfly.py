import math
from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


class ButterflyTransform(nn.Module):
    """
    Learnable orthogonal butterfly transform along the last tensor dimension.

    Each stage pairs features at strides 1, 2, 4, ..., feature_dim // 2 and
    applies independent 2x2 Givens rotations. For row-major tensors this
    computes ``x @ B.T``, matching the ButterflyQuant convention

        W_rot = W @ B.T
        A_rot = A @ B.T.

    Only the Givens angles are trainable. Dense B or B_i matrices are never
    materialized.

    ``init="hadamard"`` initializes B to a block-diagonal concatenation of
    normalized Sylvester Hadamard matrices. Fixed reflection signs are used
    together with Givens rotations to represent each Hadamard butterfly node
    exactly; the angles remain continuously learnable.
    """

    SUPPORTED_INITS = ("identity", "hadamard")

    def __init__(
        self,
        feature_dim: int,
        *,
        init: str = "identity",
        hadamard_block_size: int = 128,
        parameter_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if not isinstance(feature_dim, int) or not _is_power_of_two(feature_dim):
            raise ValueError(
                f"feature_dim must be a positive power of two, got {feature_dim}."
            )
        if init not in self.SUPPORTED_INITS:
            raise ValueError(
                f"Unsupported butterfly init {init!r}; "
                f"expected one of {self.SUPPORTED_INITS}."
            )

        self.feature_dim = feature_dim
        self.num_stages = int(math.log2(feature_dim))
        self.init = init
        self.hadamard_block_size = hadamard_block_size

        theta = torch.zeros(
            self.num_stages,
            feature_dim // 2,
            dtype=parameter_dtype,
            device=device,
        )
        stage_signs = torch.ones(
            self.num_stages,
            dtype=parameter_dtype,
            device=device,
        )

        if init == "hadamard":
            if (
                not isinstance(hadamard_block_size, int)
                or not _is_power_of_two(hadamard_block_size)
            ):
                raise ValueError(
                    "hadamard_block_size must be a positive power of two, "
                    f"got {hadamard_block_size}."
                )
            if hadamard_block_size > feature_dim:
                raise ValueError(
                    f"hadamard_block_size ({hadamard_block_size}) cannot exceed "
                    f"feature_dim ({feature_dim})."
                )
            if feature_dim % hadamard_block_size != 0:
                raise ValueError(
                    f"feature_dim ({feature_dim}) must be divisible by "
                    f"hadamard_block_size ({hadamard_block_size})."
                )

            hadamard_stages = int(math.log2(hadamard_block_size))
            theta[:hadamard_stages].fill_(-math.pi / 4)
            stage_signs[:hadamard_stages] = -1

        self.theta = nn.Parameter(theta)
        self.register_buffer(
            "_stage_signs",
            stage_signs,
            persistent=True,
        )

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, num_stages={self.num_stages}, "
            f"init={self.init!r}, hadamard_block_size={self.hadamard_block_size}"
        )

    def _stage_twiddles(
        self,
        stage: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ):
        stride = 1 << stage
        num_groups = self.feature_dim // (2 * stride)
        theta = self.theta[stage].reshape(num_groups, stride)

        # Trigonometric functions are evaluated in the parameter dtype
        # (FP32 by default), then cast for the tensor arithmetic.
        cos_theta = torch.cos(theta).to(device=device, dtype=dtype)
        sin_theta = torch.sin(theta).to(device=device, dtype=dtype)
        return stride, num_groups, cos_theta, sin_theta

    @staticmethod
    def _broadcast_twiddle(twiddle: torch.Tensor, target_ndim: int) -> torch.Tensor:
        leading_ones = (1,) * (target_ndim - twiddle.ndim)
        return twiddle.reshape(*leading_ones, *twiddle.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected last dimension {self.feature_dim}, got {x.size(-1)}."
            )
        if not x.is_floating_point():
            raise TypeError(
                f"ButterflyTransform requires a floating-point tensor, got {x.dtype}."
            )

        output = x
        leading_shape = x.shape[:-1]

        for stage in range(self.num_stages):
            stride, num_groups, cos_theta, sin_theta = self._stage_twiddles(
                stage,
                dtype=output.dtype,
                device=output.device,
            )
            paired = output.reshape(*leading_shape, num_groups, 2, stride)
            first = paired[..., 0, :]
            second = paired[..., 1, :]

            cos_theta = self._broadcast_twiddle(cos_theta, first.ndim)
            sin_theta = self._broadcast_twiddle(sin_theta, first.ndim)

            rotated_first = cos_theta * first - sin_theta * second
            rotated_second = sin_theta * first + cos_theta * second
            stage_sign = self._stage_signs[stage].to(
                device=output.device,
                dtype=output.dtype,
            )
            rotated_second = stage_sign * rotated_second

            output = torch.stack(
                (rotated_first, rotated_second),
                dim=-2,
            ).reshape(*leading_shape, self.feature_dim)

        return output

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the exact transpose/inverse of the current transform."""
        if x.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected last dimension {self.feature_dim}, got {x.size(-1)}."
            )
        if not x.is_floating_point():
            raise TypeError(
                f"ButterflyTransform requires a floating-point tensor, got {x.dtype}."
            )

        output = x
        leading_shape = x.shape[:-1]

        for stage in reversed(range(self.num_stages)):
            stride, num_groups, cos_theta, sin_theta = self._stage_twiddles(
                stage,
                dtype=output.dtype,
                device=output.device,
            )
            paired = output.reshape(*leading_shape, num_groups, 2, stride)
            first = paired[..., 0, :]
            second = paired[..., 1, :]
            stage_sign = self._stage_signs[stage].to(
                device=output.device,
                dtype=output.dtype,
            )
            second = stage_sign * second

            cos_theta = self._broadcast_twiddle(cos_theta, first.ndim)
            sin_theta = self._broadcast_twiddle(sin_theta, first.ndim)

            restored_first = cos_theta * first + sin_theta * second
            restored_second = -sin_theta * first + cos_theta * second
            output = torch.stack(
                (restored_first, restored_second),
                dim=-2,
            ).reshape(*leading_shape, self.feature_dim)

        return output


class ButterflyQuantizer(nn.Module):
    """
    Symmetric uniform fake quantizer used by ButterflyQuant.

    Quantization is performed per last-dimension group using a max-absolute
    scale. The forward pass uses hard round/clip/dequantize operations, while
    the backward pass uses the straight-through estimator (STE).

    ``group_size`` follows the conventions in ``utils_myquant.py``:

    - ``-1``: one group per row/token (channel-wise)
    - ``0``: one group for the complete tensor (per-tensor)
    - positive integer: groups of that size along the last dimension

    For ``num_bits >= 16`` this module returns its input unchanged. Rotation is
    intentionally not handled here, so ButterflyQuantizeLinear still rotates
    both operands before invoking the quantizers.
    """

    def __init__(
        self,
        num_bits: int,
        *,
        group_size: int = -1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        if not isinstance(num_bits, int):
            raise TypeError(f"num_bits must be an integer, got {type(num_bits)}.")
        if num_bits < 2:
            raise ValueError(
                "ButterflyQuantizer implements the paper's signed uniform "
                f"quantizer and requires num_bits >= 2, got {num_bits}."
            )
        if not isinstance(group_size, int) or group_size < -1:
            raise ValueError(
                f"group_size must be -1, 0, or a positive integer, got {group_size}."
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")

        self.num_bits = num_bits
        self.group_size = group_size
        self.eps = eps

    def extra_repr(self) -> str:
        return (
            f"num_bits={self.num_bits}, group_size={self.group_size}, "
            f"eps={self.eps}"
        )

    def _reshape_groups(self, x: torch.Tensor):
        original_shape = x.shape

        if self.group_size == -1:
            grouped = x.reshape(-1, x.size(-1))
        elif self.group_size == 0:
            grouped = x.reshape(1, -1)
        else:
            if x.size(-1) % self.group_size != 0:
                raise ValueError(
                    f"group_size ({self.group_size}) must divide the last "
                    f"dimension ({x.size(-1)})."
                )
            grouped = x.reshape(-1, self.group_size)

        return grouped, original_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_bits >= 16:
            return x
        if not x.is_floating_point():
            raise TypeError(
                f"ButterflyQuantizer requires a floating-point tensor, got {x.dtype}."
            )

        grouped, original_shape = self._reshape_groups(x)
        quant_min = -(2 ** (self.num_bits - 1))
        quant_max = 2 ** (self.num_bits - 1) - 1

        abs_max = grouped.abs().amax(dim=-1, keepdim=True)
        scale = (abs_max / quant_max).clamp(min=self.eps)
        quantized = torch.round(grouped / scale).clamp(
            min=quant_min,
            max=quant_max,
        )
        dequantized = quantized * scale

        # Hard fake quantization in forward, identity surrogate in backward.
        fake_quantized = grouped + (dequantized - grouped).detach()
        return fake_quantized.reshape(original_shape)


class ButterflyQuantizeLinear(nn.Linear):
    """
    Linear layer with a shared learnable ButterflyQuant transform for W and A.

    For row-major activations and PyTorch weight layout:

        A_rot = A @ B.T
        W_rot = W @ B.T
        output = linear(Q(A_rot), Q(W_rot), bias)

    Since B is orthogonal, the unquantized computation is unchanged. Operands
    with bit-width >= 16 are still rotated but bypass quantization.
    """

    def __init__(
        self,
        *args,
        w_bits: int = 16,
        a_bits: int = 16,
        w_group_size: int = -1,
        a_group_size: int = -1,
        butterfly_init: str = "identity",
        hadamard_block_size: int = 128,
        butterfly_parameter_dtype: torch.dtype = torch.float32,
        layer_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.w_bits = w_bits
        self.a_bits = a_bits
        self.w_group_size = w_group_size
        self.a_group_size = a_group_size
        self.butterfly_init = butterfly_init
        self.hadamard_block_size = hadamard_block_size
        self.layer_id = layer_id

        self.butterfly = ButterflyTransform(
            self.in_features,
            init=butterfly_init,
            hadamard_block_size=hadamard_block_size,
            parameter_dtype=butterfly_parameter_dtype,
            device=self.weight.device,
        )
        self.w_quantizer = ButterflyQuantizer(
            w_bits,
            group_size=w_group_size,
        )
        self.a_quantizer = ButterflyQuantizer(
            a_bits,
            group_size=a_group_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rotated_activation = self.butterfly(x)
        rotated_weight = self.butterfly(self.weight)

        quantized_activation = self.a_quantizer(rotated_activation)
        quantized_weight = self.w_quantizer(rotated_weight)
        return F.linear(quantized_activation, quantized_weight, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        w_bits: int = 16,
        a_bits: int = 16,
        w_group_size: int = -1,
        a_group_size: int = -1,
        butterfly_init: str = "identity",
        hadamard_block_size: int = 128,
        butterfly_parameter_dtype: torch.dtype = torch.float32,
        layer_id: Optional[str] = None,
    ):
        quantized_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            w_bits=w_bits,
            a_bits=a_bits,
            w_group_size=w_group_size,
            a_group_size=a_group_size,
            butterfly_init=butterfly_init,
            hadamard_block_size=hadamard_block_size,
            butterfly_parameter_dtype=butterfly_parameter_dtype,
            layer_id=layer_id,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        # Preserve the original Parameters rather than copying their values.
        quantized_linear.weight = linear.weight
        quantized_linear.bias = linear.bias
        return quantized_linear


def replace_linear_with_butterfly(
    model: nn.Module,
    *,
    w_bits: int = 16,
    a_bits: int = 16,
    w_group_size: int = -1,
    a_group_size: int = -1,
    butterfly_init: str = "identity",
    hadamard_block_size: int = 128,
    butterfly_parameter_dtype: torch.dtype = torch.float32,
    skip_keywords: Optional[Sequence[str]] = ("embed", "lm_head"),
) -> nn.Module:
    """Recursively replace eligible nn.Linear modules in-place."""

    skip_keywords = tuple(skip_keywords or ())
    layer_counter = 0
    replace_count = 0

    def should_skip(path: str) -> bool:
        return any(keyword in path for keyword in skip_keywords)

    def convert(module: nn.Module, prefix: str = "") -> None:
        nonlocal layer_counter, replace_count

        for name, child in list(module.named_children()):
            full_path = f"{prefix}.{name}" if prefix else name
            if should_skip(full_path):
                continue

            if isinstance(child, nn.Linear) and not isinstance(
                child,
                ButterflyQuantizeLinear,
            ):
                layer_id = f"layer_{layer_counter}_{full_path}"
                layer_counter += 1
                replacement = ButterflyQuantizeLinear.from_linear(
                    child,
                    w_bits=w_bits,
                    a_bits=a_bits,
                    w_group_size=w_group_size,
                    a_group_size=a_group_size,
                    butterfly_init=butterfly_init,
                    hadamard_block_size=hadamard_block_size,
                    butterfly_parameter_dtype=butterfly_parameter_dtype,
                    layer_id=layer_id,
                )
                setattr(module, name, replacement)
                replace_count += 1
            else:
                convert(child, full_path)

    convert(model)
    print(
        f"Replaced {replace_count} nn.Linear modules with "
        "ButterflyQuantizeLinear"
    )
    return model


__all__ = [
    "ButterflyTransform",
    "ButterflyQuantizer",
    "ButterflyQuantizeLinear",
    "replace_linear_with_butterfly",
]
