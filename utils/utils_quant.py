# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn

class LsqBinaryTernaryExtension(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()

        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle * (-q_w + q_w.round())
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle * (-q_w + q_w.round())
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class StretchedElasticQuant(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (num_bits - 1)
            shift = 0.5
        Qp = (n_levels - shift) / n_levels
        Qn = -Qp
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (
                torch.round(
                    torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift
                )
                + shift
            ) / n_levels
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        clip_val = 1 - 1e-2
        if ctx.num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (ctx.num_bits - 1)
            shift = 0.5
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle
                            * (
                                -q_w
                                + (
                                    torch.round(
                                        torch.clamp(q_w, -clip_val, clip_val) * n_levels
                                        - shift
                                    )
                                    + shift
                                )
                                / n_levels
                            )
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle
                        * (
                            -q_w
                            + (
                                torch.round(
                                    torch.clamp(q_w, -clip_val, clip_val) * n_levels
                                    - shift
                                )
                                + shift
                            )
                            / n_levels
                        )
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None



class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        symmetric=True,
        bias=False,
        w_bits=16,
        weight_layerwise=False,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=bias)
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        # params for weight quant
        if self.w_bits < 16:
            self.weight_clip_val = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 16:
            weight = self.weight
        elif self.w_bits == 2 or self.w_bits == 0:
            weight = StretchedElasticQuant.apply(
                real_weights,
                self.weight_clip_val,
                self.w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
        elif self.w_bits <= 4:
            weight = LsqBinaryTernaryExtension.apply(
                real_weights,
                self.weight_clip_val,
                self.w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
        else:
            raise NotImplementedError

        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

    @classmethod
    def from_linear(cls, linear: nn.Linear, w_bits: int, weight_layerwise: bool = False):
        """从现有 nn.Linear 创建 QuantizeLinear，复制权重"""
        quant_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            w_bits=w_bits,
            weight_layerwise=weight_layerwise,
        )
        # 复制权重（共享参数以节省内存）
        quant_linear.weight = linear.weight
        if linear.bias is not None:
            quant_linear.bias = linear.bias
        return quant_linear
    
def replace_linear_with_quantized(
    model: nn.Module,
    w_bits: int,
    weight_layerwise: bool = False,
    skip_keywords: list = None,
):
    """
    递归遍历模型，将 nn.Linear 替换为 QuantizeLinear。
    
    Args:
        model: HuggingFace 模型
        w_bits: 量化位数，>=16 时不替换（保持原始精度）
        weight_layerwise: 是否按行量化
        skip_keywords: 不量化的层名关键词列表，如 ["lm_head", "embed"]
    """
    if w_bits >= 16:
        # 不量化，直接返回
        return model
    
    if skip_keywords is None:
        skip_keywords = ["lm_head", "embed"]
    
    # 收集需要替换的模块（不能在遍历时修改）
    replace_list = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, QuantizeLinear):
            if any(kw in name for kw in skip_keywords):
                continue
            replace_list.append((name, module))
    
    # 执行替换
    for name, module in replace_list:
        quant_linear = QuantizeLinear.from_linear(module, w_bits=w_bits, weight_layerwise=weight_layerwise)
        # 递归 setattr
        _set_module_by_name(model, name, quant_linear)
    
    print(f"[ParetoQ] Replaced {len(replace_list)} nn.Linear with QuantizeLinear (w_bits={w_bits})")
    return model


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """通过点分隔的名称路径设置子模块"""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
