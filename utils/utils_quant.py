# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn

def stableqat_surrogate_gradient(v, amplitude):
    """计算 StableQAT 的傅里叶余弦平滑梯度系数 g(v)"""
    item = torch.pi * (v + v.round())
    sum_term = 0
    for idx in range(len(amplitude)):
        sum_term += amplitude[idx] * torch.cos((2 * idx + 1) * item)
    
    denom = 1 + pow(2, 0.5) * torch.pi * sum_term
    denom = torch.clamp(denom, min=1e-5)
    grad_x = (1 - pow(2, 0.5) * torch.pi * sum_term) / denom
    return grad_x


class DynamicActivationQuant(torch.autograd.Function):
    """
    动态自适应激活值量化算子 (Token-wise)
    在前向中基于 Token 的实际极值计算 scale 并在整数空间进行 rounding
    """
    @staticmethod
    def forward(ctx, input, num_bits, asymmetric=False):
        if num_bits >= 16:
            return input
        
        dim = -1
        eps = 1e-5
        
        if asymmetric:
            Qn = 0
            Qp = 2**num_bits - 1
            min_val = torch.min(input, dim=dim, keepdim=True)[0]
            max_val = torch.max(input, dim=dim, keepdim=True)[0]
            scale = (max_val - min_val).clamp(min=eps) / Qp
            zp = torch.round(-min_val / scale).clamp(Qn, Qp)
            
            q_x = torch.clamp(torch.round(input / scale) + zp, Qn, Qp)
            x_q = (q_x - zp) * scale
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1
            max_val = torch.max(torch.abs(input), dim=dim, keepdim=True)[0]
            scale = max_val.clamp(min=eps) / Qp
            
            q_x = torch.clamp(torch.round(input / scale), Qn, Qp)
            x_q = q_x * scale
            
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class LsqActivationQuant(torch.autograd.Function):
    """
    学习型激活值量化算子 (LSQ Activation)
    基于优化器梯度的自适应 step-size 量化
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, asymmetric=False):
        ctx.num_bits = num_bits
        ctx.asymmetric = asymmetric
        
        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)
        
        grad_scale = 1.0 / math.sqrt(input.numel())
        
        if asymmetric:
            # Simple symmetric/asymmetric support for LSQ activation
            Qn = 0
            Qp = 2**num_bits - 1
            min_val = torch.min(input)
            zp = torch.round(-min_val / alpha).clamp(Qn, Qp)
            
            q_x = torch.clamp(torch.round(input / alpha) + zp, Qn, Qp)
            x_q = (q_x - zp) * alpha
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1
            q_x = (input / alpha).round().clamp(Qn, Qp)
            x_q = q_x * alpha

        ctx.save_for_backward(input, alpha)
        if asymmetric:
            ctx.other = grad_scale, Qn, Qp, zp
        else:
            ctx.other = grad_scale, Qn, Qp, 0
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, zp = ctx.other
        
        q_x = input_ / alpha
        if ctx.asymmetric:
            indicate_small = (q_x + zp < Qn).float()
            indicate_big = (q_x + zp > Qp).float()
            indicate_middle = 1.0 - indicate_small - indicate_big
            
            grad_alpha = (
                (
                    indicate_small * Qn
                    + indicate_big * Qp
                    + indicate_middle * (-q_x + (q_x + zp).round() - zp)
                )
                * grad_output
                * grad_scale
            ).sum().unsqueeze(dim=0)
        else:
            indicate_small = (q_x < Qn).float()
            indicate_big = (q_x > Qp).float()
            indicate_middle = 1.0 - indicate_small - indicate_big
            
            grad_alpha = (
                (
                    indicate_small * Qn
                    + indicate_big * Qp
                    + indicate_middle * (-q_x + q_x.round())
                )
                * grad_output
                * grad_scale
            ).sum().unsqueeze(dim=0)
        
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class LsqBinaryTernaryExtension(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise, sine_soft_q=None):
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
        ctx.sine_soft_q = sine_soft_q
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None, None

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

        if ctx.sine_soft_q is not None and ctx.sine_soft_q.get('enable', False):
            grad_x = stableqat_surrogate_gradient(q_w, ctx.sine_soft_q['amplitude'])
            grad_input = indicate_middle * grad_output * grad_x
        else:
            grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None, None


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


def _quantize_activation_kernel(input, alpha, num_bits, use_lsq=False, asymmetric=False):
    if num_bits >= 16:
        return input
    if use_lsq:
        return LsqActivationQuant.apply(input, alpha, num_bits, asymmetric)
    else:
        return DynamicActivationQuant.apply(input, num_bits, asymmetric)


def quantize_activation(
    input_, 
    act_clip_val, 
    a_bits, 
    use_lsq_activation=False, 
    use_asymmetric_act=False,
    dtype=torch.float32
):
    """
    高阶激活量化管理函数
    负责在首个前向传播时进行 LSQ 自适应初始化，并路由调用相应量化算子
    """
    if a_bits >= 16:
        return input_

    # Self-initialization for LSQ act_clip_val in first forward pass
    if use_lsq_activation and (act_clip_val.device != input_.device or torch.all(act_clip_val == 1.0)):
        with torch.no_grad():
            if use_asymmetric_act:
                Qp = 2**a_bits - 1
                val = (input_.max() - input_.min()).clamp(min=1e-5) / Qp
            else:
                Qp = 2**(a_bits - 1) - 1
                val = input_.abs().max().clamp(min=1e-5) / Qp
            act_clip_val.data.fill_(val.item() if hasattr(val, 'item') else val)

    return _quantize_activation_kernel(
        input_, 
        act_clip_val, 
        a_bits, 
        use_lsq=use_lsq_activation, 
        asymmetric=use_asymmetric_act
    ).to(dtype)


def quantize_weight(
    weight,
    weight_clip_val,
    w_bits,
    weight_layerwise=False,
    use_lsq_weight=False,
    use_stableqat=False,
    sine_amplitude=None,
    dtype=torch.float32
):
    """
    高阶权重量化管理函数
    负责在非 LSQ 下的动态 Max 尺度计算，并路由调用 StretchedElastic 或 Lsq 算子
    """
    if w_bits >= 16:
        return weight

    if not use_lsq_weight:
        with torch.no_grad():
            if w_bits == 2 or w_bits == 0:
                scale, _ = torch.max(torch.abs(weight), dim=-1, keepdim=True)
            elif w_bits <= 4:
                xmax, _ = torch.max(torch.abs(weight), dim=-1, keepdim=True)
                maxq = 2 ** (w_bits - 1) - 1
                scale = xmax / maxq
            else:
                raise NotImplementedError
            weight_clip_val.copy_(scale)

    if w_bits == 2 or w_bits == 0:
        q_weight = StretchedElasticQuant.apply(
            weight,
            weight_clip_val,
            w_bits,
            weight_layerwise,
        ).to(dtype)
    elif w_bits <= 4:
        sine_soft_q = {
            'enable': use_stableqat,
            'amplitude': sine_amplitude if use_stableqat else None
        }
        q_weight = LsqBinaryTernaryExtension.apply(
            weight,
            weight_clip_val,
            w_bits,
            weight_layerwise,
            sine_soft_q,
        ).to(dtype)
    else:
        raise NotImplementedError

    return q_weight


class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        symmetric=True,
        bias=False,
        w_bits=16,
        a_bits=16,
        weight_layerwise=False,
        use_stableqat: bool = False,
        use_lsq_weight: bool = False,
        use_lsq_activation: bool = False,
        use_asymmetric_act: bool = False
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=bias)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.weight_layerwise = weight_layerwise
        self.use_stableqat = use_stableqat
        self.use_lsq_weight = use_lsq_weight
        self.use_lsq_activation = use_lsq_activation
        self.use_asymmetric_act = use_asymmetric_act

        # weight_layerwise 暂时不支持 True，会和 LsqBinaryTernaryExtension 的 alpha 维度冲突
        if self.weight_layerwise:
            raise NotImplementedError("weight_layerwise is not supported yet.")
        
        # 注册为 PyTorch Buffer，支持自动设备迁移与多卡分布式训练
        if use_stableqat:
            self.register_buffer('sine_amplitude', torch.tensor([0.21]))
        else:
            self.register_buffer('sine_amplitude', torch.tensor([]))
            
        # params for weight quant
        if self.w_bits < 16:
            if self.use_lsq_weight:
                self.weight_clip_val = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
            else:
                self.register_buffer('weight_clip_val', torch.Tensor(self.weight.shape[0], 1))
                
        # params for activation quant
        if self.a_bits < 16:
            if self.use_lsq_activation:
                self.act_clip_val = nn.Parameter(torch.ones(1))
            else:
                self.register_buffer('act_clip_val', torch.ones(1))

    def forward(self, input_):
        # 1. 量化激活值
        if self.a_bits < 16:
            input_q = quantize_activation(
                input_, 
                self.act_clip_val, 
                self.a_bits, 
                use_lsq_activation=self.use_lsq_activation, 
                use_asymmetric_act=self.use_asymmetric_act,
                dtype=input_.dtype
            )
        else:
            input_q = input_

        # 2. 量化权重
        if self.w_bits < 16:
            weight_q = quantize_weight(
                self.weight,
                self.weight_clip_val,
                self.w_bits,
                weight_layerwise=self.weight_layerwise,
                use_lsq_weight=self.use_lsq_weight,
                use_stableqat=self.use_stableqat,
                sine_amplitude=self.sine_amplitude,
                dtype=input_.dtype
            )
        else:
            weight_q = self.weight

        # 3. 线性计算
        out = nn.functional.linear(input_q, weight_q)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

    @classmethod
    def from_linear(
        cls, 
        linear: nn.Linear, 
        w_bits: int, 
        a_bits: int = 16, 
        weight_layerwise: bool = False, 
        use_stableqat: bool = False, 
        use_lsq_weight: bool = False, 
        use_lsq_activation: bool = False, 
        use_asymmetric_act: bool = False
    ):
        """从现有 nn.Linear 创建 QuantizeLinear，复制权重"""
        quant_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            w_bits=w_bits,
            a_bits=a_bits,
            weight_layerwise=weight_layerwise,
            use_stableqat=use_stableqat,
            use_lsq_weight=use_lsq_weight,
            use_lsq_activation=use_lsq_activation,
            use_asymmetric_act=use_asymmetric_act,
        )
        # 复制权重（共享参数以节省内存）
        quant_linear.weight = linear.weight
        if linear.bias is not None:
            quant_linear.bias = linear.bias
        return quant_linear

def replace_linear_with_quantized(
    model: nn.Module,
    w_bits: int,
    a_bits: int = 16,
    weight_layerwise: bool = False,
    skip_keywords: list = None,
    use_stableqat: bool = False,
    use_lsq_weight: bool = False,
    use_lsq_activation: bool = False,
    use_asymmetric_act: bool = False,
):
    """
    递归遍历模型，将 nn.Linear 替换为 QuantizeLinear。
    
    Args:
        model: HuggingFace 模型
        w_bits: 权重量化位数，>=16 时不替换（保持原始精度）
        a_bits: 激活量化位数，>=16 时不量化激活值
        weight_layerwise: 是否按行量化
        skip_keywords: 不量化的层名关键词列表，如 ["lm_head", "embed"]
        use_stableqat: 是否启用 StableQAT
        use_lsq_weight: 是否对权重启用 LSQ
        use_lsq_activation: 是否对激活启用 LSQ
        use_asymmetric_act: 是否使用非对称激活值量化
    """
    if w_bits >= 16 and a_bits >= 16:
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
        quant_linear = QuantizeLinear.from_linear(
            module, 
            w_bits=w_bits, 
            a_bits=a_bits, 
            weight_layerwise=weight_layerwise, 
            use_stableqat=use_stableqat, 
            use_lsq_weight=use_lsq_weight, 
            use_lsq_activation=use_lsq_activation, 
            use_asymmetric_act=use_asymmetric_act
        )
        # 递归 setattr
        _set_module_by_name(model, name, quant_linear)
    
    print(f"[ParetoQ] Replaced {len(replace_list)} nn.Linear with QuantizeLinear (w_bits={w_bits}, a_bits={a_bits}, use_stableqat={use_stableqat}, use_lsq_weight={use_lsq_weight}, use_lsq_activation={use_lsq_activation})")
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
