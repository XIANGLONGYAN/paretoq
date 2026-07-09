# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn



class DynamicQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits, asymmetric=False, sine_amplitude=None):
        if num_bits >= 16:
            return input
        
        ctx.num_bits = num_bits
        ctx.sine_amplitude = sine_amplitude

        if asymmetric and sine_amplitude is not None:
            raise NotImplementedError("StableQAT does not support asymmetric quantization yet.")
   
        if asymmetric:
            Qn = 0
            Qp = 2**num_bits - 1
            max_val = torch.max(input, dim=-1, keepdim=True)[0]
            min_val = torch.min(input, dim=-1, keepdim=True)[0]
            scale = (max_val - min_val) / (Qp - Qn)
            scale = torch.clamp(scale, min=1e-5)
            zp = torch.round(-min_val / scale)
            quantized = torch.round(input / scale) + zp
            indicate_middle = ((Qn <= quantized) & (quantized <= Qp)).float()
            quantized = torch.clamp(quantized, Qn, Qp)
            dequantized = (quantized - zp) * scale
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1
            abs_max = torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
            scale = abs_max / Qp
            scale = torch.clamp(scale, min=1e-5)

            scaled_input = input / scale
            quantized = torch.round(scaled_input)

            indicate_middle = ((Qn <= quantized) & (quantized <= Qp)).float()
            quantized = torch.clamp(quantized, Qn, Qp)
            dequantized = quantized * scale
        
        ctx.save_for_backward(indicate_middle, scaled_input if not asymmetric else None)

        return dequantized
    
    @staticmethod
    def backward(ctx, grad_output):

        indicate_middle, scaled_input = ctx.saved_tensors

        if ctx.num_bits >= 16:
            return grad_output, None, None, None
        if ctx.sine_amplitude is None:
            return grad_output * indicate_middle, None, None, None

        def stableqat_surrogate_gradient(v, amplitude):
            item = torch.pi * (v + v.round())
            sum_term = 0
            for idx in range(len(amplitude)):
                sum_term += amplitude[idx] * torch.cos((2 * idx + 1) * item)
            
            denom = 1 + 2**0.5 * torch.pi * sum_term
            denom = torch.clamp(denom, min=1e-5)
            grad_x = (1 - 2**0.5 * torch.pi * sum_term) / denom
            return grad_x
        
        grad_input = grad_output * stableqat_surrogate_gradient(scaled_input, ctx.sine_amplitude) * indicate_middle
        return grad_input, None, None, None

class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *args,
        w_bits=16,
        a_bits=16,
        weight_asymmetric=False,
        act_asymmetric=True,
        use_stableqat=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.weight_asymmetric = weight_asymmetric
        self.act_asymmetric = act_asymmetric
        
        self.use_stableqat = use_stableqat

        self.sine_amplitude = [0.21] if self.use_stableqat else None


    def forward(self, input):
        # 这里也可以设置 StableQAT 试一试
        input = DynamicQuantize.apply(input, self.a_bits, self.act_asymmetric, None)
        weight = DynamicQuantize.apply(self.weight, self.w_bits, self.weight_asymmetric, self.sine_amplitude if self.use_stableqat else None)
        output = nn.functional.linear(input, weight, self.bias)
        return output
        

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear, 
        w_bits=16,
        a_bits=16,
        weight_asymmetric=False,
        act_asymmetric=True,
        use_stableqat=False,
    ):
        quant_linear = cls(in_features=linear.in_features, out_features=linear.out_features, bias=linear.bias is not None,
                           w_bits=w_bits, a_bits=a_bits, weight_asymmetric=weight_asymmetric, act_asymmetric=act_asymmetric,
                           use_stableqat=use_stableqat)
        quant_linear.weight = linear.weight
        if linear.bias is not None:
            quant_linear.bias = linear.bias
        return quant_linear

def replace_linear_with_quantized(
    model: nn.Module,
    w_bits: int = 16,
    a_bits: int = 16,
    weight_asymmetric: bool = False,
    act_asymmetric: bool = True,
    skip_keywords: list = ["lm_head", "embed"], # skip replacing Linear layers that contain these keywords in their names
    use_stableqat: bool = False,
    use_robusttraining: bool = False,
    robusttraining_lambda_: float = 0.01
):
    if w_bits >= 16 and a_bits >= 16:
        raise ValueError("Both w_bits and a_bits are >= 16, no need to replace Linear with QuantizeLinear.")
    
    replace_class = QuantizeLinear
    if use_robusttraining:
        replace_class = RobustTrainingQuantizeLinear
    
    replace_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, replace_class):
            if any(kw in name for kw in skip_keywords):
                continue
            _set_module_by_name(model, name, replace_class.from_linear(
                module, 
                w_bits=w_bits, 
                a_bits=a_bits, 
                weight_asymmetric=weight_asymmetric, 
                act_asymmetric=act_asymmetric, 
                use_stableqat=use_stableqat,
                lambda_=robusttraining_lambda_
            ))
            replace_count += 1
    
    print(f"{replace_count} nn.Linear replaced with {replace_class.__name__}.")
    return model

def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    setattr(parent, parts[-1], new_module)


class RobustTrainingQuantizeLinear(QuantizeLinear):
    @staticmethod
    def robust_training_quantize(input, num_bits, lambda_, asymmetric=False):
        if num_bits >= 16:
            return input
        
        if asymmetric:
            Qn, Qp = 0, 2**num_bits - 1
            max_val, min_val = torch.max(input, dim=-1, keepdim=True)[0], torch.min(input, dim=-1, keepdim=True)[0]
            sf = (max_val - min_val) / (Qp - Qn)
            sf = torch.clamp(sf, min=1e-5).detach()
            scaled = (input - min_val) / sf
            q = scaled + (torch.round(scaled) - scaled).detach()

            x_mean = torch.mean(input, dim=-1, keepdim=True)
            q_mean = torch.mean(q, dim=-1, keepdim=True)
            x_centered = (input - x_mean).detach()
            q_centered = q - q_mean
            Cov_xq = torch.mean(x_centered * q_centered, dim=-1, keepdim=True)
            Var_q = torch.mean(q_centered * q_centered, dim=-1, keepdim=True)
            sg = Cov_xq / (Var_q + lambda_)
            x_hat = sg * (q - q_mean) + x_mean
            return x_hat
        
        Qn, Qp = -2**(num_bits - 1), 2**(num_bits - 1) - 1
        abs_max = torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
        sf = abs_max / Qp
        sf = torch.clamp(sf, min=1e-5).detach()
        scaled = input / sf
        q = scaled + (torch.round(scaled) - scaled).detach()
        sg = torch.mean(input.detach() * q, dim=-1, keepdim=True) / (torch.mean(q * q, dim=-1, keepdim=True) + lambda_)
        x_hat = sg * q
        return x_hat

    def __init__(self, lambda_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_ = lambda_

    def forward(self, input):
        input = self.robust_training_quantize(input, self.a_bits, self.lambda_, self.act_asymmetric)
        weight = self.robust_training_quantize(self.weight, self.w_bits, self.lambda_, self.weight_asymmetric)
        output = nn.functional.linear(input, weight, self.bias)
        return output
    
    @classmethod
    def from_linear(cls, lambda_, *args, **kwargs):
        return cls(lambda_, *args, **kwargs)
    



'''
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


class DsqQuant(torch.autograd.Function):
    """
    Differentiable Soft Quantization (DSQ) Custom Autograd Operator.
    Differentiable in both forward and backward pass for x, alpha, l, and u.
    """
    @staticmethod
    def forward(ctx, x, alpha, l, u, num_bits):
        ctx.num_bits = num_bits
        
        # Ensure numerical safety for l and u
        l = torch.min(l, u)
        u = torch.max(l, u)
        
        # Clip x to [l, u]
        x_clip = torch.clamp(x, l, u)
        
        # Compute Delta
        Delta = (u - l) / (2**num_bits - 1)
        Delta = torch.clamp(Delta, min=1e-5)
        
        # Find interval index i
        i = torch.floor((x_clip - l) / Delta).long()
        i = torch.clamp(i, 0, 2**num_bits - 2)
        
        m_i = l + (i.float() + 0.5) * Delta
        
        # Constrain alpha to safe range (0, 0.5]
        alpha_c = torch.clamp(alpha, 1e-4, 0.5)
        
        s = 1.0 / (1.0 - alpha_c)
        k = (1.0 / Delta) * torch.log(2.0 / alpha_c - 1.0)
        
        # Compute phi_x
        tanh_val = torch.tanh(k * (x_clip - m_i))
        phi_x = s * tanh_val
        
        # Hard quantization in forward pass
        x_q = torch.sign(phi_x)
        x_q = torch.where(x_q == 0, torch.ones_like(x_q), x_q)
        
        # Dequantize
        x_hat = l + Delta * (i.float() + (x_q + 1.0) / 2.0)
        
        ctx.save_for_backward(x, alpha_c, l, u, x_clip, phi_x, tanh_val, i.float())
        return x_hat

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha_c, l, u, x_clip, phi_x, tanh_val, i = ctx.saved_tensors
        num_bits = ctx.num_bits
        
        Delta = (u - l) / (2**num_bits - 1)
        Delta = torch.clamp(Delta, min=1e-5)
        
        s = 1.0 / (1.0 - alpha_c)
        k = (1.0 / Delta) * torch.log(2.0 / alpha_c - 1.0)
        
        m_i = l + (i + 0.5) * Delta
        
        # Indicators for clipping
        mask_low = (x < l).float()
        mask_high = (x > u).float()
        mask_mid = (1.0 - mask_low - mask_high)
        
        # 1. Gradient w.r.t input x
        d_Qs_d_x = (Delta / 2.0) * s * k * (1.0 - tanh_val ** 2)
        grad_input = grad_output * d_Qs_d_x * mask_mid
        
        # Helper function to reduce gradients across dimensions (supporting rowwise/scalar/per-token boundaries)
        def reduce_grad(grad_val, target_param):
            if target_param.numel() == 1:
                return grad_val.sum().view_as(target_param)
            else:
                # Reduce over dimensions beyond target_param's last dim
                # e.g. target (out,1) ndim=2, grad (out,in) ndim=2 → reduce dim=[1]
                # e.g. target (b,s,1) ndim=3, grad (b,s,h) ndim=3 → reduce dim=[2]
                dims = list(range(target_param.ndim - 1, grad_val.ndim))
                return grad_val.sum(dim=dims, keepdim=True).view_as(target_param)
        
        # 2. Gradient w.r.t alpha
        ds_da = 1.0 / ((1.0 - alpha_c) ** 2)
        dk_da = -2.0 / (Delta * alpha_c * (2.0 - alpha_c))
        dphi_da = ds_da * tanh_val + s * (1.0 - tanh_val ** 2) * (x_clip - m_i) * dk_da
        d_Qs_d_alpha = (Delta / 2.0) * dphi_da
        grad_alpha = reduce_grad(grad_output * d_Qs_d_alpha * mask_mid, alpha_c)
        
        # 3. Gradient w.r.t l
        q = i + (phi_x + 1.0) / 2.0
        d_Delta_d_l = -1.0 / (2**num_bits - 1)
        dphi_d_l = -s * k * (1.0 - tanh_val ** 2) * (1.0 - (i + 0.5) / (2**num_bits - 1))
        d_Qs_d_l = 1.0 + q * d_Delta_d_l + (Delta / 2.0) * dphi_d_l
        grad_l = reduce_grad(grad_output * d_Qs_d_l * mask_mid, l) + reduce_grad(grad_output * mask_low, l)
        
        # 4. Gradient w.r.t u
        d_Delta_d_u = 1.0 / (2**num_bits - 1)
        dphi_d_u = -s * k * (1.0 - tanh_val ** 2) * ((i + 0.5) / (2**num_bits - 1))
        d_Qs_d_u = q * d_Delta_d_u + (Delta / 2.0) * dphi_d_u
        grad_u = reduce_grad(grad_output * d_Qs_d_u * mask_mid, u) + reduce_grad(grad_output * mask_high, u)
        
        return grad_input, grad_alpha, grad_l, grad_u, None


class DaqQuant(torch.autograd.Function):
    """
    Distance-aware Quantization (DAQ) Custom Autograd Operator.
    Forward pass is mathematically equivalent to discrete rounding,
    while backward pass computes a smooth, distance-aware surrogate gradient.
    """
    @staticmethod
    def forward(ctx, input, num_bits, l, u, gamma=2.0, sigma_k=1.0):
        # Save original l and u
        ctx.save_for_backward(input, l, u)
        
        # Promote to float32 to prevent bfloat16 precision loss on small offsets
        input_f32 = input.float()
        l_f32 = l.float()
        u_f32 = u.float()
        
        # Ensure numerical safety symmetrically
        violation = (u_f32 - l_f32 < 1e-2)
        mid = (l_f32 + u_f32) * 0.5
        l_f32 = torch.where(violation, mid - 5e-3, l_f32)
        u_f32 = torch.where(violation, mid + 5e-3, u_f32)
        
        # Clip input
        input_clipped = torch.clamp(input_f32, l_f32, u_f32)
        
        # Scale & shift to [0, 2^b - 1]
        scale_factor = (2.0 ** num_bits - 1) / (u_f32 - l_f32).clamp(min=1e-5)
        x = (input_clipped - l_f32) * scale_factor
        
        # Quantize (forward pass is exactly round(x))
        q = torch.round(x)
        
        # Dequantize
        x_hat = l_f32 + q / scale_factor
        
        ctx.num_bits = num_bits
        ctx.gamma = gamma
        ctx.sigma_k = sigma_k
        return x_hat.to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, l_orig, u_orig = ctx.saved_tensors
        num_bits = ctx.num_bits
        gamma = ctx.gamma
        sigma_k = ctx.sigma_k
        
        # Promote to float32
        input_f32 = input.float()
        l_orig_f32 = l_orig.float()
        u_orig_f32 = u_orig.float()
        grad_output_f32 = grad_output.float()
        
        # Recompute constrained l and u symmetrically
        violation = (u_orig_f32 - l_orig_f32 < 1e-2)
        mid = (l_orig_f32 + u_orig_f32) * 0.5
        l_f32 = torch.where(violation, mid - 5e-3, l_orig_f32)
        u_f32 = torch.where(violation, mid + 5e-3, u_orig_f32)
        
        # Boundaries mask: 1 inside bounds, 0 outside
        mask_mid = ((input_f32 >= l_f32) & (input_f32 <= u_f32)).float()
        
        # Normalize input to [0, 2^b - 1]
        scale_factor = (2.0 ** num_bits - 1) / (u_f32 - l_f32).clamp(min=1e-5)
        x = (input_f32 - l_f32) * scale_factor
        
        # floor, ceil, round
        q_f = torch.floor(x)
        q_c = q_f + 1.0
        q_n = torch.round(x)
        
        # Distance scores
        dx_f = torch.exp(-torch.abs(x - q_f))
        dx_c = torch.exp(-torch.abs(x - q_c))
        
        # Gaussian kernel values kx = exp(-(q - q_n)^2 / (2 * sigma_k^2))
        C = math.exp(-1.0 / (2.0 * (sigma_k ** 2)))
        
        kx_f = torch.where(q_n == q_f, torch.ones_like(x), torch.ones_like(x) * C)
        kx_c = torch.where(q_n == q_c, torch.ones_like(x), torch.ones_like(x) * C)
        
        # Weighted scores
        s_f = kx_f * dx_f
        s_c = kx_c * dx_c
        
        # 1. Compute adaptive temperature parameter beta* (Eq. 7 in paper)
        beta = gamma / torch.abs(s_c - s_f).clamp(min=1e-5)
        
        # 2. Compute Softmax probabilities m_f and m_c (Eq. 6 in paper)
        # Use numerical stability subtraction (minus max) to prevent exp overflow
        s_max = torch.maximum(s_f, s_c)
        exp_f = torch.exp(beta * (s_f - s_max))
        exp_c = torch.exp(beta * (s_c - s_max))
        sum_exp = exp_f + exp_c
        m_f = exp_f / sum_exp
        m_c = exp_c / sum_exp
        
        # 3. Compute surrogate gradient multiplier g_mult (derivative of soft assignment w.r.t x)
        g_mult = beta * m_f * m_c * (s_c + s_f)
        
        # grad_input
        grad_input = grad_output_f32 * g_mult * mask_mid
        
        # For l and u
        p_q = q_n / (2.0 ** num_bits - 1)
        grad_l_element = torch.where(input_f32 < l_f32, torch.ones_like(input_f32), 
                                     torch.where(input_f32 > u_f32, torch.zeros_like(input_f32), 1.0 - p_q))
        grad_u_element = torch.where(input_f32 > u_f32, torch.ones_like(input_f32),
                                     torch.where(input_f32 < l_f32, torch.zeros_like(input_f32), p_q))
        
        def reduce_grad(grad_val, target_param):
            if target_param.numel() == 1:
                return grad_val.sum().view_as(target_param)
            else:
                dims = list(range(target_param.ndim - 1, grad_val.ndim))
                return grad_val.sum(dim=dims, keepdim=True).view_as(target_param)
        
        grad_l_val = reduce_grad(grad_output_f32 * grad_l_element, l_f32)
        grad_u_val = reduce_grad(grad_output_f32 * grad_u_element, u_f32)
        
        # Apply symmetric chain rule for u - l < 1e-2 constraint
        if violation.any():
            grad_avg = 0.5 * (grad_l_val + grad_u_val)
            grad_l = torch.where(violation, grad_avg, grad_l_val)
            grad_u = torch.where(violation, grad_avg, grad_u_val)
        else:
            grad_l = grad_l_val
            grad_u = grad_u_val
        
        # Scale boundary gradients to match weight gradient magnitudes
        in_features = input_f32.shape[-1]
        grad_scale = 1.0 / math.sqrt(in_features * (2.0 ** num_bits - 1))
        grad_l = grad_l * grad_scale
        grad_u = grad_u * grad_scale
            
        return grad_input.to(grad_output.dtype), None, grad_l.to(l_orig.dtype), grad_u.to(u_orig.dtype), None, None


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
    use_dsq_activation=False,
    act_dsq_alpha=None,
    use_daq_activation=False,
    daq_gamma=2.0,
    daq_sigma_k_act=2.0,
    dtype=torch.float32
):
    """
    高阶激活量化管理函数
    负责在首个前向传播时进行 LSQ 或 DSQ 自适应初始化，并路由调用相应量化算子
    """
    if a_bits >= 16:
        return input_

    if use_daq_activation:
        # Per-token dynamic range (matching DynamicActivationQuant granularity)
        with torch.no_grad():
            l = input_.min(dim=-1, keepdim=True)[0]
            u = input_.max(dim=-1, keepdim=True)[0]
            margin = (u - l).clamp(min=1e-5) * 0.01
            l = l - margin
            u = u + margin
        return DaqQuant.apply(input_, a_bits, l, u, daq_gamma, daq_sigma_k_act).to(dtype)

    if use_dsq_activation:
        # Per-token dynamic range (matching DynamicActivationQuant granularity)
        with torch.no_grad():
            l = input_.min(dim=-1, keepdim=True)[0]
            u = input_.max(dim=-1, keepdim=True)[0]
            margin = (u - l).clamp(min=1e-5) * 0.01
            l = l - margin
            u = u + margin
        return DsqQuant.apply(input_, act_dsq_alpha, l, u, a_bits).to(dtype)

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
    use_dsq_weight=False,
    weight_dsq_alpha=None,
    weight_clip_l=None,
    weight_clip_u=None,
    use_daq_weight=False,
    daq_gamma=2.0,
    daq_sigma_k_weight=1.0,
    dtype=torch.float32
):
    """
    高阶权重量化管理函数
    负责在非 LSQ 下的动态 Max 尺度计算，并路由调用 StretchedElastic 或 Lsq 算子
    """
    if w_bits >= 16:
        return weight

    if use_daq_weight:
        return DaqQuant.apply(weight, w_bits, weight_clip_l, weight_clip_u, daq_gamma, daq_sigma_k_weight).to(dtype)

    if use_dsq_weight:
        return DsqQuant.apply(weight, weight_dsq_alpha, weight_clip_l, weight_clip_u, w_bits).to(dtype)

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
        use_asymmetric_act: bool = False,
        use_dsq_weight: bool = False,
        use_dsq_activation: bool = False,
        dsq_init_alpha: float = 0.2,
        use_daq_weight: bool = False,
        use_daq_activation: bool = False,
        daq_gamma: float = 2.0,
        daq_sigma_k_weight: float = 1.0,
        daq_sigma_k_act: float = 2.0
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=bias)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.weight_layerwise = weight_layerwise
        self.use_stableqat = use_stableqat
        self.use_lsq_weight = use_lsq_weight
        self.use_lsq_activation = use_lsq_activation
        self.use_asymmetric_act = use_asymmetric_act
        self.use_dsq_weight = use_dsq_weight
        self.use_dsq_activation = use_dsq_activation
        self.dsq_init_alpha = dsq_init_alpha
        self.use_daq_weight = use_daq_weight
        self.use_daq_activation = use_daq_activation
        self.daq_gamma = daq_gamma
        self.daq_sigma_k_weight = daq_sigma_k_weight
        self.daq_sigma_k_act = daq_sigma_k_act

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
            if self.use_dsq_weight or self.use_daq_weight:
                self.weight_clip_l = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
                self.weight_clip_u = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
                if self.use_dsq_weight:
                    self.weight_dsq_alpha = nn.Parameter(torch.tensor([dsq_init_alpha]))
            elif self.use_lsq_weight:
                self.weight_clip_val = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
            else:
                self.register_buffer('weight_clip_val', torch.Tensor(self.weight.shape[0], 1))
                
        # params for activation quant
        if self.a_bits < 16:
            if self.use_dsq_activation or self.use_daq_activation:
                # act_dsq_alpha is only for DSQ, for DAQ we don't need learnable alpha (gamma and sigma_k are hyperparameters)
                if self.use_dsq_activation:
                    self.act_dsq_alpha = nn.Parameter(torch.tensor([dsq_init_alpha]))
            elif self.use_lsq_activation:
                self.act_clip_val = nn.Parameter(torch.ones(1))
            else:
                self.register_buffer('act_clip_val', torch.ones(1))

    def forward(self, input_):
        # 1. 量化激活值
        if self.a_bits < 16:
            input_q = quantize_activation(
                input_, 
                self.act_clip_val if (not self.use_dsq_activation and not self.use_daq_activation) else None, 
                self.a_bits, 
                use_lsq_activation=self.use_lsq_activation, 
                use_asymmetric_act=self.use_asymmetric_act,
                use_dsq_activation=self.use_dsq_activation,
                act_dsq_alpha=self.act_dsq_alpha if self.use_dsq_activation else None,
                use_daq_activation=self.use_daq_activation,
                daq_gamma=self.daq_gamma,
                daq_sigma_k_act=self.daq_sigma_k_act,
                dtype=input_.dtype
            )
        else:
            input_q = input_

        # 2. 量化权重
        if self.w_bits < 16:
            weight_q = quantize_weight(
                self.weight,
                self.weight_clip_val if (not self.use_dsq_weight and not self.use_daq_weight) else None,
                self.w_bits,
                weight_layerwise=self.weight_layerwise,
                use_lsq_weight=self.use_lsq_weight,
                use_stableqat=self.use_stableqat,
                sine_amplitude=self.sine_amplitude,
                use_dsq_weight=self.use_dsq_weight,
                weight_dsq_alpha=self.weight_dsq_alpha if self.use_dsq_weight else None,
                weight_clip_l=self.weight_clip_l if (self.use_dsq_weight or self.use_daq_weight) else None,
                weight_clip_u=self.weight_clip_u if (self.use_dsq_weight or self.use_daq_weight) else None,
                use_daq_weight=self.use_daq_weight,
                daq_gamma=self.daq_gamma,
                daq_sigma_k_weight=self.daq_sigma_k_weight,
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
        use_asymmetric_act: bool = False,
        use_dsq_weight: bool = False,
        use_dsq_activation: bool = False,
        dsq_init_alpha: float = 0.2,
        use_daq_weight: bool = False,
        use_daq_activation: bool = False,
        daq_gamma: float = 2.0,
        daq_sigma_k_weight: float = 1.0,
        daq_sigma_k_act: float = 2.0
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
            use_dsq_weight=use_dsq_weight,
            use_dsq_activation=use_dsq_activation,
            dsq_init_alpha=dsq_init_alpha,
            use_daq_weight=use_daq_weight,
            use_daq_activation=use_daq_activation,
            daq_gamma=daq_gamma,
            daq_sigma_k_weight=daq_sigma_k_weight,
            daq_sigma_k_act=daq_sigma_k_act
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
    use_dsq_weight: bool = False,
    use_dsq_activation: bool = False,
    dsq_init_alpha: float = 0.2,
    use_daq_weight: bool = False,
    use_daq_activation: bool = False,
    daq_gamma: float = 2.0,
    daq_sigma_k_weight: float = 1.0,
    daq_sigma_k_act: float = 2.0
):
    """
    递归遍历模型，将 nn.Linear 替换为 QuantizeLinear。
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
            use_asymmetric_act=use_asymmetric_act,
            use_dsq_weight=use_dsq_weight,
            use_dsq_activation=use_dsq_activation,
            dsq_init_alpha=dsq_init_alpha,
            use_daq_weight=use_daq_weight,
            use_daq_activation=use_daq_activation,
            daq_gamma=daq_gamma,
            daq_sigma_k_weight=daq_sigma_k_weight,
            daq_sigma_k_act=daq_sigma_k_act
        )
        # 递归 setattr
        _set_module_by_name(model, name, quant_linear)
    
    print(f"[ParetoQ] Replaced {len(replace_list)} nn.Linear with QuantizeLinear (w_bits={w_bits}, a_bits={a_bits}, use_stableqat={use_stableqat}, use_dsq_weight={use_dsq_weight}, use_dsq_activation={use_dsq_activation}, use_daq_weight={use_daq_weight}, use_daq_activation={use_daq_activation})")
    return model

'''