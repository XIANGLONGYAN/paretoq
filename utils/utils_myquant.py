import math
import torch
from torch import nn
import torch.nn.functional as F


OPTIMAL_GAUSSIAN_SCALES = {
    0: 1.2240089519030855,
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



H_128 = None

def get_H_128(device, dtype):
    global H_128
    if H_128 is None:
        H = torch.tensor([[1.0]], dtype=torch.float32)
        for _ in range(7):
            H = torch.cat([torch.cat([H, H], dim=-1),
                          torch.cat([H, -H], dim=-1)],
                          dim = 0)
        H_128 = H / math.sqrt(128)
    return H_128.to(device=device, dtype=dtype)
            

class BaseQuantizer(nn.Module):
    def __init__(
        self, 
        num_bits, 
        *args,
        group_size=-1,
        **kwargs
    ):
        super().__init__()
        self.num_bits = num_bits
        self.group_size = group_size

    def reshape_by_group_size(self, x):
        origin_shape = x.size()
        if self.group_size == -1: # Channel-wise
            return x, origin_shape
        elif self.group_size == 0: # Per-tensor
            return x.reshape(1, -1), origin_shape
        elif x.size(-1) % self.group_size != 0:
            raise ValueError(f'group_size: {self.group_size}, x.size(-1): {x.size(-1)}. group_size cannot divide x.size(-1)')
        else:
            return x.reshape(-1, self.group_size), origin_shape

    def forward(self, x, **kwargs):
        return x

class AsymQuantizer(BaseQuantizer):
    def forward(self, x, **kwargs):
        if self.num_bits >= 16:
            return x

        x, origin_shape = self.reshape_by_group_size(x)
        Qn, Qp = 0, 2**self.num_bits - 1
        min_val, max_val = x.min(dim=-1, keepdim=True)[0], x.max(dim=-1, keepdim=True)[0]
        scale = (max_val - min_val) / (Qp - Qn)
        scale = scale.clamp(min=1e-5)
        zp = (-min_val / scale).round()
        q_x = (x / scale + zp).round().clamp(min=Qn, max=Qp)
        x_q = scale * (q_x - zp)
        return (x + (x_q - x).detach()).reshape(origin_shape)

class SymMaxQuantizer(BaseQuantizer):
    def forward(self, x, **kwargs):
        if self.num_bits >= 16:
            return x
        x, origin_shape = self.reshape_by_group_size(x)
        abs_max = x.abs().max(dim=-1, keepdim=True)[0]
        scale = abs_max / (2**(self.num_bits - 1) - 1)
        scale = scale.clamp(min=1e-5)
        x_clipped = x.clamp(-abs_max, abs_max)
        x_q = scale * (x_clipped / scale).round()
        return (x + (x_q - x).detach()).reshape(origin_shape)


class AlignedSymMaxQuantizer(BaseQuantizer):
    def forward(self, x, **kwargs):
        if self.num_bits >= 16:
            return x
        x, origin_shape = self.reshape_by_group_size(x)
        abs_max = x.abs().max(dim=-1, keepdim=True)[0]
        scale = 2 * abs_max / (2**self.num_bits - 1)
        scale = scale.clamp(min=1e-5)
        x_clipped = x.clamp(-abs_max, abs_max)
        x_q = scale * (x_clipped / scale + 0.5).round() - scale / 2
        return (x + (x_q - x).detach()).reshape(origin_shape)


class SymMeanQuantizer(BaseQuantizer):
    def forward(self, x, **kwargs):
        if self.num_bits >= 16:
            return x
        x, origin_shape = self.reshape_by_group_size(x)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        scale = abs_mean / (2**(self.num_bits - 1) - 1)
        scale = scale.clamp(min=1e-5)
        x_clipped = x.clamp(-abs_mean, abs_mean)
        x_q = scale * (x_clipped / scale).round()
        return (x + (x_q - x).detach()).reshape(origin_shape)


class AlignedSymMeanQuantizer(BaseQuantizer):
    def forward(self, x, **kwargs):
        if self.num_bits >= 16:
            return x
        x, origin_shape = self.reshape_by_group_size(x)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        scale = 2 * abs_mean / (2**self.num_bits - 1)
        scale = scale.clamp(min=1e-5)
        x_clipped = x.clamp(-abs_mean, abs_mean)
        x_q = scale * (x_clipped / scale + 0.5).round() - scale / 2
        return (x + (x_q - x).detach()).reshape(origin_shape)
        

class HadamardGaussianQuantizer(BaseQuantizer):
    def hadamard_transform(self, x):
        H_128 = get_H_128(device=x.device, dtype=x.dtype)
        if x.size(-1) % 128 != 0:
            raise ValueError(f'x.size(-1) is {x.size(-1)} which cannot be divided by 128.')
        origin_shape = x.size()
        x_reshaped = x.reshape(-1, x.size(-1) // 128, 128)
        x_had = x_reshaped @ H_128.T
        return x_had.reshape(origin_shape)
    
    def inverse_hadamard_transform(self, x):
        H_128 = get_H_128(device=x.device, dtype=x.dtype)
        if x.size(-1) % 128 != 0:
            raise ValueError(f'x.size(-1) is {x.size(-1)} which cannot be divided by 128.')
        origin_shape = x.size()
        x_reshaped = x.reshape(-1, x.size(-1) // 128, 128)
        x_inv_had = x_reshaped @ H_128
        return x_inv_had.reshape(origin_shape)

    def forward(self, x, **kwargs):
        if self.num_bits >= 16:
            return x
        x, origin_shape = self.reshape_by_group_size(x)
        x_had = self.hadamard_transform(x)
        alpha_star = OPTIMAL_GAUSSIAN_SCALES[self.num_bits]
        rms = (x_had**2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-5)
        scale = alpha_star * rms

        step = scale / (2**(self.num_bits - 1) - 1)
        x_clipped = x_had.clamp(min=-scale, max=scale)
        x_q = step * (x_clipped / step).round()

        x_inv_had = self.inverse_hadamard_transform(x_had + (x_q - x_had).detach())

        return x_inv_had.reshape(origin_shape)

class HadamardGaussianTrustQuantizer(HadamardGaussianQuantizer):
    def __init__(self, *args, trust_style='mask', **kwargs):
        super().__init__(*args, **kwargs)
        self.trust_style = trust_style

    def forward(self, x, trust_style='mask', **kwargs):
        if self.num_bits >= 16:
            return x
        x, origin_shape = self.reshape_by_group_size(x)
        x_had = self.hadamard_transform(x)
        alpha_star = OPTIMAL_GAUSSIAN_SCALES[self.num_bits]
        rms = (x_had**2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-5)
        scale = alpha_star * rms

        step = scale / (2**(self.num_bits - 1) - 1)
        x_clipped = x_had.clamp(min=-scale, max=scale)
        x_q = step * (x_clipped / step).round()


        dist = (x_q - x_had).abs()
        if self.trust_style == 'mask':
            trust_threshold_scale = kwargs['trust_threshold_scale']
            trust_threshold = trust_threshold_scale * (step / 2)
            trust_mask = (dist <= trust_threshold).to(dtype=x.dtype)
        elif self.trust_style == 'linear':
            trust_mask = -(dist - step) / step
        elif self.trust_style == 'cosine':
            trust_mask = (torch.cos(torch.pi * (dist / step)) + 1) / 2
        else:
            raise ValueError(f'trust_style: {trust_style} is not valid.')

        trust_mask = trust_mask.detach()

        x_masked = x_had * trust_mask

        x_gradflow = x_masked + (x_q - x_masked).detach()

        x_inv_had = self.inverse_hadamard_transform(x_gradflow)

        return x_inv_had.reshape(origin_shape)


class AlignedHadamardGaussianQuantizer(HadamardGaussianQuantizer):
    def forward(self, x, **kwargs):
        if self.num_bits >= 16:
            return x
        x, origin_shape = self.reshape_by_group_size(x)
        x_had = self.hadamard_transform(x)
        alpha_star = OPTIMAL_GAUSSIAN_SCALES[self.num_bits]
        rms = (x_had**2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-5)
        scale = alpha_star * rms

        step = 2 * scale / (2**self.num_bits - 1)
        x_clipped = x_had.clamp(min=-scale, max=scale)
        x_q = step * (x_clipped / step + 0.5).round() - step / 2

        x_inv_had = self.inverse_hadamard_transform(x_had + (x_q - x_had).detach())

        return x_inv_had.reshape(origin_shape)


class AlignedHadamardGaussianTrustQuantizer(HadamardGaussianQuantizer):
    def __init__(self, *args, trust_style='mask', **kwargs):
        super().__init__(*args, **kwargs)
        self.trust_style = trust_style

    def forward(self, x, **kwargs):
        if self.num_bits >= 16:
            return x
        x, origin_shape = self.reshape_by_group_size(x)
        x_had = self.hadamard_transform(x)
        alpha_star = OPTIMAL_GAUSSIAN_SCALES[self.num_bits]
        rms = (x_had**2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-5)
        scale = alpha_star * rms

        step = 2 * scale / (2**self.num_bits - 1)
        x_clipped = x_had.clamp(min=-scale, max=scale)
        x_q = step * (x_clipped / step + 0.5).round() - step / 2

        dist = (x_q - x_had).abs()
        if self.trust_style == 'mask':
            trust_threshold_scale = kwargs['trust_threshold_scale']
            trust_threshold = trust_threshold_scale * (step / 2)
            trust_mask = (dist <= trust_threshold).to(dtype=x.dtype)
        elif self.trust_style == 'linear':
            trust_mask = -(dist - step) / step
        elif self.trust_style == 'cosine':
            trust_mask = (torch.cos(torch.pi * (dist / step)) + 1) / 2
        else:
            raise ValueError(f'trust_style: {trust_style} is not valid.')

        trust_mask = trust_mask.detach()

        x_masked = x_had * trust_mask

        x_gradflow = x_masked + (x_q - x_masked).detach()

        x_inv_had = self.inverse_hadamard_transform(x_gradflow)

        return x_inv_had.reshape(origin_shape)



QUANTIZER_MAP = {
    'AsymQuantizer': AsymQuantizer,
    'SymMaxQuantizer': SymMaxQuantizer,
    'AlignedSymMaxQuantizer': AlignedSymMaxQuantizer,
    'SymMeanQuantizer': SymMeanQuantizer,
    'AlignedSymMeanQuantizer': AlignedSymMeanQuantizer,
    'HadamardGaussianQuantizer': HadamardGaussianQuantizer,
    'HadamardGaussianTrustQuantizer': HadamardGaussianTrustQuantizer,
    'AlignedHadamardGaussianQuantizer': AlignedHadamardGaussianQuantizer,
    'AlignedHadamardGaussianTrustQuantizer': AlignedHadamardGaussianTrustQuantizer
}


class MyQuantizeLinear(nn.Linear):
    def __init__(
        self,
        *args,
        w_bits=16,
        a_bits=16,
        w_group_size=-1,
        a_group_size=-1,
        w_quant_type='AlignedHadamardGaussianTrustQuantizer',
        a_quant_type='AlignedHadamardGaussianTrustQuantizer',
        layer_id=None,
        trust_style='mask',
        trust_threshold_scale=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        assert layer_id is not None
        # assert layer_id is not None and trust_scale is not None, f"layer_id: {layer_id}, trust_scale: {trust_scale} cannot be None"

        w_quantizer_cls = QUANTIZER_MAP[w_quant_type]
        a_quantizer_cls = QUANTIZER_MAP[a_quant_type]
        self.w_quantizer = w_quantizer_cls(w_bits, w_group_size, trust_style)
        self.a_quantizer = a_quantizer_cls(a_bits, a_group_size, trust_style)

        self.layer_id = layer_id
        self.trust_threshold_scale = trust_threshold_scale
        

    def forward(self, x):
        w = self.weight
        a = x

        w_q = self.w_quantizer(w, trust_threshold_scale=self.trust_threshold_scale)
        a_q = self.a_quantizer(a, trust_threshold_scale=self.trust_threshold_scale)

        return F.linear(a_q, w_q, self.bias)
      
    @classmethod
    def from_linear(
        cls,
        linear,
        w_bits=16,
        a_bits=16,
        w_group_size=-1,
        a_group_size=-1,
        w_quant_type='AlignedHadamardGaussianTrustQuantizer',
        a_quant_type='AlignedHadamardGaussianTrustQuantizer',
        layer_id=None,
        trust_threshold_scale=None,
        trust_style='mask'
    ):
        quantize_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            w_bits=w_bits,
            a_bits=a_bits,
            w_group_size=w_group_size,
            a_group_size=a_group_size,
            w_quant_type=w_quant_type,
            a_quant_type=a_quant_type,
            layer_id=layer_id,
            trust_threshold_scale=trust_threshold_scale,
            trust_style=trust_style
        )
        quantize_linear.weight = linear.weight
        quantize_linear.bias = None
        if linear.bias is not None:
            quantize_linear.bias = linear.bias
        
        return quantize_linear


def replace_linear_with_myquantize(
    model: nn.Module,
    w_bits=16,
    a_bits=16,
    w_group_size=-1,
    a_group_size=-1,
    w_quant_type='AlignedHadamardGaussianTrustQuantizer',
    a_quant_type='AlignedHadamardGaussianTrustQuantizer',
    skip_keywords=['embed', 'lm_head'],
    trust_style='mask',
    trust_scale_dict=None
):
    layer_counter = [0]
    replace_count = [0]

    def _convert(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            if name in skip_keywords:
                continue
            full_path = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear) and not isinstance(child, MyQuantizeLinear):
                layer_id = f"layer_{layer_counter[0]}_{full_path}"
                layer_counter[0] += 1

                trust_threshold_scale = trust_scale_dict.get(layer_id, None) if trust_scale_dict is not None else 1.0

                q_layer = MyQuantizeLinear.from_linear(
                    child,
                    w_bits=w_bits,
                    a_bits=a_bits,
                    w_group_size=w_group_size,
                    a_group_size=a_group_size,
                    w_quant_type=w_quant_type,
                    a_quant_type=a_quant_type,
                    layer_id=layer_id,
                    trust_threshold_scale=trust_threshold_scale,
                    trust_style=trust_style
                )
                setattr(module, name, q_layer)
                replace_count[0] += 1
            else:
                _convert(child, full_path)

    _convert(model)
    
    print(f'Replaced {replace_count[0]} nn.Linear with MyQuantizeLinear')

    return model


def load_trust_scale_dict(file_path, key_alias='trust_scales'):
    import pickle
    with open(file_path, 'rb') as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and key_alias in payload:
        return payload[key_alias]
    raise ValueError(f"Unsupported calibration payload type: {type(payload)}")