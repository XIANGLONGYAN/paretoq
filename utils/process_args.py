# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Optional, List

import transformers


@dataclass
class ModelArguments:
    local_dir: str = field(
        default=None, metadata={"help": "Local Path of storing inputs and outputs "}
    )
    input_model_filename: Optional[str] = field(
        default="test-input", metadata={"help": "Input model relative manifold path"}
    )
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )
    output_model_local_path: str = field(
        default=None, metadata={"help": "Output model local path, do not set manually"}
    )
    w_bits: Optional[int] = field(
        default=32,
        metadata={
            "help": "#bits to use for quantization; use 16 for evaluating base model. choices=[4, 8, 32]"
        },
    )
    a_bits: Optional[int] = field(
        default=16,
        metadata={
            "help": "#bits to use for activation quantization; use 16 for full precision."
        },
    )
    w_group_size: Optional[int] = field(
        default=0,
        metadata={"help": "Group size for weight quantization: 0=per-tensor, -1=per-channel."}
    )
    a_group_size: Optional[int] = field(
        default=0,
        metadata={"help": "Group size for activation quantization: 0=per-tensor, -1=per-channel."}
    )
    w_quant_type: Optional[str] = field(
        default="AlignedHadamardGaussianTrustQuantizer",
        metadata={"help": "Quantizer type for weight."}
    )
    a_quant_type: Optional[str] = field(
        default="AlignedHadamardGaussianTrustQuantizer",
        metadata={"help": "Quantizer type for activation."}
    )
    
'''
    weight_asymmetric: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use asymmetric weight quantization."}
    )
    act_asymmetric: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use asymmetric activation quantization."}
    )
    contain_weight_clip_val: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Set contain_weight_clip_val=True when load a trained quantized model."
        },
    )
'''
@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    train_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Train data local path"}
    )
    eval_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Eval data local path"}
    )
    max_train_tokens: Optional[int] = field(
        default=None, metadata={"help": "Max training tokens to preprocess/train on"}
    )

@dataclass
class EvalArguments:
    tasks: Optional[str] = field(
        default="piqa,arc_easy,arc_challenge,hellaswag,winogrande,mmlu,gsm8k_llama",
        metadata={"help": "Comma-separated list of lm_eval tasks (e.g., piqa,arc_easy,hellaswag)"},
    )
    eval_ppl: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run PPL evaluation on wikitext2 and c4"},
    )
    eval_lm_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run lm_eval zero-shot evaluation"},
    )
    eval_batch_size: Optional[int] = field(
        default=64,
        metadata={"help": "Batch size for lm_eval evaluation"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    qat: Optional[bool] = field(default=False)
    use_muon: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use Muon optimizer. When True, bypasses native optim validation."}
    )
    adamw_learning_rate: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for auxiliary AdamW optimizer."}
    )
    train_lm_head_embed: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train lm_head and embed_tokens parameters."}
    )
    train_rmsnorm: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train RMSNorm layers parameters."}
    )
    use_stableqat: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to enable StableQAT (Fourier surrogate gradient) during QAT."}
    )
    use_robusttraining: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use RobustTrainingQuantizeLinear for weight quantization during QAT."}
    )
    use_quest: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use QuEST (Hadamard + Trust Gradient) quantization during QAT."}
    )
    quest_hadamard_block_size: Optional[int] = field(
        default=128,
        metadata={"help": "Block size for QuEST Hadamard transform (must be power of 2)."}
    )
    quest_trust_scale_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Trust threshold multiplier for weight quantization in QuEST."}
    )
    quest_trust_scale_act: Optional[float] = field(
        default=1.0,
        metadata={"help": "Trust threshold multiplier for activation quantization in QuEST."}
    )
    use_hestia: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use Hestia (softmax-differentiable) QAT for extremely low-bit weights."}
    )
    hestia_compress_ratio: Optional[float] = field(
        default=0.2,
        metadata={"help": "Fraction of training steps allocated to Hestia compress phase."}
    )
    hestia_init_temp: Optional[float] = field(
        default=1.0,
        metadata={"help": "Initial temperature for Hestia softmax relaxation."}
    )
    hestia_end_temp: Optional[float] = field(
        default=0.0,
        metadata={"help": "Final temperature for Hestia (0 = hard quantization)."}
    )
    hestia_anneal_ratio: Optional[float] = field(
        default=0.8,
        metadata={"help": "Fraction of training steps allocated to Hestia anneal phase."}
    )
    hessian_traces_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-computed Hessian traces pickle from offline_calibration.py."}
    )
    hestia_enable_calib: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to apply per-layer temperature scaling from Hessian calibration."}
    )
    use_winq: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply WinQ acceleration wrapper (noise injection + periodic weight reset)."}
    )
    winq_sigma: Optional[float] = field(
        default=1e-3,
        metadata={"help": "Standard deviation of Gaussian noise for WinQ noise injection."}
    )
    winq_alpha: Optional[float] = field(
        default=0.2,
        metadata={"help": "Interpolation coefficient for WinQ periodic weight re-initialization."}
    )
    winq_reset_interval: Optional[int] = field(
        default=40000,
        metadata={"help": "Number of steps between WinQ weight resets."}
    )
    use_my: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use my quantization"}
    )
    log_weight_distribution: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Log HadamardGaussianTrustQuantizer weight-transition "
                "diagnostics during use_my QAT."
            )
        },
    )
    distribution_output_dir: Optional[str] = field(
        default="./distribution",
        metadata={"help": "Directory for weight-transition diagnostic text files."},
    )
    distribution_log_interval: Optional[int] = field(
        default=500,
        metadata={"help": "Optimizer-step interval between diagnostic snapshots."},
    )
    distribution_boundary_epsilon: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Half-width of the near-boundary band, measured in quantization steps."
        },
    )
    distribution_frequent_flip_threshold: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "Minimum flips within one logging window for an element "
                "to be considered frequently flipping."
            )
        },
    )
    distribution_sample_size: Optional[int] = field(
        default=65536,
        metadata={
            "help": "Maximum deterministic weight sample tracked per quantized layer."
        },
    )
'''
    use_lsq_weight: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use LSQ for weight quantization during QAT."}
    )
    use_lsq_activation: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use LSQ for activation quantization during QAT."}
    )
    use_dsq_weight: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use DSQ (Differentiable Soft Quantization) for weight quantization during QAT."}
    )
    use_dsq_activation: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use DSQ (Differentiable Soft Quantization) for activation quantization during QAT."}
    )
    dsq_init_alpha: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial value of similarity factor alpha for DSQ."}
    )
    dsq_alpha_lambda: Optional[float] = field(
        default=1e-4,
        metadata={"help": "Regularization coefficient lambda for DSQ alpha parameters."}
    )
    use_daq_weight: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use DAQ (Distance-aware Quantization) for weight quantization during QAT."}
    )
    use_daq_activation: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use DAQ (Distance-aware Quantization) for activation quantization during QAT."}
    )
    daq_gamma: Optional[float] = field(
        default=2.0,
        metadata={"help": "Gamma parameter for DAQ (dynamic temperature control strength)."}
    )
    daq_sigma_k_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Sigma_k parameter (Gaussian kernel std) for weight in DAQ."}
    )
    daq_sigma_k_act: Optional[float] = field(
        default=2.0,
        metadata={"help": "Sigma_k parameter (Gaussian kernel std) for activation in DAQ."}
    )
'''

def process_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, EvalArguments)
    )
    model_args, data_args, training_args, eval_args = parser.parse_args_into_dataclasses()

    os.makedirs(model_args.local_dir, exist_ok=True)

    assert model_args.output_model_local_path is None

    model_args.output_model_local_path = os.path.join(
        model_args.local_dir, "models", str(model_args.output_model_filename)
    )

    return model_args, data_args, training_args, eval_args
