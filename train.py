# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import distributed as dist
import transformers
from transformers import AutoModelForCausalLM, default_data_collator, Trainer

from utils import utils, datautils
from utils.process_args import process_args
from utils.eval import run_evaluation


log = utils.get_logger("clm")


class MuonTrainer(Trainer):
    def get_decay_parameter_names(self, model):
        decay_parameters = super().get_decay_parameter_names(model)
        return [
            name
            for name in decay_parameters
            if ".butterfly.theta" not in name
        ]

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        if not self.args.use_muon:
            return super().create_optimizer()

        from optimizer.muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

        # Muon only applies to hidden 2D matrices (>= 2D, not embed, not lm_head, not weight_clip_val)
        muon_params = [
            p
            for n, p in self.model.named_parameters()
            if (
                p.requires_grad
                and p.ndim >= 2
                and "embed" not in n
                and "lm_head" not in n
                and ".butterfly.theta" not in n
            )
            # and "clip_val" not in n
            # and "clip_l" not in n and "clip_u" not in n and "dsq_alpha" not in n
        ]
        adamw_params = [
            p
            for n, p in self.model.named_parameters()
            if p.requires_grad
            and p.ndim < 2
            and ".butterfly.theta" not in n
            # or "clip_val" in n
            # or "clip_l" in n or "clip_u" in n or "dsq_alpha" in n
        ]
        butterfly_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and ".butterfly.theta" in n
        ]

        # Resolve learning rates
        muon_lr = self.args.learning_rate
        adamw_lr = self.args.adamw_learning_rate
        if (
            len(adamw_params) > 0 or len(butterfly_params) > 0
        ) and adamw_lr is None:
            raise ValueError(
                "adamw_learning_rate must be explicitly specified when "
                "there are parameters to be optimized by AdamW."
            )

        param_groups = [
            {
                "params": muon_params,
                "lr": muon_lr,
                "momentum": 0.95,
                "weight_decay": self.args.weight_decay,
                "use_muon": True,
            }
        ]

        if len(adamw_params) > 0:
            param_groups.append({
                "params": adamw_params,
                "lr": adamw_lr,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": self.args.weight_decay,
                "use_muon": False,
            })

        if len(butterfly_params) > 0:
            param_groups.append({
                "params": butterfly_params,
                "lr": adamw_lr,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "use_muon": False,
            })

        if dist.is_initialized():
            self.optimizer = MuonWithAuxAdam(param_groups)
        else:
            self.optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        return self.optimizer

    '''
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        else:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=False, **kwargs), None

        # 前置显式检查：如果未启用权重或激活值的 DSQ，直接返回原版 Loss，避免多余计算
        use_dsq_w = getattr(self.args, "use_dsq_weight", False)
        use_dsq_a = getattr(self.args, "use_dsq_activation", False)
        if not (use_dsq_w or use_dsq_a):
            return (loss, outputs) if return_outputs else loss

        dsq_reg = 0.0
        dsq_alpha_lambda = getattr(self.args, "dsq_alpha_lambda", 1e-4)
        has_dsq = False
        for name, param in model.named_parameters():
            if "dsq_alpha" in name:
                dsq_reg += torch.sum(param ** 2)
                has_dsq = True
        
        if has_dsq:
            loss = loss + dsq_alpha_lambda * dsq_reg
            
        return (loss, outputs) if return_outputs else loss
    '''

def train():
    dist.init_process_group(backend="nccl")
    model_args, data_args, training_args, eval_args = process_args()
    
    transformers.set_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)

    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    skip_keywords = []

    if not training_args.train_lm_head_embed:
        skip_keywords.extend(['lm_head', 'embed'])
        log.info("lm_head and embed will be skipped.")

    log.info("Start to load model...")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        cache_dir=training_args.cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map='cpu',
        attn_implementation="eager"
    )
    log.info("Complete model loading...")

    log.info("Start to load tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        add_bos_token=False,
        add_eos_token=False,
    )
    log.info("Complete tokenizer loading...")

    train_data, valid_data = None, None

    if training_args.do_train or training_args.do_eval:
        import os
        # Create a sanitised model name to avoid vocabulary collision across different models
        model_safe_name = model_args.input_model_filename.replace("/", "_").replace("\\", "_")
        # Get dataset name and append max_train_tokens to avoid cache collision
        dataset_safe_name = os.path.basename(data_args.train_data_local_path).replace(".jsonl", "")
        if data_args.max_train_tokens is not None:
            dataset_folder = f"{dataset_safe_name}_tokens_{data_args.max_train_tokens}"
        else:
            dataset_folder = f"{dataset_safe_name}_all"
        cache_dir = os.path.join(model_args.local_dir, "dataset_cache", model_safe_name, dataset_folder)

        train_bin, valid_bin = datautils.preprocess_jsonl_to_bin_split(
            train_path=data_args.train_data_local_path,
            valid_path=data_args.eval_data_local_path
            if data_args.eval_data_local_path is not None
            else None,
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            max_train_tokens=data_args.max_train_tokens
        )

        train_data = datautils.CustomBinDataset(
            train_bin, block_size=training_args.model_max_length
        )
        valid_data = datautils.CustomBinDataset(
            valid_bin, block_size=training_args.model_max_length
        )

    if training_args.qat and (model_args.w_bits < 16 or model_args.a_bits < 16):
        enabled_quantization_methods = [
            method_name
            for method_name, enabled in (
                ("my", training_args.use_my),
                ("butterfly", training_args.use_butterfly),
                ("hestia", training_args.use_hestia),
                ("quest", training_args.use_quest),
            )
            if enabled
        ]
        if len(enabled_quantization_methods) != 1:
            raise ValueError(
                "Exactly one QAT quantization method must be enabled; got "
                f"{enabled_quantization_methods or 'none'}."
            )

        if training_args.use_my:
            from utils.utils_myquant import replace_linear_with_myquantize, load_trust_scale_dict
            trust_scale_dict = load_trust_scale_dict('./trace_estimation/meta-llama_Llama-3.2-1B_Hestia_src/hessian_traces.pkl', 'temp_scales')
            model = replace_linear_with_myquantize(
                model,
                w_bits=model_args.w_bits,
                a_bits=model_args.a_bits,
                w_group_size=model_args.w_group_size,
                a_group_size=model_args.a_group_size,
                w_quant_type=model_args.w_quant_type,
                a_quant_type=model_args.a_quant_type,
                skip_keywords=['embed', 'lm_head'],
                trust_style='mask',
                trust_scale_dict=trust_scale_dict
            )

        elif training_args.use_butterfly:
            from utils.utils_butterfly import replace_linear_with_butterfly

            model = replace_linear_with_butterfly(
                model,
                w_bits=model_args.w_bits,
                a_bits=model_args.a_bits,
                w_group_size=model_args.w_group_size,
                a_group_size=model_args.a_group_size,
                butterfly_init=training_args.butterfly_init,
                hadamard_block_size=(
                    training_args.butterfly_hadamard_block_size
                ),
                skip_keywords=skip_keywords,
            )

        elif training_args.use_hestia:
            # --- Hestia path ---
            import utils.utils_hestia as hestia_mod

            temp_scales = None
            if training_args.hestia_enable_calib and training_args.hessian_traces_path:
                temp_scales = hestia_mod.load_temp_scales(training_args.hessian_traces_path)

            model = hestia_mod.replace_linear_with_hestia(
                model,
                bits=model_args.w_bits,
                group_size=model_args.w_group_size,
                compress_ratio=training_args.hestia_compress_ratio,
                init_temp=training_args.hestia_init_temp,
                end_temp=training_args.hestia_end_temp,
                anneal_ratio=training_args.hestia_anneal_ratio,
                temp_scales_dict=temp_scales,
                strict_temp_scale=training_args.hestia_enable_calib,
                skip_keywords=skip_keywords,
            )

        elif training_args.use_quest:
            from utils.utils_quest import replace_linear_with_quest
            model = replace_linear_with_quest(
                model,
                w_bits=model_args.w_bits,
                a_bits=model_args.a_bits,
                hadamard_block_size=training_args.quest_hadamard_block_size,
                trust_scale_weight=training_args.quest_trust_scale_weight,
                trust_scale_act=training_args.quest_trust_scale_act,
                skip_keywords=skip_keywords,
            )
        else:
            raise ValueError('No matched quantization method.')
            '''
            from utils.utils_quant import replace_linear_with_quantized
            model = replace_linear_with_quantized(
                model,
                w_bits=model_args.w_bits,
                a_bits=model_args.a_bits,
                weight_asymmetric=model_args.weight_asymmetric,
                act_asymmetric=model_args.act_asymmetric,
                skip_keywords=skip_keywords,
                use_stableqat=training_args.use_stableqat,
                use_robusttraining=training_args.use_robusttraining
            )
            '''

    # --- WinQ wrapper (applied after quantizer selection, before .cuda()) ---
    winq_layers = []
    if training_args.use_winq:
        from utils.utils_winq import apply_winq_to_model
        winq_layers = apply_winq_to_model(
            model,
            sigma=training_args.winq_sigma,
            alpha=training_args.winq_alpha,
        )
        log.info(f"[WinQ] {len(winq_layers)} layers wrapped")

        '''
        if not model_args.contain_weight_clip_val:
            init_clip_val(model, model_args)
        else:
            load_clip_val(model, model_args)
        '''

    model.cuda()
    log.info("Complete model loading...")

    # Freeze parameters based on arguments
    if not training_args.train_lm_head_embed:
        log.info("Freezing lm_head and embed_tokens...")
        for name, param in model.named_parameters():
            if "lm_head" in name or "embed_tokens" in name:
                param.requires_grad = False

    if not training_args.train_rmsnorm:
        log.info("Freezing RMSNorm layers...")
        for name, param in model.named_parameters():
            if "layernorm" in name or "norm.weight" in name:
                param.requires_grad = False



    model.config.use_cache = False
    model.enable_input_require_grads()

    myTrainer = MuonTrainer
    trainer = myTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=valid_data if training_args.do_eval else None,
        data_collator=default_data_collator,
    )

    # Register Hestia step callback if using Hestia
    if training_args.use_hestia:
        from utils.utils_hestia import HestiaStepCallback
        trainer.add_callback(HestiaStepCallback())

    # Register WinQ callback if using WinQ
    if training_args.use_winq and winq_layers:
        from utils.utils_winq import WinQCallback
        trainer.add_callback(WinQCallback(
            winq_layers,
            interval=training_args.winq_reset_interval,
        ))
        log.info(f"[WinQ] Callback registered (reset every {training_args.winq_reset_interval} steps)")

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_state()
        utils.safe_save_model_for_hf_trainer(trainer, model_args.output_model_local_path)

    if training_args.do_eval:
        raise NotImplementedError("do_eval is not supported. Please use --eval_ppl or --eval_lm_eval for evaluation.")

    # Evaluation
    if eval_args.eval_ppl or eval_args.eval_lm_eval:
        tasks = [t.strip() for t in eval_args.tasks.split(",")]
        model.to("cuda")
        run_evaluation(
            model,
            tokenizer,
            tasks=tasks,
            eval_ppl=eval_args.eval_ppl,
            eval_lm_eval=eval_args.eval_lm_eval,
            batch_size=eval_args.eval_batch_size,
        )

    torch.distributed.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train()

'''
def init_clip_val(model, model_args):
    for name, module in model.named_modules():
        if not isinstance(module, QuantizeLinear) or model_args.w_bits >= 16:
            continue
        weight_param = module.weight
        if model_args.w_bits == 1:
            scale = torch.mean(weight_param.abs(), dim=-1, keepdim=True).detach()
        elif model_args.w_bits == 0 or model_args.w_bits == 2:
            scale, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
        elif model_args.w_bits == 3 or model_args.w_bits == 4:
            xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
            maxq = 2 ** (model_args.w_bits - 1) - 1
            scale = xmax / maxq
        else:
            raise NotImplementedError

        if getattr(module, 'use_dsq_weight', False) or getattr(module, 'use_daq_weight', False):
            module.weight_clip_l.data.copy_(-xmax)
            module.weight_clip_u.data.copy_(xmax)
        else:
            # weight_clip_val will passed as alpha(quantization step size), so it should be initialized to scale
            module.weight_clip_val.data.copy_(scale)

def load_clip_val(model, model_args):

    log.info("Loading saved quantized parameters from checkpoint...")
    import os
    import json
    state_dict = {}
    checkpoint_dir = model_args.input_model_filename
    
    index_path_safe = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    index_path_bin = os.path.join(checkpoint_dir, "pytorch_model.bin.index.json")

    dsq_kws = ["weight_clip_val", "act_clip_val", "weight_clip_l", "weight_clip_u", "weight_dsq_alpha", "act_dsq_alpha"]
    def is_quant_param(k):
        return any(kw in k for kw in dsq_kws)

    loaded = False
    if os.path.exists(index_path_safe):
        from safetensors.torch import load_file as safe_load_file
        with open(index_path_safe, "r") as f:
            index_data = json.load(f)
        shard_files = {v for k, v in index_data["weight_map"].items() if is_quant_param(k)}
        for shard_file in shard_files:
            shard_path = os.path.join(checkpoint_dir, shard_file)
            shard_sd = safe_load_file(shard_path)
            for k, v in shard_sd.items():
                if is_quant_param(k):
                    state_dict[k] = v
        loaded = True
    elif os.path.exists(index_path_bin):
        with open(index_path_bin, "r") as f:
            index_data = json.load(f)
        shard_files = {v for k, v in index_data["weight_map"].items() if is_quant_param(k)}
        for shard_file in shard_files:
            shard_path = os.path.join(checkpoint_dir, shard_file)
            shard_sd = torch.load(shard_path, map_location="cpu")
            for k, v in shard_sd.items():
                if is_quant_param(k):
                    state_dict[k] = v
        loaded = True
    else:
        # Check for single file
        single_safe = os.path.join(checkpoint_dir, "model.safetensors")
        single_bin = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(single_safe):
            from safetensors.torch import load_file as safe_load_file
            full_sd = safe_load_file(single_safe)
            state_dict = {k: v for k, v in full_sd.items() if is_quant_param(k)}
            loaded = True
        elif os.path.exists(single_bin):
            full_sd = torch.load(single_bin, map_location="cpu")
            state_dict = {k: v for k, v in full_sd.items() if is_quant_param(k)}
            loaded = True
    
    if loaded and len(state_dict) > 0:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded {len(state_dict)} quantization parameters. Missing keys size: {len(missing_keys)}, Unexpected keys size: {len(unexpected_keys)}")
    elif not loaded:
        log.warning("Could not find index.json or checkpoint files in input_model_filename to load quantization parameters!")
    else:
        log.warning("No quantization parameters found in the checkpoint files!")
'''
