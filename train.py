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
from utils.utils_quant import replace_linear_with_quantized, QuantizeLinear


log = utils.get_logger("clm")


class MuonTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        if not self.args.use_muon:
            return super().create_optimizer()

        from optimizer.muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

        # Muon only applies to hidden 2D matrices (>= 2D, not embed, not lm_head, not weight_clip_val)
        hidden_matrix_params = [
            p for n, p in self.model.named_parameters() 
            if p.ndim >= 2 and "embed" not in n and "lm_head" not in n # and "weight_clip_val" not in n
        ]
        other_params = [
            p for n, p in self.model.named_parameters() 
            if p.ndim < 2 or "embed" in n or "lm_head" in n # or "weight_clip_val" in n
        ]

        muon_lr = (
            self.args.muon_learning_rate 
            if self.args.muon_learning_rate is not None 
            else self.args.learning_rate * 10
        )

        param_groups = [
            {
                "params": hidden_matrix_params,
                "lr": muon_lr,
                "momentum": 0.95,
                "weight_decay": self.args.weight_decay,
                "use_muon": True,
            },
            {
                "params": other_params,
                "lr": self.args.learning_rate,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": self.args.weight_decay,
                "use_muon": False,
            }
        ]

        if dist.is_initialized():
            self.optimizer = MuonWithAuxAdam(param_groups)
        else:
            self.optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        return self.optimizer


def train():
    dist.init_process_group(backend="nccl")
    model_args, data_args, training_args, eval_args = process_args()

    log.info("Start to load model...")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        cache_dir=training_args.cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )

    if training_args.qat and (model_args.w_bits < 16 or model_args.a_bits < 16):
        model = replace_linear_with_quantized(
            model,
            w_bits=model_args.w_bits,
            a_bits=model_args.a_bits,
            weight_layerwise=False,
            skip_keywords=["lm_head", "embed"],
            use_stableqat=training_args.use_stableqat,
            use_lsq_weight=training_args.use_lsq_weight,
            use_lsq_activation=training_args.use_lsq_activation,
            use_asymmetric_act=training_args.use_asymmetric_act,
        )
        if not model_args.contain_weight_clip_val:
            for name, module in model.named_modules():
                if isinstance(module, QuantizeLinear) and model_args.w_bits < 16:
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

                    module.weight_clip_val.data.copy_(scale)
        else:
            log.info("Loading saved quantized parameters (weight_clip_val/act_clip_val) from checkpoint...")
            import os
            import json
            state_dict = {}
            checkpoint_dir = model_args.input_model_filename
            
            index_path_safe = os.path.join(checkpoint_dir, "model.safetensors.index.json")
            index_path_bin = os.path.join(checkpoint_dir, "pytorch_model.bin.index.json")
            
            loaded = False
            if os.path.exists(index_path_safe):
                from safetensors.torch import load_file as safe_load_file
                with open(index_path_safe, "r") as f:
                    index_data = json.load(f)
                shard_files = {v for k, v in index_data["weight_map"].items() if "weight_clip_val" in k or "act_clip_val" in k}
                for shard_file in shard_files:
                    shard_path = os.path.join(checkpoint_dir, shard_file)
                    shard_sd = safe_load_file(shard_path)
                    for k, v in shard_sd.items():
                        if "weight_clip_val" in k or "act_clip_val" in k:
                            state_dict[k] = v
                loaded = True
            elif os.path.exists(index_path_bin):
                with open(index_path_bin, "r") as f:
                    index_data = json.load(f)
                shard_files = {v for k, v in index_data["weight_map"].items() if "weight_clip_val" in k or "act_clip_val" in k}
                for shard_file in shard_files:
                    shard_path = os.path.join(checkpoint_dir, shard_file)
                    shard_sd = torch.load(shard_path, map_location="cpu")
                    for k, v in shard_sd.items():
                        if "weight_clip_val" in k or "act_clip_val" in k:
                            state_dict[k] = v
                loaded = True
            else:
                # Check for single file
                single_safe = os.path.join(checkpoint_dir, "model.safetensors")
                single_bin = os.path.join(checkpoint_dir, "pytorch_model.bin")
                if os.path.exists(single_safe):
                    from safetensors.torch import load_file as safe_load_file
                    full_sd = safe_load_file(single_safe)
                    state_dict = {k: v for k, v in full_sd.items() if "weight_clip_val" in k or "act_clip_val" in k}
                    loaded = True
                elif os.path.exists(single_bin):
                    full_sd = torch.load(single_bin, map_location="cpu")
                    state_dict = {k: v for k, v in full_sd.items() if "weight_clip_val" in k or "act_clip_val" in k}
                    loaded = True
            
            if loaded and len(state_dict) > 0:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                log.info(f"Loaded {len(state_dict)} weight_clip_val/act_clip_val parameters. Missing keys size: {len(missing_keys)}, Unexpected keys size: {len(unexpected_keys)}")
            elif not loaded:
                log.warning("Could not find index.json or checkpoint files in input_model_filename to load weight_clip_val/act_clip_val!")
            else:
                log.warning("No weight_clip_val/act_clip_val parameters found in the checkpoint files!")

    model.cuda()
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

    train_dataset, valid_dataset = datautils.get_train_val_dataset(
        train_path=data_args.train_data_local_path,
        valid_path=data_args.eval_data_local_path
        if data_args.eval_data_local_path is not None
        else None,
    )
    train_data = datautils.CustomJsonDataset(
        train_dataset, tokenizer, block_size=training_args.model_max_length
    )
    valid_data = datautils.CustomJsonDataset(
        valid_dataset, tokenizer, block_size=min(training_args.model_max_length, 1024)
    )
    model.config.use_cache = False
    myTrainer = MuonTrainer
    trainer = myTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=valid_data if training_args.do_eval else None,
        data_collator=default_data_collator,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_state()
        utils.safe_save_model_for_hf_trainer(trainer, model_args.output_model_local_path)

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
