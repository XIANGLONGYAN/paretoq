#!/usr/bin/env python3
import argparse
import os
import pickle
import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
from utils import datautils

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trace_estimation.calibrator import HessianTraceCalibrator
import utils.utils_hestia as hestia_mod

# from hestia.quant_config import get_quant_config

'''
def build_tokenizer(model_path: str):
    """Build tokenizer from model path."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_data_collator(tokenizer):
    """Build data collator for language modeling."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )


def build_dataset(data_path: str, seed: int = 42):
    """Load and shuffle dataset."""
    dataset = load_from_disk(data_path)
    dataset = dataset.shuffle(seed=seed)
    return dataset
'''

'''
def create_calibration_dataloader(
    train_dataset,
    data_collator,
    tokenizer,
    calibrate_num_batches=5,
    calibrate_batch_size=1,
):
    """
    Create calibration dataloader - EXACTLY matches train.py logic.

    Args:
        train_dataset: Full training dataset
        data_collator: Data collator function
        tokenizer: Tokenizer for processing
        calibrate_num_batches: Number of batches to use
        calibrate_batch_size: Batch size for calibration

    Returns:
        DataLoader for calibration
    """
    # Use a small subset of the dataset for calibration
    calibrate_size = calibrate_num_batches * calibrate_batch_size
    calibrate_size = min(calibrate_size, len(train_dataset))

    # Create subset
    indices = list(range(calibrate_size))
    calibrate_subset = Subset(train_dataset, indices)

    # Create dataloader
    calibrate_dataloader = DataLoader(
        calibrate_subset,
        batch_size=calibrate_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
    )

    return calibrate_dataloader
    
    '''


def create_calib_dataloader(
    calib_dataset,
    calib_num_batches,
    calib_batch_size
):
    calib_size = min(calib_num_batches * calib_batch_size, len(calib_dataset))
    
    indices = list(range(calib_size))
    calib_subset = Subset(calib_dataset, indices)
    calib_dataloader = DataLoader(
        calib_subset,
        batch_size=calib_batch_size,
        shuffle=True,
        num_workers=0
    )
    return calib_dataloader

def run_calibration(
    model,
    dataloader,
    num_sketch,
    num_query,
    num_batches,
    device='cuda',
    use_gradient_checkpointing=True,
    max_seq_len=None,
    calibration_granularity='layer',
    kappa=1.0,
    alpha=0.5,
    skip_keywords=None,
):
    """
    Run Hessian trace calibration - EXACTLY matches train.py callback logic.

    Args:
        model: The model to calibrate
        quant_layers: List of HestiaLinear layers
        dataloader: Calibration dataloader
        num_sketch: Hutch++ sketch dimension
        num_query: Number of random samples
        num_batches: Number of batches for estimation
        device: Device to run on
        use_gradient_checkpointing: Enable gradient checkpointing to save memory
        max_seq_len: Truncate sequences to this length (saves memory)

    Returns:
        Dictionary with keys: scores, traces, stats
    """
    # Loss function - computes cross-entropy loss
    def loss_func(outputs, targets):
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            return outputs.loss
        # For CausalLM outputs, compute loss manually
        # outputs.logits shape: [batch_size, seq_len, vocab_size]
        # targets shape: [batch_size, seq_len]
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        return loss

    # Create calibrator with memory optimizations
    calibrator = HessianTraceCalibrator(
        model=model,
        loss_func=loss_func,
        dataloader=dataloader,
        device=device,
        use_gradient_checkpointing=use_gradient_checkpointing,
        max_seq_len=max_seq_len,
    )

    # Run calibration - this injects scores into layer.hessian_score
    outputs = calibrator.calibrate(
        target_modules=[nn.Linear],  # Will match HestiaLinear layers
        num_sketch=num_sketch,
        num_query=num_query,
        num_batches=num_batches,
        granularity=calibration_granularity,
        kappa=kappa,
        alpha=alpha,
        skip_keywords=skip_keywords,
        )

    scores_dict = outputs.get("scores", {}) if isinstance(outputs, dict) else {}
    traces_dict = outputs.get("traces", {}) if isinstance(outputs, dict) else {}
    temp_scales_dict = outputs.get("temp_scales", {}) if isinstance(outputs, dict) else {}
    stats_dict = outputs.get("stats", {}) if isinstance(outputs, dict) else {}

    return {"scores": scores_dict, "traces": traces_dict, "temp_scales": temp_scales_dict, "stats": stats_dict}

def parse_args():

    parser = argparse.ArgumentParser(
        description="Offline Hessian trace calibration for Hestia"
    )

    # Required arguments
    parser.add_argument(
        "--input-model-filename",
        type=str,
        required=True,
        help="Path to base model checkpoint",
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=2048
    )
    parser.add_argument(
        "--data-bin-path",
        type=str,
        required=True,
        help="Path to calibration dataset(bin)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for saved traces (e.g., traces.pkl). Will also save a .json version for easy reading.",
    )

    # Calibration parameters (override config if specified)
    parser.add_argument(
        "--num-sketch",
        type=int,
        required=True,
        help="Hutch++ sketch dimension (overrides config)",
    )
    parser.add_argument(
        "--num-query",
        type=int,
        required=True,
        help="Number of random samples (overrides config)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        required=True,
        help="Number of batches for estimation (overrides config)",
    )
    parser.add_argument(
        "--calibrate-batch-size",
        type=int,
        required=True,
        help="Batch size for calibration dataloader",
    )

    # Skip keywords
    parser.add_argument(
        "--skip-keywords",
        nargs="*",
        default=["lm_head", "embed"],
        help="Names of modules to skip during quantization",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run calibration on",
    )
    
    # Memory optimization options
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (uses more memory but faster)",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    parser.add_argument(
        "--calibration-granularity",
        choices=["tensor", "layer", "component"],
        default="tensor"
    )

    parser.add_argument(
        "--kappa",
        type=float,
        default=1.0,
        help="Sigmoid gain factor for temperature scaling (kappa).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Asynchrony coefficient for temperature scaling (alpha).",
    )

    return parser.parse_args()

def main():

    args = parse_args()

    # Print configuration
    print("=" * 60)
    print("OFFLINE HESSIAN CALIBRATION")
    print("=" * 60)
    print(f"Model: {args.input_model_filename}")
    print(f"Data: {args.data_bin_path}")
    print(f"Output: {args.output_path}")
    print(f"Device: {args.device}")
    print("-" * 60)

    # 1. Load quantization config
    print("[1/6] Loading quantization config...")

    print(f"  - Hessian params: sketch={args.num_sketch}, "
          f"query={args.num_query}, batches={args.num_batches}")

    # 2. Load model
    print("[2/6] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.input_model_filename,
        torch_dtype=torch.float32,  # Use float32 for calibration stability
        trust_remote_code=True,
        attn_implementation="eager",  # SDPA kernels do not support double backward
    )
    print(f"  - Model loaded: {type(model).__name__}")
    model.to(args.device)

    print(f"  - Model to {args.device}")

    # 3. Convert model to HestiaLinear layers
    print("[3/6] Converting model to HestiaLinear layers...")
    model = hestia_mod.replace_linear_with_hestia(
        model,
        bits=32,
        symmetric=True,
        group_size=0,
        compress_ratio=0.2,
        init_temp=1.0,
        end_temp=0.0,
        anneal_ratio=0.8,
        temp_scales_dict=None,
        skip_keywords=["lm_head", "embed"],
    )
    model.to(args.device)

    # 4. Prepare calibration data
    print("[4/6] Preparing calibration data...")
    
    dataset = datautils.CustomBinDataset(
        args.data_bin_path, args.model_max_length
    )

    dataloader = create_calib_dataloader(
        dataset, args.num_batches, args.calibrate_batch_size
    )
    
    print(f"  - Calibration dataset size: {len(dataset)}")
    print(f"  - Calibration batches: {args.num_batches}")
    print(f"  - Batch size: {args.calibrate_batch_size}")

    # 5. Bind total steps (required by scheduler)
    print("[5/6] Binding scheduler parameters...")
    # Dummy total steps since we're not training


    hestia_mod.global_total_steps = 1000  # Dummy value, only needed for scheduler assert
    print("  - Scheduler bound (dummy total_steps=1000)")

    # 6. Run calibration
    print("[6/6] Running Hessian trace calibration (Hutch++)...")
    use_gc = not args.no_gradient_checkpointing
    print(f"  - Gradient checkpointing: {'enabled' if use_gc else 'disabled'}")
    print(f"  - Max sequence length: {args.model_max_length}")
    print(f"  - This may take several minutes depending on model size...")

    outputs = run_calibration(
        model=model,
        dataloader=dataloader,
        num_sketch=args.num_sketch,
        num_query=args.num_query,
        num_batches=args.num_batches,
        device=args.device,
        use_gradient_checkpointing=use_gc,
        max_seq_len=args.model_max_length,
        calibration_granularity=args.calibration_granularity,
        kappa=args.kappa,
        alpha=args.alpha,
        skip_keywords=args.skip_keywords,
    )

    # Save results
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    scores_dict = outputs.get("scores", {})
    traces_dict = outputs.get("traces", {})
    temp_scales_dict = outputs.get("temp_scales", {})
    stats_dict = outputs.get("stats", {})
    print(f"Scores computed: {len(scores_dict)}")
    print(f"Temperature scales computed: {len(temp_scales_dict)}")

    # Show score statistics
    if scores_dict:
        all_scores = list(scores_dict.values())
        print("Score statistics:")
        print(f"  - Min: {min(all_scores):.6f}")
        print(f"  - Max: {max(all_scores):.6f}")
        print(f"  - Mean: {sum(all_scores)/len(all_scores):.6f}")
        print(f"  - Std:  {torch.tensor(all_scores).std():.6f}")
        print(f"  - Granularity: {args.calibration_granularity}")
        print(f"  - Total scores: {len(scores_dict)}")

        if stats_dict:
            print(f"  - log(trace) mean: {stats_dict.get('log_mean', 0.0):.6f}")
            print(f"  - log(trace) std:  {stats_dict.get('log_std', 0.0):.6f}")

        # Show first few score details
        print("\nFirst 5 score details:")
        for layer_id, score_value in list(scores_dict.items())[:5]:
            print(f"  - {layer_id}: {score_value:.6f}")

    # Show temperature scaling factor statistics
    if temp_scales_dict:
        all_temp_scales = list(temp_scales_dict.values())
        print("\nTemperature scaling factor statistics:")
        print(f"  - Min: {min(all_temp_scales):.6f}")
        print(f"  - Max: {max(all_temp_scales):.6f}")
        print(f"  - Mean: {sum(all_temp_scales)/len(all_temp_scales):.6f}")
        print(f"  - Std:  {torch.tensor(all_temp_scales).std():.6f}")
        print(f"  - Total temp_scales: {len(temp_scales_dict)}")

        # Show first few temp_scale details
        print("\nFirst 5 temp_scale details:")
        for layer_id, temp_scale_value in list(temp_scales_dict.items())[:5]:
            print(f"  - {layer_id}: {temp_scale_value:.6f}")

   # Save to file - both pkl (for program) and json (for human reading)
   # Save as pickle (for program loading)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(outputs, f)

    # Also save as JSON for easy reading
    json_path = args.output_path.replace('.pkl', '.json').replace('.pickle', '.json')
    if json_path == args.output_path:
        json_path = args.output_path + '.json'

    with open(json_path, 'w') as f:
        json.dump(outputs, f, indent=2)

    print(f"\nSaved calibration results to:")
    print(f"  - {args.output_path} (pickle, for program)")
    print(f"  - {json_path} (JSON, for human reading)")
    print("\nUsage in training:")
    print(f"  python train.py --hessian-traces-path {args.output_path} ...")
    print("=" * 60)


if __name__ == "__main__":
    main()
