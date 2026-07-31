#!/usr/bin/env python3
"""Inspect trained Butterfly theta parameters directly from a checkpoint."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


THETA_SUFFIX = ".butterfly.theta"
DEFAULT_THRESHOLDS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare saved Butterfly theta parameters with their initialization. "
            "The script reads model weights directly and does not construct the model."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Checkpoint directory or a model.safetensors/pytorch_model.bin file."
        ),
    )
    parser.add_argument(
        "--init",
        choices=("identity", "hadamard"),
        default="hadamard",
        help="Initialization used for the inspected run.",
    )
    parser.add_argument(
        "--hadamard-block-size",
        type=int,
        default=128,
        help=(
            "Hadamard block size used at initialization. With block size 128, "
            "stages 0-6 start at -pi/4 and later stages start at zero."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance used to classify a theta value as unchanged.",
    )
    parser.add_argument(
        "--top-layers",
        type=int,
        default=20,
        help=(
            "Print this many layers ranked by maximum |theta - theta_init|. "
            "Use 0 to print every layer."
        ),
    )
    parser.add_argument(
        "--largest-values",
        type=int,
        default=20,
        help="Print this many individual theta values with the largest changes.",
    )
    parser.add_argument(
        "--show-stages",
        action="store_true",
        help="Also print per-stage statistics for every theta tensor.",
    )
    return parser.parse_args()


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def _load_torch_file(path, requested_keys=None):
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")

    if isinstance(state_dict, dict) and isinstance(
        state_dict.get("state_dict"),
        dict,
    ):
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"{path} does not contain a state dictionary.")

    keys = requested_keys if requested_keys is not None else state_dict.keys()
    return {
        key: state_dict[key]
        for key in keys
        if key in state_dict and key.endswith(THETA_SUFFIX)
    }


def _load_safetensors_file(path, requested_keys=None):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "Reading a safetensors checkpoint requires the safetensors package."
        ) from exc

    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        available_keys = set(handle.keys())
        keys = requested_keys if requested_keys is not None else available_keys
        for key in keys:
            if key in available_keys and key.endswith(THETA_SUFFIX):
                tensors[key] = handle.get_tensor(key)
    return tensors


def _load_weight_file(path, requested_keys=None):
    if path.suffix == ".safetensors":
        return _load_safetensors_file(path, requested_keys)
    return _load_torch_file(path, requested_keys)


def _find_weight_files(checkpoint):
    if checkpoint.is_file():
        return [(checkpoint, None)]
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    index_names = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    for index_name in index_names:
        index_path = checkpoint / index_name
        if not index_path.is_file():
            continue

        with index_path.open("r", encoding="utf-8") as stream:
            weight_map = json.load(stream).get("weight_map", {})
        theta_map = {
            key: filename
            for key, filename in weight_map.items()
            if key.endswith(THETA_SUFFIX)
        }
        if not theta_map:
            raise KeyError(f"No {THETA_SUFFIX!r} keys found in {index_path}.")

        keys_by_file = defaultdict(list)
        for key, filename in theta_map.items():
            keys_by_file[filename].append(key)
        return [
            (checkpoint / filename, keys)
            for filename, keys in sorted(keys_by_file.items())
        ]

    single_file_names = (
        "model.safetensors",
        "pytorch_model.bin",
    )
    for filename in single_file_names:
        weight_path = checkpoint / filename
        if weight_path.is_file():
            return [(weight_path, None)]

    candidates = sorted(checkpoint.glob("model*.safetensors"))
    candidates.extend(sorted(checkpoint.glob("pytorch_model*.bin")))
    if candidates:
        return [(path, None) for path in candidates]

    raise FileNotFoundError(
        "No model.safetensors or pytorch_model.bin weights were found under "
        f"{checkpoint}."
    )


def load_theta_tensors(checkpoint):
    theta_tensors = {}
    for weight_path, requested_keys in _find_weight_files(checkpoint):
        if not weight_path.is_file():
            raise FileNotFoundError(f"Weight shard does not exist: {weight_path}")
        loaded = _load_weight_file(weight_path, requested_keys)
        duplicate_keys = set(theta_tensors).intersection(loaded)
        if duplicate_keys:
            raise KeyError(
                f"Duplicate theta keys encountered: {sorted(duplicate_keys)}"
            )
        theta_tensors.update(loaded)

    if not theta_tensors:
        raise KeyError(
            f"No parameters ending in {THETA_SUFFIX!r} were found in {checkpoint}."
        )
    return theta_tensors


def build_expected_theta(theta, init, hadamard_block_size):
    if theta.ndim != 2:
        raise ValueError(
            f"Expected a 2D theta tensor, got shape {tuple(theta.shape)}."
        )

    num_stages, half_feature_dim = theta.shape
    feature_dim = 2 * half_feature_dim
    if not _is_power_of_two(feature_dim):
        raise ValueError(
            f"Inferred feature dimension must be a power of two, got {feature_dim}."
        )
    expected_num_stages = int(math.log2(feature_dim))
    if num_stages != expected_num_stages:
        raise ValueError(
            f"Theta shape {tuple(theta.shape)} implies feature_dim={feature_dim}, "
            f"which expects {expected_num_stages} stages."
        )

    expected = torch.zeros_like(theta, dtype=torch.float64, device="cpu")
    if init == "hadamard":
        if (
            not _is_power_of_two(hadamard_block_size)
            or hadamard_block_size > feature_dim
            or feature_dim % hadamard_block_size != 0
        ):
            raise ValueError(
                f"Invalid hadamard_block_size={hadamard_block_size} for "
                f"feature_dim={feature_dim}."
            )
        hadamard_stages = int(math.log2(hadamard_block_size))
        expected[:hadamard_stages].fill_(-math.pi / 4)

    return expected, feature_dim


def tensor_summary(name, theta, expected, atol):
    theta64 = theta.detach().to(device="cpu", dtype=torch.float64)
    delta = theta64 - expected
    abs_delta = delta.abs()
    return {
        "name": name,
        "theta": theta64,
        "expected": expected,
        "delta": delta,
        "count": delta.numel(),
        "mean_abs_delta": abs_delta.mean().item(),
        "rms_delta": delta.square().mean().sqrt().item(),
        "max_abs_delta": abs_delta.max().item(),
        "changed_fraction": (abs_delta > atol).double().mean().item(),
    }


def print_global_summary(summaries, atol):
    all_deltas = torch.cat(
        [summary["delta"].reshape(-1) for summary in summaries]
    )
    all_abs_deltas = all_deltas.abs()
    quantile_levels = torch.tensor(
        [0.5, 0.9, 0.99, 0.999],
        dtype=torch.float64,
    )
    quantiles = torch.quantile(all_abs_deltas, quantile_levels)

    print("\n=== Global summary ===")
    print(f"theta tensors              : {len(summaries)}")
    print(f"theta parameters           : {all_deltas.numel():,}")
    print(f"mean |delta|               : {all_abs_deltas.mean().item():.9e}")
    print(
        "RMS(delta)                 : "
        f"{all_deltas.square().mean().sqrt().item():.9e}"
    )
    print(f"max |delta|                : {all_abs_deltas.max().item():.9e}")
    print(f"unchanged fraction (@{atol:g}): {(all_abs_deltas <= atol).double().mean().item():.6%}")
    for threshold in DEFAULT_THRESHOLDS:
        changed = (all_abs_deltas > threshold).double().mean().item()
        print(f"fraction |delta| > {threshold:>7g}: {changed:.6%}")
    print(
        "|delta| quantiles          : "
        + ", ".join(
            f"p{level * 100:g}={value.item():.9e}"
            for level, value in zip(quantile_levels, quantiles)
        )
    )

    if torch.all(all_abs_deltas <= atol):
        print(
            f"verdict                     : theta is unchanged within atol={atol:g}"
        )
    else:
        print(
            f"verdict                     : theta changed beyond atol={atol:g}"
        )


def print_layer_summaries(summaries, top_layers):
    ranked = sorted(
        summaries,
        key=lambda item: item["max_abs_delta"],
        reverse=True,
    )
    selected = ranked if top_layers == 0 else ranked[:top_layers]

    print("\n=== Layers ranked by max |delta| ===")
    for index, summary in enumerate(selected, start=1):
        print(
            f"{index:3d}. {summary['name']} "
            f"shape={tuple(summary['theta'].shape)} "
            f"mean|d|={summary['mean_abs_delta']:.6e} "
            f"rms(d)={summary['rms_delta']:.6e} "
            f"max|d|={summary['max_abs_delta']:.6e} "
            f"changed={summary['changed_fraction']:.3%}"
        )


def print_stage_summaries(summaries, atol):
    print("\n=== Per-stage summaries ===")
    for summary in sorted(summaries, key=lambda item: item["name"]):
        print(f"\n{summary['name']}")
        theta = summary["theta"]
        expected = summary["expected"]
        for stage in range(theta.shape[0]):
            values = theta[stage]
            expected_value = expected[stage, 0].item()
            abs_delta = (values - expected_value).abs()
            print(
                f"  stage={stage:2d} init={expected_value:+.9f} "
                f"mean={values.mean().item():+.9f} "
                f"std={values.std(unbiased=False).item():.3e} "
                f"min={values.min().item():+.9f} "
                f"max={values.max().item():+.9f} "
                f"mean|d|={abs_delta.mean().item():.3e} "
                f"max|d|={abs_delta.max().item():.3e} "
                f"changed={(abs_delta > atol).double().mean().item():.3%}"
            )


def print_largest_values(summaries, count):
    if count <= 0:
        return

    candidates = []
    for summary in summaries:
        flat_abs_delta = summary["delta"].abs().reshape(-1)
        local_count = min(count, flat_abs_delta.numel())
        values, indices = torch.topk(flat_abs_delta, local_count)
        width = summary["theta"].shape[1]
        for abs_value, flat_index in zip(values.tolist(), indices.tolist()):
            stage = flat_index // width
            pair_index = flat_index % width
            theta_value = summary["theta"][stage, pair_index].item()
            expected_value = summary["expected"][stage, pair_index].item()
            candidates.append(
                (
                    abs_value,
                    summary["name"],
                    stage,
                    pair_index,
                    theta_value,
                    expected_value,
                    theta_value - expected_value,
                )
            )

    candidates.sort(key=lambda item: item[0], reverse=True)
    print("\n=== Individual theta values with largest changes ===")
    for rank, item in enumerate(candidates[:count], start=1):
        _, name, stage, pair_index, theta_value, expected_value, delta = item
        print(
            f"{rank:3d}. {name}[stage={stage}, pair={pair_index}] "
            f"theta={theta_value:+.9f} init={expected_value:+.9f} "
            f"delta={delta:+.9e}"
        )


def main():
    args = parse_args()
    if args.atol < 0:
        raise ValueError(f"atol must be non-negative, got {args.atol}.")
    if args.top_layers < 0:
        raise ValueError(
            f"top_layers must be non-negative, got {args.top_layers}."
        )
    if args.largest_values < 0:
        raise ValueError(
            "largest_values must be non-negative, "
            f"got {args.largest_values}."
        )

    checkpoint = args.checkpoint.expanduser().resolve()
    theta_tensors = load_theta_tensors(checkpoint)

    print(f"checkpoint                  : {checkpoint}")
    print(f"declared initialization     : {args.init}")
    if args.init == "hadamard":
        stages = int(math.log2(args.hadamard_block_size))
        print(f"Hadamard block size         : {args.hadamard_block_size}")
        print(
            "expected theta convention   : "
            f"stages 0-{stages - 1} = -pi/4, later stages = 0"
        )
        print(
            "note                        : -pi/4 is required by this "
            "implementation's Givens/sign convention"
        )

    summaries = []
    feature_dims = set()
    for name, theta in sorted(theta_tensors.items()):
        expected, feature_dim = build_expected_theta(
            theta,
            args.init,
            args.hadamard_block_size,
        )
        feature_dims.add(feature_dim)
        summaries.append(tensor_summary(name, theta, expected, args.atol))

    print(f"feature dimensions found    : {sorted(feature_dims)}")
    print_global_summary(summaries, args.atol)
    print_layer_summaries(summaries, args.top_layers)
    print_largest_values(summaries, args.largest_values)
    if args.show_stages:
        print_stage_summaries(summaries, args.atol)


if __name__ == "__main__":
    main()
