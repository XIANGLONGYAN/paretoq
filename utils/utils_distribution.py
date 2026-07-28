from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import torch
from torch import distributed as dist
from transformers import TrainerCallback


_SUMMARY_QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
_MARGIN_HISTOGRAM_EDGES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
_DETAIL_COUNT = 16


def _is_rank_zero():
    return not dist.is_initialized() or dist.get_rank() == 0


def _sample_statistics(values):
    values = values.detach().reshape(-1).float()
    if values.numel() == 0:
        return {}

    probabilities = torch.tensor(
        _SUMMARY_QUANTILES,
        device=values.device,
        dtype=torch.float32,
    )
    quantile_values = torch.quantile(values, probabilities)
    quantiles = {
        f"p{probability * 100:g}": float(value)
        for probability, value in zip(
            _SUMMARY_QUANTILES,
            quantile_values.cpu().tolist(),
        )
    }
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "abs_mean": float(values.abs().mean()),
        "rms": float(values.square().mean().sqrt()),
        "quantiles": quantiles,
    }


def _margin_histogram(values):
    values = values.detach().reshape(-1).float()
    histogram = {}
    for lower, upper in zip(
        _MARGIN_HISTOGRAM_EDGES[:-1],
        _MARGIN_HISTOGRAM_EDGES[1:],
    ):
        count = values.ge(lower).logical_and(values.lt(upper)).sum()
        histogram[f"[{lower:g},{upper:g})"] = int(count)
    histogram[f"[{_MARGIN_HISTOGRAM_EDGES[-1]:g},inf)"] = int(
        values.ge(_MARGIN_HISTOGRAM_EDGES[-1]).sum()
    )
    return histogram


class LayerWeightTransitionTracker:
    """Tracks a deterministic sample of quantized coefficients for one layer."""

    def __init__(
        self,
        layer_name,
        sample_size,
        boundary_epsilon,
        frequent_flip_threshold,
        seed,
    ):
        if sample_size <= 0:
            raise ValueError("distribution_sample_size must be positive")
        if not 0.0 < boundary_epsilon < 0.5:
            raise ValueError(
                "distribution_boundary_epsilon must be between 0 and 0.5"
            )
        if not 1 <= frequent_flip_threshold <= 254:
            raise ValueError(
                "distribution_frequent_flip_threshold must be in [1, 254]"
            )

        self.layer_name = layer_name
        self.sample_size = sample_size
        self.boundary_epsilon = boundary_epsilon
        self.frequent_flip_threshold = frequent_flip_threshold
        self.flip_count_cap = max(10, frequent_flip_threshold)

        layer_digest = hashlib.sha256(layer_name.encode("utf-8")).digest()
        layer_seed = int.from_bytes(layer_digest[:8], byteorder="little")
        self.seed = (int(seed) + layer_seed) % (2**32)

        self.current_step = 0
        self.window_start_step = 1
        self.last_observed_step = None
        self.observed_transition_steps = 0

        self._sample_indices = None
        self._previous_codes = None
        self._flip_counts = None

    def set_current_step(self, step):
        self.current_step = int(step)

    def set_window_start(self, step):
        self.window_start_step = int(step)

    def _ensure_buffers(self, numel, device):
        if self._sample_indices is not None:
            if self._sample_indices.device != device:
                raise RuntimeError(
                    f"Tracked weight for {self.layer_name} moved from "
                    f"{self._sample_indices.device} to {device}"
                )
            return

        tracked_count = min(self.sample_size, numel)
        if tracked_count == numel:
            sample_indices = list(range(numel))
        else:
            generator = random.Random(self.seed)
            sample_indices = generator.sample(range(numel), tracked_count)

        self._sample_indices = torch.tensor(
            sample_indices,
            dtype=torch.long,
            device=device,
        )
        self._flip_counts = torch.zeros(
            tracked_count,
            dtype=torch.uint8,
            device=device,
        )

    @torch.no_grad()
    def observe(self, q_index):
        """Observe at most once per optimizer step.

        Gradient accumulation and checkpoint recomputation can execute the same
        quantizer multiple times. The global-step guard prevents those forwards
        from being counted as weight transitions.
        """
        if self.last_observed_step == self.current_step:
            return

        flat_codes = q_index.detach().reshape(-1)
        self._ensure_buffers(flat_codes.numel(), flat_codes.device)
        current_codes = flat_codes.index_select(0, self._sample_indices).to(
            dtype=torch.int8
        )

        if self._previous_codes is None:
            self._previous_codes = current_codes.clone()
        else:
            changed = current_codes.ne(self._previous_codes)
            can_increment = self._flip_counts.lt(self.flip_count_cap)
            self._flip_counts.add_(
                changed.logical_and(can_increment).to(dtype=torch.uint8)
            )
            self._previous_codes.copy_(current_codes)
            self.observed_transition_steps += 1

        self.last_observed_step = self.current_step

    def _sample_step_values(self, step, transformed_weight):
        group_width = transformed_weight.size(-1)
        group_indices = torch.div(
            self._sample_indices,
            group_width,
            rounding_mode="floor",
        )
        return step.detach().reshape(-1).index_select(0, group_indices)

    def _detail_rows(
        self,
        positions,
        raw_sample,
        transformed_sample,
        normalized_sample,
        code_sample,
        margin_sample,
    ):
        if positions.numel() == 0:
            return []

        positions = positions.to(dtype=torch.long)
        flat_indices = self._sample_indices.index_select(0, positions)
        flip_counts = self._flip_counts.index_select(0, positions)

        columns = [
            flat_indices.cpu().tolist(),
            raw_sample.index_select(0, positions).float().cpu().tolist(),
            transformed_sample.index_select(0, positions).float().cpu().tolist(),
            normalized_sample.index_select(0, positions).float().cpu().tolist(),
            code_sample.index_select(0, positions).to(torch.int16).cpu().tolist(),
            margin_sample.index_select(0, positions).float().cpu().tolist(),
            flip_counts.cpu().tolist(),
        ]
        return [
            {
                "flat_index": int(flat_index),
                "raw_weight": float(raw_value),
                "hadamard_weight": float(transformed_value),
                "normalized_weight": float(normalized_value),
                "quantized_code": int(code),
                "boundary_margin_in_steps": float(margin),
                "window_flip_count": int(flip_count),
            }
            for (
                flat_index,
                raw_value,
                transformed_value,
                normalized_value,
                code,
                margin,
                flip_count,
            ) in zip(*columns)
        ]

    @torch.no_grad()
    def build_snapshot(
        self,
        raw_weight,
        origin_shape,
        transformed_weight,
        scale,
        step,
        q_index,
        num_bits,
        group_size,
    ):
        self.observe(q_index)

        qmax = int(round(2 ** (num_bits - 1) - 1))
        if qmax < 1:
            raise ValueError(
                f"Unsupported num_bits={num_bits}: positive qmax is required"
            )

        raw_flat = raw_weight.detach().reshape(-1)
        transformed_flat = transformed_weight.detach().reshape(-1)
        q_index_flat = q_index.detach().reshape(-1)

        raw_sample = raw_flat.index_select(0, self._sample_indices)
        transformed_sample = transformed_flat.index_select(
            0, self._sample_indices
        )
        sample_step = self._sample_step_values(step, transformed_weight)
        normalized_sample = transformed_sample / sample_step
        code_sample = q_index_flat.index_select(0, self._sample_indices)

        code_float = code_sample.float()
        lower_margin = (
            normalized_sample.float() - (code_float - 0.5)
        ).abs()
        upper_margin = (
            normalized_sample.float() - (code_float + 0.5)
        ).abs()
        lower_margin.masked_fill_(code_float.le(-qmax), float("inf"))
        upper_margin.masked_fill_(code_float.ge(qmax), float("inf"))
        margin_sample = torch.minimum(lower_margin, upper_margin)

        # Exact full-layer boundary counts. Adjacent decision boundaries are
        # one step apart, so epsilon < 0.5 guarantees non-overlapping bands.
        normalized_weight = transformed_weight.detach() / step.detach()
        near_boundary_by_pair = {}
        near_boundary_count = 0
        for lower_code in range(-qmax, qmax):
            boundary = lower_code + 0.5
            count = int(
                normalized_weight.sub(boundary)
                .abs()
                .le(self.boundary_epsilon)
                .sum()
            )
            near_boundary_by_pair[f"{lower_code}<->{lower_code + 1}"] = {
                "count": count,
                "ratio": count / transformed_weight.numel(),
            }
            near_boundary_count += count

        code_occupancy = {}
        for code in range(-qmax, qmax + 1):
            count = int(q_index.eq(code).sum())
            code_occupancy[str(code)] = {
                "count": count,
                "ratio": count / q_index.numel(),
            }

        saturation_count = int(
            transformed_weight.detach().abs().ge(scale.detach()).sum()
        )

        flip_counts = self._flip_counts
        tracked_count = flip_counts.numel()
        ever_flipped_count = int(flip_counts.gt(0).sum())
        frequent_flip_count = int(
            flip_counts.ge(self.frequent_flip_threshold).sum()
        )
        flip_histogram = {
            "0": int(flip_counts.eq(0).sum()),
            "1": int(flip_counts.eq(1).sum()),
            "2": int(flip_counts.eq(2).sum()),
            "3-4": int(
                flip_counts.ge(3).logical_and(flip_counts.le(4)).sum()
            ),
            "5-9": int(
                flip_counts.ge(5).logical_and(flip_counts.le(9)).sum()
            ),
            ">=10": int(flip_counts.ge(10).sum()),
        }

        detail_count = min(_DETAIL_COUNT, tracked_count)
        closest_positions = torch.topk(
            margin_sample,
            k=detail_count,
            largest=False,
        ).indices

        positive_flip_positions = torch.nonzero(
            flip_counts.gt(0),
            as_tuple=False,
        ).reshape(-1)
        if positive_flip_positions.numel() > 0:
            positive_flip_values = flip_counts.index_select(
                0, positive_flip_positions
            ).to(torch.int16)
            frequent_order = torch.topk(
                positive_flip_values,
                k=min(detail_count, positive_flip_values.numel()),
                largest=True,
            ).indices
            frequent_positions = positive_flip_positions.index_select(
                0, frequent_order
            )
        else:
            frequent_positions = positive_flip_positions

        random_positions = torch.arange(
            min(detail_count, tracked_count),
            device=self._sample_indices.device,
        )

        sample_details = {
            "closest_to_boundary": self._detail_rows(
                closest_positions,
                raw_sample,
                transformed_sample,
                normalized_sample,
                code_sample,
                margin_sample,
            ),
            "most_frequently_flipped": self._detail_rows(
                frequent_positions,
                raw_sample,
                transformed_sample,
                normalized_sample,
                code_sample,
                margin_sample,
            ),
            "deterministic_random_sample": self._detail_rows(
                random_positions,
                raw_sample,
                transformed_sample,
                normalized_sample,
                code_sample,
                margin_sample,
            ),
        }

        del normalized_weight

        return {
            "layer_name": self.layer_name,
            "shape": list(origin_shape),
            "numel": raw_weight.numel(),
            "num_bits": num_bits,
            "group_size": group_size,
            "sample_size": tracked_count,
            "window_start_step": self.window_start_step,
            "window_end_step": self.current_step,
            "observed_transition_steps": self.observed_transition_steps,
            "raw_weight_sample_statistics": _sample_statistics(raw_sample),
            "hadamard_weight_sample_statistics": _sample_statistics(
                transformed_sample
            ),
            "normalized_weight_sample_statistics": _sample_statistics(
                normalized_sample
            ),
            "scale_statistics": _sample_statistics(scale),
            "step_statistics": _sample_statistics(step),
            "code_occupancy_exact": code_occupancy,
            "saturation": {
                "count": saturation_count,
                "ratio": saturation_count / transformed_weight.numel(),
            },
            "near_boundary": {
                "epsilon_in_steps": self.boundary_epsilon,
                "count": near_boundary_count,
                "ratio": near_boundary_count / transformed_weight.numel(),
                "by_transition_pair": near_boundary_by_pair,
            },
            "sampled_boundary_margin_statistics": _sample_statistics(
                margin_sample
            ),
            "sampled_boundary_margin_histogram": _margin_histogram(
                margin_sample
            ),
            "sampled_transitions": {
                "frequent_flip_threshold": self.frequent_flip_threshold,
                "flip_count_cap": self.flip_count_cap,
                "ever_flipped_count": ever_flipped_count,
                "ever_flipped_ratio": ever_flipped_count / tracked_count,
                "frequently_flipped_count": frequent_flip_count,
                "frequently_flipped_ratio": frequent_flip_count / tracked_count,
                "flip_count_histogram": flip_histogram,
            },
            "sample_details": sample_details,
        }

    def reset_window(self, next_window_start_step):
        if self._flip_counts is not None:
            self._flip_counts.zero_()
        self.observed_transition_steps = 0
        self.window_start_step = int(next_window_start_step)


class HadamardWeightDistributionCallback(TrainerCallback):
    """Writes rank-zero diagnostics every N optimizer steps."""

    def __init__(
        self,
        model,
        output_dir="./distribution",
        interval=500,
        boundary_epsilon=0.05,
        frequent_flip_threshold=5,
        sample_size=65536,
        seed=42,
    ):
        if interval <= 0:
            raise ValueError("distribution_log_interval must be positive")

        self.enabled = _is_rank_zero()
        self.interval = int(interval)
        self.output_dir = Path(output_dir)
        self.run_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.layers = []

        if not self.enabled:
            return

        from utils.utils_myquant import (
            HadamardGaussianTrustQuantizer,
            MyQuantizeLinear,
        )

        for layer_name, layer in model.named_modules():
            if not isinstance(layer, MyQuantizeLinear):
                continue
            if not isinstance(
                layer.w_quantizer,
                HadamardGaussianTrustQuantizer,
            ):
                continue
            if layer.w_quantizer.num_bits >= 16:
                continue

            tracker = LayerWeightTransitionTracker(
                layer_name=layer_name,
                sample_size=sample_size,
                boundary_epsilon=boundary_epsilon,
                frequent_flip_threshold=frequent_flip_threshold,
                seed=seed,
            )
            layer.w_quantizer.set_distribution_tracker(tracker)
            self.layers.append((layer_name, layer, tracker))

        self.quantizer_name = HadamardGaussianTrustQuantizer.__name__
        if self.layers:
            (self.output_dir / self.quantizer_name).mkdir(
                parents=True,
                exist_ok=True,
            )

    @property
    def num_layers(self):
        return len(self.layers)

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        start_step = int(state.global_step)
        for _, _, tracker in self.layers:
            tracker.set_current_step(start_step)
            tracker.set_window_start(start_step + 1)

    def on_step_end(self, args, state, control, **kwargs):
        if not self.enabled or not self.layers:
            return

        current_step = int(state.global_step)
        for _, _, tracker in self.layers:
            tracker.set_current_step(current_step)

        if current_step <= 0 or current_step % self.interval != 0:
            return

        snapshots = []
        for _, layer, _ in self.layers:
            snapshot = layer.w_quantizer.capture_distribution_snapshot(
                layer.weight
            )
            if snapshot is not None:
                snapshots.append(snapshot)

        self._write_snapshot(current_step, snapshots)

        for _, _, tracker in self.layers:
            tracker.reset_window(current_step + 1)

    def _global_summary(self, snapshots):
        total_numel = sum(snapshot["numel"] for snapshot in snapshots)
        near_count = sum(
            snapshot["near_boundary"]["count"] for snapshot in snapshots
        )
        saturation_count = sum(
            snapshot["saturation"]["count"] for snapshot in snapshots
        )
        sample_count = sum(snapshot["sample_size"] for snapshot in snapshots)
        frequent_count = sum(
            snapshot["sampled_transitions"]["frequently_flipped_count"]
            for snapshot in snapshots
        )
        ever_flipped_count = sum(
            snapshot["sampled_transitions"]["ever_flipped_count"]
            for snapshot in snapshots
        )

        occupancy_count = {}
        for snapshot in snapshots:
            for code, values in snapshot["code_occupancy_exact"].items():
                occupancy_count[code] = (
                    occupancy_count.get(code, 0) + values["count"]
                )
        occupancy = {
            code: {
                "count": count,
                "ratio": count / total_numel,
            }
            for code, count in occupancy_count.items()
        }

        top_near_boundary_layers = sorted(
            (
                {
                    "layer_name": snapshot["layer_name"],
                    "ratio": snapshot["near_boundary"]["ratio"],
                }
                for snapshot in snapshots
            ),
            key=lambda item: item["ratio"],
            reverse=True,
        )[:20]
        top_frequently_flipped_layers = sorted(
            (
                {
                    "layer_name": snapshot["layer_name"],
                    "ratio": snapshot["sampled_transitions"][
                        "frequently_flipped_ratio"
                    ],
                }
                for snapshot in snapshots
            ),
            key=lambda item: item["ratio"],
            reverse=True,
        )[:20]

        return {
            "layer_count": len(snapshots),
            "total_weight_elements": total_numel,
            "near_boundary_count": near_count,
            "near_boundary_ratio": near_count / total_numel,
            "saturation_count": saturation_count,
            "saturation_ratio": saturation_count / total_numel,
            "transition_sample_count": sample_count,
            "ever_flipped_sample_count": ever_flipped_count,
            "ever_flipped_sample_ratio": ever_flipped_count / sample_count,
            "frequently_flipped_sample_count": frequent_count,
            "frequently_flipped_sample_ratio": frequent_count / sample_count,
            "code_occupancy": occupancy,
            "top_near_boundary_layers": top_near_boundary_layers,
            "top_frequently_flipped_layers": top_frequently_flipped_layers,
        }

    def _write_snapshot(self, current_step, snapshots):
        if not snapshots:
            return

        generated_at = datetime.now().isoformat(timespec="seconds")
        file_path = (
            self.output_dir
            / self.quantizer_name
            / f"{self.run_time}_{current_step:08d}.txt"
        )
        temporary_path = file_path.with_suffix(".txt.tmp")

        lines = [
            "# HadamardGaussianTrustQuantizer weight-transition diagnostics",
            f"generated_at = {generated_at}",
            f"global_step = {current_step}",
            f"quantizer = {self.quantizer_name}",
            "global_summary = "
            f"{json.dumps(self._global_summary(snapshots), ensure_ascii=False, sort_keys=True)}",
        ]

        for snapshot in snapshots:
            lines.append("")
            lines.append(f"[layer {json.dumps(snapshot['layer_name'])}]")
            for key, value in snapshot.items():
                if key == "layer_name":
                    continue
                lines.append(
                    f"{key} = "
                    f"{json.dumps(value, ensure_ascii=False, sort_keys=True)}"
                )

        temporary_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(file_path)
