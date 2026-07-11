"""
HESTIA -> Optional Hessian Calibration (Hutch++)

Computes per-layer sensitivity scores via Hessian trace approximation
using the Hutch++ algorithm, then converts them to temperature scaling
factors for fine-grained annealing control.

Usage:
    temp_scales = calibrate_hestia(model, dataloader, ...)
    model = replace_linear_with_hestia(model, temp_scales_dict=temp_scales, ...)

If calibration is skipped, all layers use temp_scale = 1.0 (uniform schedule).
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict, List


# ============================================================================
# Hutch++ State Manager (memory-optimized: stores intermediates on CPU)
# ============================================================================

class HutchPlusPlusState:
    """
    Manages Hutch++ trace estimation for a single parameter group.

    The algorithm runs in three phases:
      Phase 1:  Sketch subspace via H @ S (random projections)
      Phase 2:  Low-rank trace via Q^T H Q (subspace eigenvalues)
      Phase 3:  Residual trace via Hutchinson on complement subspace
    """

    def __init__(self, param_numel: int, num_sketch: int, num_query: int, device: str):
        self.numel = param_numel
        self.compute_device = device
        self.storage_device = "cpu"

        self.num_sketch = min(num_sketch, param_numel)
        self.num_query = num_query

        self.trace_low = 0.0
        self.trace_res = 0.0
        self.final_trace = 0.0

        self.S: Optional[torch.Tensor] = None
        self.Y_accum: Optional[torch.Tensor] = None
        self.Q: Optional[torch.Tensor] = None
        self.G_accum: Optional[torch.Tensor] = None
        self.Omega: Optional[torch.Tensor] = None
        self.Z_accum: Optional[torch.Tensor] = None
        self.batch_count = 0

    def init_phase1_sketch(self):
        self.S = torch.randint(0, 2, (self.numel, self.num_sketch),
                               device=self.storage_device).float() * 2 - 1
        self.Y_accum = torch.zeros_like(self.S)
        self.batch_count = 0
        return self.S

    def accumulate_phase1(self, hvp_chunk: torch.Tensor):
        self.Y_accum += hvp_chunk.to(self.storage_device)

    def finalize_phase1(self, num_batches: int):
        Y_avg = self.Y_accum / max(1, num_batches)
        self.Q, _ = torch.linalg.qr(Y_avg, mode="reduced")
        self.S = None
        self.Y_accum = None

    def init_phase2_subspace(self):
        self.G_accum = torch.zeros_like(self.Q)
        self.batch_count = 0
        return self.Q

    def accumulate_phase2(self, hvp_chunk: torch.Tensor):
        self.G_accum += hvp_chunk.to(self.storage_device)

    def finalize_phase2(self, num_batches: int):
        G_avg = self.G_accum / max(1, num_batches)
        self.trace_low = torch.sum(self.Q * G_avg).item()
        self.G_accum = None
        if self.num_sketch >= self.numel:
            self.final_trace = self.trace_low
            return True
        return False

    def init_phase3_residual(self):
        self.Omega = torch.randint(0, 2, (self.numel, self.num_query),
                                   device=self.storage_device).float() * 2 - 1
        Omega_proj = self.Q @ (self.Q.T @ self.Omega)
        Omega_perp = self.Omega - Omega_proj
        self.Z_accum = torch.zeros_like(self.Omega)
        self.batch_count = 0
        return Omega_perp

    def accumulate_phase3(self, hvp_chunk: torch.Tensor):
        self.Z_accum += hvp_chunk.to(self.storage_device)

    def finalize_phase3(self, num_batches: int):
        Y_perp_avg = self.Z_accum / max(1, num_batches)
        Y_perp_proj = self.Q @ (self.Q.T @ Y_perp_avg)
        Z = Y_perp_avg - Y_perp_proj
        self.trace_res = torch.sum(self.Omega * Z).item() / self.num_query
        self.final_trace = self.trace_low + self.trace_res
        self.Q = None
        self.Omega = None
        self.Z_accum = None


# ============================================================================
# Hessian-vector product computation
# ============================================================================

def _compute_hvp_single_vector(
    loss: torch.Tensor,
    param: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Hessian-vector product via double backward: H @ v = d/dp (grad @ v)."""
    grad = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True)[0]
    hvp = torch.autograd.grad(
        grad, param, grad_outputs=vector, retain_graph=True
    )[0]
    return hvp.detach()


# ============================================================================
# Calibration entry point
# ============================================================================

def calibrate_hestia(
    model: nn.Module,
    dataloader,
    num_samples: int = 256,
    num_sketch: int = 10,
    num_query: int = 10,
    device: str = "cuda",
    alpha: float = 1.0,
    skip_keywords: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Run Hutch++ Hessian trace calibration and return per-layer temperature scales.

    Args:
        model:        model with raw nn.Linear layers (calibration runs BEFORE replacement)
        dataloader:   calibration data loader (yields dict with "input_ids")
        num_samples:  number of calibration samples to process
        num_sketch:   Hutch++ sketch dimension
        num_query:    Hutch++ query dimension (residual phase)
        device:       compute device
        alpha:          temperature scaling strength (higher = more differentiation)
        skip_keywords:  layer name substrings to skip (e.g. ["lm_head", "embed"])

    Returns:
        dict mapping layer_id -> temp_scale (float)
    """
    # Collect nn.Linear layers (skip frozen / excluded layers)
    if skip_keywords is None:
        skip_keywords = []
    layer_params = []
    hutch_states = []

    for name, module in model.named_modules():
        if any(kw in name for kw in skip_keywords):
            continue
        if isinstance(module, nn.Linear):
            param = module.weight
            numel = param.numel()
            layer_params.append((name, param))
            hutch_states.append(
                HutchPlusPlusState(numel, num_sketch, num_query, device)
            )

    if len(layer_params) == 0:
        print("[Hestia Calib] No nn.Linear layers found, skipping calibration")
        return {}

    print(f"[Hestia Calib] Calibrating {len(layer_params)} layers "
          f"with {num_samples} samples...")

    # --- Phase 1: Sketch ---
    # For each layer, init sketch vectors
    sketch_vectors = []
    for state in hutch_states:
        sketch_vectors.append(state.init_phase1_sketch().to(device))

    model.train()
    samples_processed = 0
    for batch in dataloader:
        if samples_processed >= num_samples:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Forward + backward per batch
        for i, (name, param) in enumerate(layer_params):
            model.zero_grad()
            # One batch per layer for HVP (memory-constrained)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          labels=input_ids)
            loss = outputs.loss

            for j in range(sketch_vectors[i].shape[1]):
                vec = sketch_vectors[i][:, j:j+1].expand_as(param)
                hvp = _compute_hvp_single_vector(loss, param, vec)
                hutch_states[i].accumulate_phase1(hvp[:, 0].unsqueeze(-1))

        samples_processed += input_ids.size(0)
        if samples_processed % 64 == 0:
            print(f"  [Hestia Calib] Phase 1: {samples_processed}/{num_samples} samples")

    num_batches = max(1, samples_processed)
    for state in hutch_states:
        state.finalize_phase1(num_batches)

    # --- Phase 2: Subspace ---
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if samples_processed >= num_samples * 2:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            for i, (name, param) in enumerate(layer_params):
                if hutch_states[i].final_trace > 0:
                    continue
                model.zero_grad()
                Q_device = hutch_states[i].init_phase2_subspace().to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                              labels=input_ids)
                loss = outputs.loss
                for j in range(Q_device.shape[1]):
                    vec = Q_device[:, j:j+1].expand_as(param)
                    hvp = _compute_hvp_single_vector(loss, param, vec)
                    hutch_states[i].accumulate_phase2(hvp[:, 0].unsqueeze(-1))
            samples_processed += input_ids.size(0)

    for state in hutch_states:
        state.finalize_phase2(num_batches)

    # --- Phase 3: Residual (skip if already done) ---
    with torch.no_grad():
        for batch in dataloader:
            if samples_processed >= num_samples * 3:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            all_done = True
            for i, (name, param) in enumerate(layer_params):
                if hutch_states[i].final_trace > 0:
                    continue
                all_done = False
                model.zero_grad()
                omega_perp = hutch_states[i].init_phase3_residual().to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                              labels=input_ids)
                loss = outputs.loss
                for j in range(omega_perp.shape[1]):
                    vec = omega_perp[:, j:j+1].expand_as(param)
                    hvp = _compute_hvp_single_vector(loss, param, vec)
                    hutch_states[i].accumulate_phase3(hvp[:, 0].unsqueeze(-1))
            if all_done:
                break
            samples_processed += input_ids.size(0)

    for state in hutch_states:
        if state.final_trace == 0:
            state.finalize_phase3(num_batches)

    # --- Compute sensitivity scores and temperature scales ---
    traces = [state.final_trace for state in hutch_states]
    log_traces = [math.log(max(t, 1e-10)) for t in traces]

    mean_log = sum(log_traces) / len(log_traces)
    std_log = (sum((lt - mean_log) ** 2 for lt in log_traces) / len(log_traces)) ** 0.5
    std_log = max(std_log, 1e-8)

    temp_scales = {}
    for i, (name, _) in enumerate(layer_params):
        layer_id = f"layer_{i}"
        # Standardized sigmoid -> [0, 1], then scale by alpha
        z = alpha * (log_traces[i] - mean_log) / std_log
        s_i = 1.0 / (1.0 + math.exp(-z))
        # temp_scale = exp(beta * s_i); higher sensitivity -> higher temp -> slower cooling
        temp_scale = math.exp(s_i)
        temp_scales[layer_id] = temp_scale

    print(f"[Hestia Calib] Done. Trace range: [{min(traces):.2e}, {max(traces):.2e}], "
          f"temp_scale range: [{min(temp_scales.values()):.3f}, {max(temp_scales.values()):.3f}]")
    return temp_scales
