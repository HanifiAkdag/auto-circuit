"""Edge Pruning algorithm for AutoCircuit.

Implements the gradient-based L0 edge pruning method from:
    "Finding Transformer Circuits with Edge Pruning"
    (Bhaskar, Wettig, Friedman, Chen — NeurIPS 2024)
    https://arxiv.org/abs/2406.16778

This integrates the core Edge-Pruning optimisation into AutoCircuit's ``PatchableModel``
infrastructure. The algorithm discovers sparse circuits via L0-regularised masks with
Lagrangian dual optimisation (learned sparsity penalty).

Key differences from :func:`subnetwork_probing_prune_scores`:
    1. **Learned regularisation** — λ₁, λ₂ updated via gradient ascent (``maximize=True``).
    2. **Node masks** — per-source-node masks accelerate convergence to high sparsity.
    3. **Sparsity warmup** — targets linearly ramp from 0 to the final target.
    4. **Binary search discretisation** — evaluation-time thresholding via
       :func:`~auto_circuit.utils.l0_utils.deterministic_z`.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set

import plotly.graph_objects as go
import torch as t
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.optim import AdamW

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import AblationType, BatchKey, Edge, PruneScores
from auto_circuit.utils.ablation_activations import batch_src_ablations
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import patch_mode, set_all_masks
from auto_circuit.utils.l0_utils import (
    deterministic_z,
    get_linear_schedule_with_warmup,
)
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import multibatch_kl_div, sample_hard_concrete

EP = "Edge Pruning"


def edge_pruning_prune_scores(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    official_edges: Optional[Set[Edge]],
    # --- Optimisation ---
    learning_rate: float = 0.8,
    lambda_learning_rate: float = 0.8,
    steps: int = 3000,
    warmup_steps: int = 0,
    # --- Sparsity targets ---
    target_edge_sparsity: float = 1.1,
    target_node_sparsity: float = 0.72,
    start_edge_sparsity: float = 0.0,
    start_node_sparsity: float = 0.0,
    sparsity_warmup_steps: int = 2500,
    # --- Node pruning ---
    use_node_masks: bool = True,
    # --- Misc ---
    show_train_graph: bool = False,
    init_log_alpha: float = 10.0,
    disable_linear_regularization: bool = False,
) -> PruneScores:
    """Discover a circuit using L0-regularised edge pruning with Lagrangian duality.

    The method optimises continuous masks on each edge (and optionally each node) to
    find a sparse subgraph that preserves the model's output distribution on the task.
    Sparsity is controlled by learnable Lagrangian multipliers, eliminating the need
    to manually tune a fixed regularisation coefficient.

    Args:
        model: The patchable model to find the circuit for.
        dataloader: Provides ``(clean, corrupt)`` input pairs for training.
        official_edges: Not used (present for API compatibility).
        learning_rate: Learning rate for edge and node mask parameters.
        lambda_learning_rate: Learning rate for Lagrangian multiplier parameters
            (``maximize=True``).
        steps: Total number of training epochs (passes over the dataloader).
        warmup_steps: Number of steps for linear LR warmup (0 → ``learning_rate``).
        target_edge_sparsity: Target edge sparsity at end of warmup.  Values > 1.0
            are valid (pushes sparsity toward maximum without overshooting).
        target_node_sparsity: Target node sparsity at end of warmup.
        start_edge_sparsity: Edge sparsity target at training start.
        start_node_sparsity: Node sparsity target at training start.
        sparsity_warmup_steps: Steps over which sparsity targets linearly ramp.
        use_node_masks: Whether to use per-source-node masks alongside edge masks.
            This is a core part of the paper and significantly helps convergence.
        show_train_graph: Whether to display a plot of training metrics.
        init_log_alpha: Initial ``log_alpha`` magnitude. All masks start as "keep"
            (negative in AC convention → HC samples near 0 → edges kept).
        disable_linear_regularization: If ``True``, freeze ``λ₁`` terms at 0 and only
            use the quadratic ``λ₂`` terms for regularisation.

    Returns:
        :class:`PruneScores` mapping module names to edge importance tensors.
        Higher score = more important edge.
    """
    assert len(dataloader) > 0, "Dataloader is empty"
    device = next(model.parameters()).device
    out_slice = model.out_slice
    n_edges = model.n_edges
    n_srcs = len(model.srcs)

    # ============================================================
    # Step 1: Collect clean logits (full model, target for KL div)
    # ============================================================
    clean_logits: Dict[BatchKey, t.Tensor] = {}
    with t.inference_mode():
        for batch in dataloader:
            clean_logits[batch.key] = model(batch.clean)[out_slice]

    # ============================================================
    # Step 2: Compute corrupted source activations (interchange ablation)
    # ============================================================
    src_outs: Dict[BatchKey, t.Tensor] = batch_src_ablations(
        model,
        dataloader,
        ablation_type=AblationType.RESAMPLE,
        clean_corrupt="corrupt",
    )

    # ============================================================
    # Step 3: Create learnable parameters
    # ============================================================

    # --- Edge mask log_alphas (separate from patch_mask) ---
    # AutoCircuit convention: mask=0 → keep, mask=1 → ablate
    # Negative log_alpha → HC samples near 0 → edges kept at init
    edge_log_alphas: Dict[str, nn.Parameter] = {}
    for mod_name, pm in model.patch_masks.items():
        edge_log_alphas[mod_name] = nn.Parameter(
            t.full_like(pm, -init_log_alpha, device=device)
        )

    # --- Node mask log_alphas (one per source node) ---
    node_log_alphas: Dict[int, nn.Parameter] = {}
    if use_node_masks:
        for src in sorted(model.srcs, key=lambda s: s.src_idx):
            node_log_alphas[src.src_idx] = nn.Parameter(
                t.full((1,), -init_log_alpha, device=device, dtype=t.float32)
            )

    # --- Lagrangian multiplier parameters ---
    if disable_linear_regularization:
        lambda_edge_1 = t.tensor(0.0, device=device)  # frozen buffer
        lambda_node_1 = t.tensor(0.0, device=device)
    else:
        lambda_edge_1 = nn.Parameter(t.tensor(0.0, device=device))
        lambda_node_1 = nn.Parameter(t.tensor(0.0, device=device))
    lambda_edge_2 = nn.Parameter(t.tensor(0.0, device=device))
    lambda_node_2 = nn.Parameter(t.tensor(0.0, device=device))

    # ============================================================
    # Step 4: Build single optimizer with 4 param groups
    # ============================================================
    # Mirrors Edge-Pruning's get_optimizers() (fpt2_ioi.py:524-565):
    #   - Group 0: edge log_alphas  (minimize)
    #   - Group 1: edge lambdas     (maximize)
    #   - Group 2: node log_alphas  (minimize)
    #   - Group 3: node lambdas     (maximize)
    param_groups: List[dict] = [
        {"params": list(edge_log_alphas.values()), "lr": learning_rate},
    ]
    lambda_params_edge = [p for p in [lambda_edge_1, lambda_edge_2] if isinstance(p, nn.Parameter)]
    if lambda_params_edge:
        param_groups.append(
            {"params": lambda_params_edge, "lr": lambda_learning_rate, "maximize": True}
        )

    if use_node_masks:
        param_groups.append(
            {"params": list(node_log_alphas.values()), "lr": learning_rate}
        )
        lambda_params_node = [p for p in [lambda_node_1, lambda_node_2] if isinstance(p, nn.Parameter)]
        if lambda_params_node:
            param_groups.append(
                {"params": lambda_params_node, "lr": lambda_learning_rate, "maximize": True}
            )

    optimizer = AdamW(param_groups, lr=learning_rate)
    total_training_steps = steps * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)

    # ============================================================
    # Step 5: Build src_idx → wrapper mapping for node mask application
    # ============================================================
    # For each dest wrapper, we need to know which global src indices are in its
    # in_srcs slice so we can look up the corresponding node masks.
    dest_wrapper_dict: Dict[str, "PatchWrapperImpl"] = {}  # noqa: F821
    for wrapper in model.dest_wrappers:
        dest_wrapper_dict[wrapper.module_name] = wrapper

    # ============================================================
    # Step 6: Training loop
    # ============================================================
    losses_hist: List[float] = []
    kl_hist: List[float] = []
    edge_sparsity_hist: List[float] = []
    node_sparsity_hist: List[float] = []
    edge_reg_hist: List[float] = []
    node_reg_hist: List[float] = []

    model.eval()
    for step in (step_pbar := tqdm(range(steps))):
        # --- Sparsity warmup: linearly ramp targets ---
        warmup_frac = min(step / max(sparsity_warmup_steps, 1), 1.0)
        t_edge = start_edge_sparsity + (target_edge_sparsity - start_edge_sparsity) * warmup_frac
        t_node = start_node_sparsity + (target_node_sparsity - start_node_sparsity) * warmup_frac

        desc_parts = []
        if losses_hist:
            desc_parts.append(f"Loss: {losses_hist[-1]:.4f}")
            desc_parts.append(f"KL: {kl_hist[-1]:.4f}")
            desc_parts.append(f"EdgeSp: {edge_sparsity_hist[-1]:.3f}")
            if use_node_masks:
                desc_parts.append(f"NodeSp: {node_sparsity_hist[-1]:.3f}")
        step_pbar.set_description_str(f"{EP} Step {step} " + ", ".join(desc_parts))

        for batch in dataloader:
            batch_size = batch.clean.shape[0]
            patch_outs = src_outs[batch.key].clone().detach()

            # --- Sample edge masks (with gradient tracking) ---
            sampled_edge_z: Dict[str, t.Tensor] = {}
            all_edge_z_flat: List[t.Tensor] = []
            for mod_name, ela in edge_log_alphas.items():
                z = sample_hard_concrete(ela, batch_size)  # (batch, *mask_shape)
                sampled_edge_z[mod_name] = z
                all_edge_z_flat.append(z.mean(dim=0).flatten())  # avg over batch for reg

            # --- Sample node masks (with gradient tracking) ---
            sampled_node_z: Dict[int, t.Tensor] = {}
            all_node_z_flat: List[t.Tensor] = []
            if use_node_masks:
                for src_idx, nla in node_log_alphas.items():
                    z = sample_hard_concrete(nla, batch_size)  # (batch, 1)
                    sampled_node_z[src_idx] = z
                    all_node_z_flat.append(z.mean(dim=0).flatten())

            # --- Compute effective masks (combine edge + node) ---
            effective_masks: Dict[str, t.Tensor] = {}
            for mod_name, wrapper in dest_wrapper_dict.items():
                m_edge = sampled_edge_z[mod_name]  # (batch, *mask_shape)

                if use_node_masks and wrapper.in_srcs is not None:
                    in_srcs = wrapper.in_srcs
                    # Build node mask tensor for this wrapper's sources
                    node_z_parts: List[t.Tensor] = []
                    for global_idx in range(in_srcs.start, in_srcs.stop):
                        if global_idx in sampled_node_z:
                            node_z_parts.append(sampled_node_z[global_idx])
                        else:
                            node_z_parts.append(
                                t.zeros(batch_size, 1, device=device)
                            )
                    # (batch, n_local_srcs)
                    node_z = t.cat(node_z_parts, dim=-1)

                    # Broadcast: add dims between batch and src to match m_edge shape
                    # m_edge: (batch, [seq], [head], src)  →  node_z: (batch, 1, ..., 1, src)
                    n_extra_dims = m_edge.ndim - node_z.ndim
                    for _ in range(n_extra_dims):
                        node_z = node_z.unsqueeze(1)

                    # Combine: effective ablation = 1 - (1-edge)*(1-node)
                    # In AC convention: 0 = keep, 1 = ablate
                    effective = 1.0 - (1.0 - m_edge) * (1.0 - node_z)
                else:
                    effective = m_edge

                effective_masks[mod_name] = effective

            # --- Apply effective masks and run forward ---
            # We temporarily override the wrapper's mask state to use our
            # pre-computed effective masks (with full gradient graph).
            # Use mask_fn=None (direct application) and set batch_size so
            # the einsum includes the batch dimension.
            orig_state: Dict[str, dict] = {}
            with patch_mode(model, patch_outs):
                for mod_name, wrapper in dest_wrapper_dict.items():
                    orig_state[mod_name] = {
                        "patch_mask": wrapper.patch_mask,
                        "mask_fn": wrapper.mask_fn,
                        "batch_size": wrapper.batch_size,
                    }
                    wrapper.patch_mask = effective_masks[mod_name]
                    wrapper.mask_fn = None
                    wrapper.batch_size = batch_size

                train_logits = model(batch.clean)[out_slice]

                # Restore original state immediately
                for mod_name, wrapper in dest_wrapper_dict.items():
                    wrapper.patch_mask = orig_state[mod_name]["patch_mask"]
                    wrapper.mask_fn = orig_state[mod_name]["mask_fn"]
                    wrapper.batch_size = orig_state[mod_name]["batch_size"]

            # --- KL divergence loss ---
            kl_loss = multibatch_kl_div(
                log_softmax(train_logits, dim=-1),
                log_softmax(clean_logits[batch.key], dim=-1),
            )

            # --- Edge sparsity & regularisation ---
            # Sparsity from average sampled mask values (matching EP implementation:
            # modeling_fpt2.py:1377 uses sum of sampled z / n_edges)
            all_edge_z = t.cat(all_edge_z_flat)
            edge_sparsity = all_edge_z.sum() / n_edges
            edge_reg = lambda_edge_1 * (edge_sparsity - t_edge) + lambda_edge_2 * (
                edge_sparsity - t_edge
            ) ** 2

            # --- Node sparsity & regularisation ---
            node_reg = t.tensor(0.0, device=device)
            node_sparsity_val = 0.0
            if use_node_masks:
                all_node_z = t.cat(all_node_z_flat)
                node_sparsity = all_node_z.sum() / n_srcs
                node_sparsity_val = node_sparsity.item()
                node_reg = lambda_node_1 * (node_sparsity - t_node) + lambda_node_2 * (
                    node_sparsity - t_node
                ) ** 2

            # --- Total loss ---
            loss = kl_loss + edge_reg + node_reg

            # --- Optimiser step ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # --- Logging ---
            losses_hist.append(loss.item())
            kl_hist.append(kl_loss.item())
            edge_sparsity_hist.append(edge_sparsity.item())
            node_sparsity_hist.append(node_sparsity_val)
            edge_reg_hist.append(edge_reg.item())
            node_reg_hist.append(node_reg.item() if isinstance(node_reg, t.Tensor) else 0.0)

    # ============================================================
    # Step 7: Show training graph (optional)
    # ============================================================
    if show_train_graph:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=losses_hist, name="Total Loss"))
        fig.add_trace(go.Scatter(y=kl_hist, name="KL Divergence"))
        fig.add_trace(go.Scatter(y=edge_sparsity_hist, name="Edge Sparsity"))
        fig.add_trace(go.Scatter(y=edge_reg_hist, name="Edge Regularisation"))
        if use_node_masks:
            fig.add_trace(go.Scatter(y=node_sparsity_hist, name="Node Sparsity"))
            fig.add_trace(go.Scatter(y=node_reg_hist, name="Node Regularisation"))
        fig.update_layout(title="Edge Pruning Training", xaxis_title="Step")
        fig.show()

    # ============================================================
    # Step 8: Extract PruneScores via deterministic discretisation
    # ============================================================
    # Use deterministic_z (binary search thresholding) from the paper.
    # In AC convention: high mask value = edge ablated = UNimportant.
    # PruneScores convention: high score = important.
    # So: score = 1 - effective_ablation_mask
    prune_scores: PruneScores = {}
    for mod_name, ela in edge_log_alphas.items():
        edge_z = deterministic_z(ela)  # AC convention: high = ablated

        if use_node_masks:
            wrapper = dest_wrapper_dict[mod_name]
            if wrapper.in_srcs is not None:
                in_srcs = wrapper.in_srcs
                # Build deterministic node mask and combine with edge mask
                for i, global_idx in enumerate(range(in_srcs.start, in_srcs.stop)):
                    if global_idx in node_log_alphas:
                        nz = deterministic_z(node_log_alphas[global_idx]).item()
                        # Combine: effective_ablation = 1 - (1-edge)*(1-node)
                        # Index [..., i] selects source i across all other dims
                        edge_z[..., i] = 1.0 - (1.0 - edge_z[..., i]) * (1.0 - nz)

        # score = 1.0 - ablation_value: kept edges (low z) get high scores
        prune_scores[mod_name] = (1.0 - edge_z).detach().clone()

    return prune_scores
