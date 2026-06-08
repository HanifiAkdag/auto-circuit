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
    3. **Sparsity warmup** — targets linearly ramp from 0 to the final target over global
       optimizer steps (one per batch), matching the original EP trainer.
    4. **Continuous score extraction** — final scores use the soft sigmoid mask
       ``σ(log_alpha / T * F)`` rather than the binary-search-thresholded
       ``deterministic_z``. This gives a smooth ranking that AutoCircuit's downstream
       threshold sweep can act on directly.

Known structural difference from the original EP implementation:
    In the original, edge and node masks compose *sequentially* through the model's
    forward pass (read → attention/MLP computation → write). Here they are
    pre-composed into a single effective mask before patching, which is a
    legitimate simplification given AutoCircuit's wrapper-based patching mechanism,
    but may produce slightly different gradient dynamics.
"""
from __future__ import annotations

import math
from itertools import cycle
from typing import Callable, Dict, List, Optional, Set

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
    FACTOR,
    TEMPERATURE,
    get_linear_schedule_with_warmup,
)
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import multibatch_kl_div, sample_hard_concrete
from auto_circuit.utils.patch_wrapper import PatchWrapperImpl

EP = "Edge Pruning"


def _soft_mask(log_alpha: t.Tensor) -> t.Tensor:
    """Ablation probability for an edge: σ(log_alpha / T * F) ∈ (0, 1).

    High value → edge tends to be ablated (low importance).
    Used only for score extraction; training uses stochastic HC sampling.
    """
    return t.sigmoid(log_alpha / TEMPERATURE * FACTOR)


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
    disable_node_loss: bool = False,
    # --- Misc ---
    show_train_graph: bool = False,
    init_log_alpha: float = 10.0,
    disable_linear_regularization: bool = False,
    # --- Multi-fidelity HPO hooks (no-op when None) ---
    eval_callback: Optional[Callable[[int, "PruneScores"], bool]] = None,
    eval_every: Optional[int] = None,
    # --- Discovery ablation ---
    ablation_type: AblationType = AblationType.RESAMPLE,
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
        learning_rate: Learning rate for edge and node mask log_alpha parameters.
        lambda_learning_rate: Learning rate for Lagrangian multiplier parameters
            (``maximize=True``).
        steps: Total number of optimizer steps (one step = one forward-backward pass
            over one batch). Matches the original EP trainer's ``--max_steps``.
            The dataloader is cycled as needed.
        warmup_steps: Number of *global optimiser steps* for linear LR warmup
            (0 → ``learning_rate``).  Matches HuggingFace Trainer convention.
        target_edge_sparsity: Target edge sparsity at end of sparsity warmup.
            Values > 1.0 are valid — they push toward maximum sparsity without a
            hard ceiling.
        target_node_sparsity: Target node sparsity at end of sparsity warmup.
            Only used when ``use_node_masks=True`` and ``disable_node_loss=False``.
        start_edge_sparsity: Edge sparsity target at the very first training step.
        start_node_sparsity: Node sparsity target at the very first training step.
        sparsity_warmup_steps: Number of *global optimiser steps* over which the
            sparsity targets linearly ramp from ``start_*`` to ``target_*``.
            Matches ``num_sparsity_warmup_steps`` in the original EP trainer.
        use_node_masks: Whether to create per-source-node masks alongside edge masks.
            This is a core part of the paper; it accelerates convergence to high
            sparsity.  When ``False``, only edge masks are trained.
        disable_node_loss: When ``True``, node masks are created and applied
            (they can still prune nodes implicitly via gradient from the KL loss) but
            the Lagrangian regularisation on node sparsity is disabled — the node λ
            parameters are frozen buffers at 0.  This matches the default IOI sweep in
            the original paper (``--disable_node_loss``).  Set to ``False`` to actively
            drive node sparsity toward ``target_node_sparsity``.
        show_train_graph: Whether to write a Plotly training-metrics HTML file.
        init_log_alpha: Magnitude of the initial log_alpha values.  Stored as
            ``-init_log_alpha`` so that HC samples start near 0 (edges *kept*) in
            AutoCircuit's convention (0 = keep, 1 = ablate).
        disable_linear_regularization: If ``True``, freeze the λ₁ (linear) terms at 0
            and use only the quadratic λ₂ terms.  The original paper finds the
            quadratic term alone is sufficient in some settings.

    Returns:
        :class:`PruneScores` mapping module names to edge importance tensors.
        Higher score = more important (kept) edge.  Scores are continuous values
        in (0, 1) derived from the soft sigmoid mask of the trained log_alphas,
        suitable for AutoCircuit's threshold sweep.
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
    # Step 2: Compute corrupted source activations (resample ablation)
    # ============================================================
    src_outs: Dict[BatchKey, t.Tensor] = batch_src_ablations(
        model,
        dataloader,
        ablation_type=ablation_type,
        clean_corrupt="corrupt",
    )

    # ============================================================
    # Step 3: Create learnable parameters
    # ============================================================

    # --- Edge mask log_alphas ---
    # Stored as -init_log_alpha so HC samples start near 0 (= keep in AC convention).
    edge_log_alphas: Dict[str, nn.Parameter] = {}
    for mod_name, pm in model.patch_masks.items():
        param = t.empty_like(pm, device=device)
        nn.init.normal_(param, mean=-init_log_alpha, std=0.01)
        edge_log_alphas[mod_name] = nn.Parameter(param)

    # --- Node mask log_alphas (one scalar per source node) ---
    # Created whenever use_node_masks=True, regardless of disable_node_loss.
    node_log_alphas: Dict[int, nn.Parameter] = {}
    if use_node_masks:
        for src in sorted(model.srcs, key=lambda s: s.src_idx):
            param = t.empty((1,), device=device, dtype=t.float32)
            nn.init.normal_(param, mean=-init_log_alpha, std=0.01)
            node_log_alphas[src.src_idx] = nn.Parameter(param)

    # --- Lagrangian multiplier parameters ---
    # λ₁ (linear): frozen buffer when disable_linear_regularization=True.
    # Node λs: frozen buffers when disable_node_loss=True (matches --disable_node_loss).
    if disable_linear_regularization:
        lambda_edge_1: t.Tensor = t.tensor(0.0, device=device)
    else:
        lambda_edge_1 = nn.Parameter(t.tensor(0.0, device=device))

    lambda_edge_2 = nn.Parameter(t.tensor(0.0, device=device))

    if disable_node_loss or not use_node_masks:
        lambda_node_1: t.Tensor = t.tensor(0.0, device=device)
        lambda_node_2: t.Tensor = t.tensor(0.0, device=device)
    else:
        lambda_node_1 = nn.Parameter(t.tensor(0.0, device=device))
        lambda_node_2 = nn.Parameter(t.tensor(0.0, device=device))

    # ============================================================
    # Step 4: Build optimizer with up to 4 param groups
    # ============================================================
    # Mirrors Edge-Pruning's get_optimizers() (fpt2_ioi.py:524-565):
    #   Group 0: edge log_alphas   (minimize)
    #   Group 1: edge λs           (maximize)
    #   Group 2: node log_alphas   (minimize)   — only if use_node_masks
    #   Group 3: node λs           (maximize)   — only if use_node_masks & !disable_node_loss
    param_groups: List[dict] = [
        {"params": list(edge_log_alphas.values()), "lr": learning_rate},
    ]
    lambda_params_edge = [
        p for p in [lambda_edge_1, lambda_edge_2] if isinstance(p, nn.Parameter)
    ]
    if lambda_params_edge:
        param_groups.append(
            {"params": lambda_params_edge, "lr": lambda_learning_rate, "maximize": True}
        )

    if use_node_masks:
        param_groups.append(
            {"params": list(node_log_alphas.values()), "lr": learning_rate}
        )
        lambda_params_node = [
            p for p in [lambda_node_1, lambda_node_2] if isinstance(p, nn.Parameter)
        ]
        if lambda_params_node:
            param_groups.append(
                {"params": lambda_params_node, "lr": lambda_learning_rate, "maximize": True}
            )

    optimizer = AdamW(param_groups, lr=learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, steps)

    # ============================================================
    # Step 5: Build module-name → dest-wrapper lookup
    # ============================================================
    dest_wrapper_dict: Dict[str, PatchWrapperImpl] = {
        wrapper.module_name: wrapper for wrapper in model.dest_wrappers
    }

    # ============================================================
    # Step 6: Training loop
    # ============================================================
    losses_hist: List[float] = []
    kl_hist: List[float] = []
    edge_sparsity_hist: List[float] = []
    node_sparsity_hist: List[float] = []
    edge_reg_hist: List[float] = []
    node_reg_hist: List[float] = []

    # Cycle the dataloader so we can pull exactly `steps` batches regardless of
    # dataloader length — matches the original EP trainer's step-based loop.
    batch_cycle = cycle(dataloader)

    model.eval()
    for global_step in (step_pbar := tqdm(range(steps))):
        desc_parts = []
        if losses_hist:
            desc_parts.append(f"Loss: {losses_hist[-1]:.4f}")
            desc_parts.append(f"KL: {kl_hist[-1]:.4f}")
            desc_parts.append(f"EdgeSp: {edge_sparsity_hist[-1]:.3f}")
            if use_node_masks:
                desc_parts.append(f"NodeSp: {node_sparsity_hist[-1]:.3f}")
        step_pbar.set_description_str(f"{EP} Step {global_step} " + ", ".join(desc_parts))

        # --- Sparsity warmup: ramp targets based on optimizer step count ---
        # Matches original's get_current_edge_target_sparsity(state.global_step).
        warmup_frac = min(global_step / max(sparsity_warmup_steps, 1), 1.0)
        t_edge = start_edge_sparsity + (target_edge_sparsity - start_edge_sparsity) * warmup_frac
        t_node = start_node_sparsity + (target_node_sparsity - start_node_sparsity) * warmup_frac

        batch = next(batch_cycle)
        batch_size = batch.clean.shape[0]
        patch_outs = src_outs[batch.key].clone().detach()

        # --- Sample edge masks (stochastic HC, gradient tracked) ---
        sampled_edge_z: Dict[str, t.Tensor] = {}
        all_edge_z_flat: List[t.Tensor] = []
        for mod_name, ela in edge_log_alphas.items():
            z = sample_hard_concrete(ela, batch_size)  # (batch, *mask_shape)
            sampled_edge_z[mod_name] = z
            all_edge_z_flat.append(z.mean(dim=0).flatten())  # avg over batch

        # --- Sample node masks (stochastic HC, gradient tracked) ---
        sampled_node_z: Dict[int, t.Tensor] = {}
        all_node_z_flat: List[t.Tensor] = []
        if use_node_masks:
            for src_idx, nla in node_log_alphas.items():
                z = sample_hard_concrete(nla, batch_size)  # (batch, 1)
                sampled_node_z[src_idx] = z
                all_node_z_flat.append(z.mean(dim=0).flatten())

        # --- Compose effective masks (edge ∨ node in ablation space) ---
        # AC convention: z = 0 → keep, z = 1 → ablate.
        # For an edge to be kept, BOTH its edge mask AND its node mask must be 0.
        # De Morgan dual: effective_ablation = 1 - (1-z_edge)*(1-z_node).
        effective_masks: Dict[str, t.Tensor] = {}
        for mod_name, wrapper in dest_wrapper_dict.items():
            m_edge = sampled_edge_z[mod_name]  # (batch, *mask_shape)

            if use_node_masks and wrapper.in_srcs is not None:
                in_srcs = wrapper.in_srcs
                # Stack node z values for every src this wrapper reads from.
                node_z_parts: List[t.Tensor] = [
                    sampled_node_z.get(
                        global_idx,
                        t.zeros(batch_size, 1, device=device),
                    )
                    for global_idx in range(in_srcs.start, in_srcs.stop)
                ]
                node_z = t.cat(node_z_parts, dim=-1)  # (batch, n_local_srcs)

                # Broadcast node_z to match m_edge: insert singleton dims between
                # batch dim and src dim.
                # m_edge: (batch, [seq], [head], src)
                # node_z: (batch,              src) → add extra middle dims
                for _ in range(m_edge.ndim - node_z.ndim):
                    node_z = node_z.unsqueeze(1)

                effective = 1.0 - (1.0 - m_edge) * (1.0 - node_z)
            else:
                effective = m_edge

            effective_masks[mod_name] = effective

        # --- Apply effective masks and run forward pass ---
        # Temporarily swap the wrapper's patch_mask for our pre-computed
        # differentiable mask, run the forward, then restore immediately.
        orig_state: Dict[str, dict] = {}
        with patch_mode(model, patch_outs):
            for mod_name, wrapper in dest_wrapper_dict.items():
                orig_state[mod_name] = {
                    "patch_mask": wrapper.patch_mask,
                    "mask_fn": wrapper.mask_fn,
                    "batch_size": wrapper.batch_size,
                }
                del wrapper.patch_mask
                wrapper.patch_mask = effective_masks[mod_name]  # type: ignore
                wrapper.mask_fn = None
                wrapper.batch_size = batch_size

            train_logits = model(batch.clean)[out_slice]

            for mod_name, wrapper in dest_wrapper_dict.items():
                del wrapper.patch_mask
                wrapper.patch_mask = orig_state[mod_name]["patch_mask"]
                wrapper.mask_fn = orig_state[mod_name]["mask_fn"]
                wrapper.batch_size = orig_state[mod_name]["batch_size"]

        # --- KL divergence loss ---
        kl_loss = multibatch_kl_div(
            log_softmax(train_logits, dim=-1),
            log_softmax(clean_logits[batch.key], dim=-1),
        )

        # --- Edge sparsity & regularisation ---
        # sparsity = mean sampled z (0 = keep, 1 = ablate) → equals 1 - kept_frac,
        # matching original's `model_edge_sparsity = 1 - z_sum/n_edges` (z=1→keep).
        all_edge_z = t.cat(all_edge_z_flat)
        edge_sparsity = all_edge_z.sum() / n_edges
        edge_reg = lambda_edge_1 * (edge_sparsity - t_edge) + lambda_edge_2 * (
            edge_sparsity - t_edge
        ) ** 2

        # --- Node sparsity & regularisation ---
        # node_reg stays 0 when disable_node_loss=True (frozen λ buffers).
        node_reg = t.tensor(0.0, device=device)
        node_sparsity_val = 0.0
        if use_node_masks:
            all_node_z = t.cat(all_node_z_flat)
            node_sparsity = all_node_z.sum() / n_srcs
            node_sparsity_val = node_sparsity.item()
            node_reg = lambda_node_1 * (node_sparsity - t_node) + lambda_node_2 * (
                node_sparsity - t_node
            ) ** 2

        # --- Total loss & optimiser step ---
        loss = kl_loss + edge_reg + node_reg
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
        node_reg_hist.append(
            node_reg.item() if isinstance(node_reg, t.Tensor) else 0.0
        )

        # Multi-fidelity callback: snapshot current scores and report to Optuna.
        if (
            eval_callback is not None
            and eval_every is not None
            and (global_step + 1) % eval_every == 0
        ):
            with t.no_grad():
                snap_node_z: Dict[int, float] = {}
                if use_node_masks:
                    for idx, nla in node_log_alphas.items():
                        snap_node_z[idx] = _soft_mask(nla).item()
                snapshot: "PruneScores" = {}
                for mod_name, ela in edge_log_alphas.items():
                    edge_z = _soft_mask(ela)
                    if use_node_masks:
                        wrapper = dest_wrapper_dict[mod_name]
                        if wrapper.in_srcs is not None:
                            for i, gidx in enumerate(
                                range(wrapper.in_srcs.start, wrapper.in_srcs.stop)
                            ):
                                nz = snap_node_z.get(gidx, 0.0)
                                edge_z[..., i] = 1.0 - (1.0 - edge_z[..., i]) * (1.0 - nz)
                    snapshot[mod_name] = (1.0 - edge_z).detach().clone()
            if eval_callback(global_step, snapshot):
                break

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
        fig.update_layout(title="Edge Pruning Training", xaxis_title="Global Step")
        try:
            fig.write_html("edge_pruning_train_graph.html")
        except Exception:
            pass

    # ============================================================
    # Step 8: Extract PruneScores — continuous soft mask
    # ============================================================
    # We use the continuous soft mask σ(log_alpha / T * F) rather than the
    # binary-search-thresholded deterministic_z, giving AutoCircuit's downstream
    # threshold sweep a smooth ranking to work with.
    #
    # In AC convention z=1 → ablated, so score = 1 - z gives high scores to
    # kept (important) edges, matching the PruneScores contract.
    prune_scores: PruneScores = {}

    with t.no_grad():
        # Precompute soft node ablation probabilities once.
        soft_node_z: Dict[int, float] = {}
        if use_node_masks:
            for idx, nla in node_log_alphas.items():
                soft_node_z[idx] = _soft_mask(nla).item()

        for mod_name, ela in edge_log_alphas.items():
            edge_z = _soft_mask(ela)

            if use_node_masks:
                wrapper = dest_wrapper_dict[mod_name]
                if wrapper.in_srcs is not None:
                    in_srcs = wrapper.in_srcs
                    for i, global_idx in enumerate(range(in_srcs.start, in_srcs.stop)):
                        nz = soft_node_z.get(global_idx, 0.0)
                        edge_z[..., i] = 1.0 - (1.0 - edge_z[..., i]) * (1.0 - nz)

            prune_scores[mod_name] = (1.0 - edge_z).detach().clone()

    return prune_scores
