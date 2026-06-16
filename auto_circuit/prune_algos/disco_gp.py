"""DiscoGP sheaf pruning algorithm for AutoCircuit.

Implements joint weight + edge pruning via Gumbel-sigmoid masks, as described in:
  Yu et al., "Sheaf Discovery with Joint Computation Graph Pruning and Flexible
  Granularity" (EMNLP 2025) — https://arxiv.org/abs/2407.03779

Key differences from Subnetwork Probing / Edge Pruning in this library:
  - Uses Gumbel-sigmoid straight-through estimator (not Hard Concrete)
  - Joint two-phase optimisation: weight masks first, then edge masks
  - Two faith / complete loss modes (``loss_mode``):
      * ``argmax_ce`` (default): full-vocab CE against the clean model's argmax;
        completeness = CE against uniform over the full vocab.
      * ``answer_margin_ce`` (paper-faithful): 2-class CE on
        (mean correct logit, mean wrong logit) from ``batch.answers`` /
        ``batch.wrong_answers``; completeness uses the same pair vs. uniform 0.5.
  - Zero ablation for edge patching
  - Original model weights are RESTORED after the algorithm returns — no in-place mutation
"""

import gc
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple

import plotly.graph_objects as go
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from auto_circuit.data import BatchKey, PromptDataLoader
from auto_circuit.types import AblationType, Edge, PruneScores
from auto_circuit.utils.ablation_activations import (
    BATCH_SPECIFIC_ABLATIONS,
    batch_src_ablations,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import patch_mode
from auto_circuit.utils.gumbel_sigmoid_utils import gumbel_sigmoid, schedule_epoch_lambda
from auto_circuit.utils.patch_wrapper import PatchWrapperImpl
from auto_circuit.utils.patchable_model import PatchableModel

DISCO_GP = "DiscoGP"

# Names of parameter-name substrings that should NOT get weight masks
_WEIGHT_MASK_SKIP = ("emb", "ln", "edge", "unembed")


def _is_maskable_weight(name: str) -> bool:
    """Return True for linear layer weights that DiscoGP masks."""
    return not any(s in name for s in _WEIGHT_MASK_SKIP)


def disco_gp_prune_scores(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    official_edges: Optional[Set[Edge]],
    # ---- Weight mask hparams ----
    use_weight_masks: bool = True,
    gs_temp_weight: float = 0.01,
    logits_w_init: float = 1.0,
    w_lr: float = 0.1,
    w_lambda_sparse_init: float = 1.0,
    w_lambda_complete_init: float = 0.0,
    w_max_times_lambda_sparse: float = 1000.0,
    w_min_times_lambda_sparse: float = 1.0,
    w_train_epochs: int = 500,
    w_n_epoch_warmup: int = 500,
    w_n_epoch_cooldown: int = 1,
    # ---- Edge mask hparams ----
    use_edge_masks: bool = True,
    gs_temp_edge: float = 1.0,
    logits_e_init: float = 1.0,
    e_lr: float = 0.1,
    e_lambda_sparse_init: float = 1.0,
    e_lambda_complete_init: float = 1.0,
    e_max_times_lambda_sparse: float = 100.0,
    e_min_times_lambda_sparse: float = 0.01,
    e_train_epochs: int = 100,
    e_n_epoch_warmup: int = 20,
    e_n_epoch_cooldown: int = 20,
    # ---- Misc ----
    show_train_graph: bool = False,
    hard: bool = True,
    loss_mode: Literal["argmax_ce", "answer_margin_ce"] = "answer_margin_ce",
    # ---- Multi-fidelity HPO hooks (no-op when None) ----
    eval_callback: Optional[Callable[[int, "PruneScores"], bool]] = None,
    eval_every: Optional[int] = None,
    # ---- Discovery ablation ----
    ablation_type: AblationType = AblationType.ZERO,
) -> PruneScores:
    """Discover a sparse circuit via joint weight and edge pruning with Gumbel-sigmoid masks.

    Implements the DiscoGP algorithm (Yu et al., 2025).  The function performs two
    sequential optimisation phases:

    **Phase 1 — Weight Pruning** (if ``use_weight_masks=True``):
      Creates per-element learnable logits for all linear-layer weights (excluding
      embeddings, layer-norms, and bias terms).  Optimises them to minimise
      ``faith_loss + lambda_sparse * sparsity`` (loss form picked by ``loss_mode``).
      After training, weight masks are frozen deterministically (``logit > 0``).

    **Phase 2 — Edge Pruning** (if ``use_edge_masks=True``):
      With weight masks baked in, optimises per-edge logits on the AutoCircuit
      ``PatchWrapperImpl`` modules using the same faithfulness loss plus a
      completeness term applied with *reversed* masks.

    The original model weights are **restored** before the function returns, so the
    caller's model is not modified.

    Args:
        model: The patchable model to find a circuit for.
        dataloader: Provides ``(clean, corrupt)`` input pairs.
        official_edges: Unused (present for AutoCircuit API compatibility).
        use_weight_masks: Whether to run Phase 1 weight optimisation.
        gs_temp_weight: Gumbel-sigmoid temperature. Lower values (e.g., 0.01) approximate hard binary masks.
        logits_w_init: Initial value for weight mask logits.
        w_lr: AdamW learning rate for weight mask logits.
        w_lambda_sparse_init: Base sparsity coefficient. Drives the network to prune weights.
        w_lambda_complete_init: Base completeness coefficient. Forces the pruned weights to be uninformative (0 disables).
        w_max_times_lambda_sparse: Peak sparsity multiplier reached at end of warmup (creates strong pressure to prune).
        w_min_times_lambda_sparse: Final sparsity multiplier after cooldown (relaxes pressure for fine-tuning).
        w_train_epochs: Total epochs for Phase 1.
        w_n_epoch_warmup: Epochs to ramp up the sparsity penalty multiplier to max_times.
        w_n_epoch_cooldown: Epochs to decay the sparsity penalty multiplier down to min_times.
        use_edge_masks: Whether to run Phase 2 edge optimisation.
        gs_temp_edge: Gumbel-sigmoid temperature. Typically 1.0 for softer edge gating.
        logits_e_init: Initial value for edge mask logits.
        e_lr: AdamW learning rate for edge mask logits.
        e_lambda_sparse_init: Base sparsity coefficient for edges.
        e_lambda_complete_init: Base completeness coefficient for edges.
        e_max_times_lambda_sparse: Peak sparsity multiplier for edges.
        e_min_times_lambda_sparse: Final sparsity multiplier for edges.
        e_train_epochs: Total epochs for Phase 2.
        e_n_epoch_warmup: Epochs to ramp up the edge sparsity penalty.
        e_n_epoch_cooldown: Epochs to decay the edge sparsity penalty.
        show_train_graph: Save training curves to HTML files.
        hard: Whether to use hard (binary STE) or soft Gumbel-sigmoid sampling.
        loss_mode: Faithfulness / completeness loss formulation. ``"argmax_ce"``
            uses full-vocab CE vs. the unmasked model's argmax (and uniform 1/V
            for completeness). ``"answer_margin_ce"`` uses 2-class CE on
            (mean correct logit, mean wrong logit) from ``batch.answers`` /
            ``batch.wrong_answers`` (paper-faithful for IOI; aligns with the
            standard logit-diff metric for Greater-Than / Docstring).

    Returns:
        :class:`~auto_circuit.types.PruneScores` mapping destination-wrapper module
        names to edge importance tensors.  Values are continuous ``sigmoid(logit)``
        scores in ``(0, 1)`` — higher means more important (edge should be kept).
    """
    assert len(dataloader) > 0, "Dataloader is empty"
    device = next(model.parameters()).device
    out_slice = model.out_slice

    # =========================================================
    # Step 1: Cache clean logits and compute target argmax
    # =========================================================
    # Only needed for loss_mode="argmax_ce". The answer-margin-CE mode reads
    # supervision directly from batch.answers / batch.wrong_answers each step.
    clean_targets: Dict[BatchKey, t.Tensor] = {}
    if loss_mode == "argmax_ce":
        with t.no_grad():
            for batch in dataloader:
                logits = model(batch.clean)[out_slice]
                clean_targets[batch.key] = logits.argmax(dim=-1).clone()

    # =========================================================
    # Step 2: Snapshot original model weights for later restoration
    # =========================================================
    original_state: Dict[str, t.Tensor] = {}
    for name, param in model.named_parameters():
        if _is_maskable_weight(name):
            original_state[name] = param.data.clone()

    # =========================================================
    # Phase 1: Weight mask optimisation
    # =========================================================
    weight_mask_logits: Dict[str, nn.Parameter] = {}
    w_losses: List[float] = []
    w_faith_hist: List[float] = []
    w_sparse_hist: List[float] = []

    if use_weight_masks:
        # Create per-element logit parameters for each maskable weight
        N_weight = 0
        for name, param in model.named_parameters():
            if _is_maskable_weight(name):
                param.requires_grad_(False)
                logit = nn.Parameter(
                    t.full_like(param, logits_w_init) + 0.01 * t.randn_like(param)
                )
                weight_mask_logits[name] = logit
                N_weight += param.numel()

        w_optimizer = AdamW(list(weight_mask_logits.values()), lr=w_lr)

        model.eval()
        for epoch in (w_pbar := tqdm(range(w_train_epochs))):
            lambda_sparse = schedule_epoch_lambda(
                epoch, w_lambda_sparse_init,
                max_times=w_max_times_lambda_sparse,
                min_times=w_min_times_lambda_sparse,
                n_epoch_warmup=w_n_epoch_warmup,
                n_epoch_cooldown=w_n_epoch_cooldown,
            )
            lambda_complete = schedule_epoch_lambda(epoch, w_lambda_complete_init)
            if epoch > 0:
                w_pbar.set_description_str(
                    f"{DISCO_GP} W Epoch {epoch} "
                    f"loss={w_losses[-1]:.4f} faith={w_faith_hist[-1]:.4f} "
                    f"sparse={w_sparse_hist[-1]:.4f} λ_s={lambda_sparse:.2f}"
                )

            for batch in dataloader:
                # --- Sample binary weight masks and apply in-place ---
                sampled_masks: Dict[str, t.Tensor] = {}
                for name, logit in weight_mask_logits.items():
                    sampled_masks[name] = gumbel_sigmoid(logit, gs_temp=gs_temp_weight, hard=hard)

                for name, param in model.named_parameters():
                    if name in weight_mask_logits:
                        param.copy_(sampled_masks[name] * original_state[name])

                # Faithfulness forward pass (with masked weights, no edge masking)
                train_logits = model(batch.clean)[out_slice]
                faith_loss = _faith_loss_dispatch(
                    train_logits, batch, clean_targets, loss_mode
                )

                # Sparsity: sigmoid(logit).mean() averaged over all weight elements
                sparse_loss = t.cat(
                    [t.sigmoid(l).flatten() for l in weight_mask_logits.values()]
                ).sum() / max(N_weight, 1)

                loss = faith_loss + lambda_sparse * sparse_loss

                loss.backward()
                w_optimizer.step()
                w_optimizer.zero_grad()

                # Restore original weights (matching original turn_off_weight_masks)
                with t.no_grad():
                    for name, param in model.named_parameters():
                        if name in original_state:
                            param.copy_(original_state[name])
                            param.detach_()

                # --- Completeness backward (reversed masks, separate step) ---
                if lambda_complete > 0:
                    rev_sampled: Dict[str, t.Tensor] = {}
                    for name, logit in weight_mask_logits.items():
                        rev_sampled[name] = 1.0 - gumbel_sigmoid(
                            logit, gs_temp=gs_temp_weight, hard=hard
                        )

                    for name, param in model.named_parameters():
                        if name in weight_mask_logits:
                            param.copy_(rev_sampled[name] * original_state[name])

                    rev_logits = model(batch.clean)[out_slice]
                    complete_loss = _complete_loss_dispatch(
                        rev_logits, batch, loss_mode
                    )
                    (complete_loss * lambda_complete).backward()
                    w_optimizer.step()
                    w_optimizer.zero_grad()

                    # Restore again
                    with t.no_grad():
                        for name, param in model.named_parameters():
                            if name in original_state:
                                param.copy_(original_state[name])
                                param.detach_()

                w_losses.append(loss.item())
                w_faith_hist.append(faith_loss.item())
                w_sparse_hist.append(sparse_loss.item())

        # Bake deterministic weight masks for Phase 2
        with t.no_grad():
            for name, param in model.named_parameters():
                if name in weight_mask_logits:
                    det_mask = (weight_mask_logits[name] > 0).float()
                    param.copy_(det_mask * original_state[name])
                    param.detach_()

        del w_optimizer
        gc.collect()

    if show_train_graph and w_losses:
        _save_train_graph(
            {"Total Loss": w_losses, "Faithfulness": w_faith_hist, "Sparsity": w_sparse_hist},
            title="DiscoGP Weight Pruning",
            filename="disco_gp_weight_train_graph.html",
        )

    # =========================================================
    # Phase 2: Edge mask optimisation
    # =========================================================
    # batch_src_ablations requires clean_corrupt set iff the ablation is batch-specific
    # (per-batch corrupt activations); non-batch ablations (zero, dataset-mean) take None.
    clean_corrupt = "corrupt" if ablation_type in BATCH_SPECIFIC_ABLATIONS else None
    src_outs: Dict[BatchKey, t.Tensor] = batch_src_ablations(
        model, dataloader, ablation_type=ablation_type, clean_corrupt=clean_corrupt
    )

    dest_wrapper_dict: Dict[str, PatchWrapperImpl] = {
        w.module_name: w for w in model.dest_wrappers
    }

    edge_logits: Dict[str, nn.Parameter] = {}
    N_edge = 0
    for mod_name, wrapper in dest_wrapper_dict.items():
        logit = nn.Parameter(
            t.full_like(wrapper.patch_mask, logits_e_init)
            + 0.01 * t.randn_like(wrapper.patch_mask)
        )
        edge_logits[mod_name] = logit
        N_edge += wrapper.patch_mask.numel()

    e_losses: List[float] = []
    e_faith_hist: List[float] = []
    e_sparse_hist: List[float] = []
    e_complete_hist: List[float] = []

    if use_edge_masks:
        e_optimizer = AdamW(list(edge_logits.values()), lr=e_lr)

        model.eval()
        for epoch in (e_pbar := tqdm(range(e_train_epochs))):
            lambda_sparse = schedule_epoch_lambda(
                epoch, e_lambda_sparse_init,
                max_times=e_max_times_lambda_sparse,
                min_times=e_min_times_lambda_sparse,
                n_epoch_warmup=e_n_epoch_warmup,
                n_epoch_cooldown=e_n_epoch_cooldown,
            )
            lambda_complete = schedule_epoch_lambda(epoch, e_lambda_complete_init)
            if epoch > 0:
                e_pbar.set_description_str(
                    f"{DISCO_GP} E Epoch {epoch} "
                    f"loss={e_losses[-1]:.4f} faith={e_faith_hist[-1]:.4f} "
                    f"sparse={e_sparse_hist[-1]:.4f} "
                    f"λ_s={lambda_sparse:.3f} λ_c={lambda_complete:.3f}"
                )

            for batch in dataloader:
                patch_outs = src_outs[batch.key].clone().detach()

                # --- Faithfulness pass: sample binary edge masks (mask=1 → ablated) ---
                # AutoCircuit convention: patch_mask=1 patches the edge (replaces with
                # ablated activation), patch_mask=0 keeps the original activation.
                # DiscoGP convention: mask=1 keeps, mask=0 ablates.
                # We negate: AC_mask = 1 - DiscoGP_mask, so ablated edges have AC_mask=1.
                faith_masks = _sample_edge_masks(
                    edge_logits, gs_temp_edge, reverse=False, hard=hard
                )
                train_logits = _forward_with_masks(
                    model, batch.clean, patch_outs,
                    faith_masks, dest_wrapper_dict, out_slice
                )
                faith_loss = _faith_loss_dispatch(
                    train_logits, batch, clean_targets, loss_mode
                )

                # Sparsity
                sparse_loss = t.cat(
                    [t.sigmoid(l).flatten() for l in edge_logits.values()]
                ).sum() / max(N_edge, 1)

                # --- Completeness pass (reversed masks) ---
                complete_loss_val = t.tensor(0.0, device=device)
                if lambda_complete > 0:
                    rev_masks = _sample_edge_masks(
                        edge_logits, gs_temp_edge, reverse=True, hard=hard
                    )
                    rev_logits = _forward_with_masks(
                        model, batch.clean, patch_outs,
                        rev_masks, dest_wrapper_dict, out_slice
                    )
                    complete_loss_val = _complete_loss_dispatch(
                        rev_logits, batch, loss_mode
                    )

                loss = (
                    faith_loss
                    + lambda_sparse * sparse_loss
                    + lambda_complete * complete_loss_val
                )

                e_optimizer.zero_grad()
                loss.backward()
                e_optimizer.step()

                e_losses.append(loss.item())
                e_faith_hist.append(faith_loss.item())
                e_sparse_hist.append(sparse_loss.item())
                e_complete_hist.append(complete_loss_val.item())

            # Multi-fidelity callback: snapshot current scores, report to Optuna.
            # Returning True from the callback means the trial should be pruned.
            if (
                eval_callback is not None
                and eval_every is not None
                and (epoch + 1) % eval_every == 0
            ):
                with t.no_grad():
                    snapshot: PruneScores = {
                        mod: t.sigmoid(logit).detach().clone()
                        for mod, logit in edge_logits.items()
                    }
                if eval_callback(epoch, snapshot):
                    break

        del e_optimizer
        gc.collect()

    if show_train_graph and e_losses:
        _save_train_graph(
            {
                "Total Loss": e_losses,
                "Faithfulness": e_faith_hist,
                "Sparsity": e_sparse_hist,
                "Completeness": e_complete_hist,
            },
            title="DiscoGP Edge Pruning",
            filename="disco_gp_edge_train_graph.html",
        )

    # =========================================================
    # Step 3: Restore original weights (no in-place mutation)
    # =========================================================
    with t.no_grad():
        for name, param in model.named_parameters():
            if name in original_state:
                param.copy_(original_state[name])
                param.detach_()
                param.requires_grad_(False)

    del original_state
    if use_weight_masks:
        del weight_mask_logits
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()

    # =========================================================
    # Step 4: Extract continuous PruneScores from edge logits
    # =========================================================
    # Return sigmoid(logit): high value = edge likely kept = important.
    # AutoCircuit threshold-based evaluation uses abs(score) >= threshold,
    # so scores near 1.0 will be included first (most important edges).
    prune_scores: PruneScores = {}
    with t.no_grad():
        for mod_name, logit in edge_logits.items():
            prune_scores[mod_name] = t.sigmoid(logit).detach().clone()

    return prune_scores


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _faith_loss(train_logits: t.Tensor, clean_targets: t.Tensor) -> t.Tensor:
    """Cross-entropy of masked model output against clean model's argmax prediction.

    Args:
        train_logits: Masked model output ``(B, vocab)`` or ``(B, seq, vocab)``.
        clean_targets: Integer class indices from ``clean_logits.argmax(-1)``.
    """
    return F.cross_entropy(
        train_logits.reshape(-1, train_logits.shape[-1]),
        clean_targets.reshape(-1),
    )


def _gather_mean(logits: t.Tensor, idxs) -> t.Tensor:
    """Per-prompt mean of ``logits`` gathered at ``idxs``.

    Mirrors the indexing pattern used by
    :func:`auto_circuit.utils.tensor_ops.batch_answer_diffs`, which supports both
    fixed-width ``(B, n)`` tensors and per-prompt variable-length lists.

    Args:
        logits: ``(B, V)`` logits after ``out_slice``.
        idxs: Either a ``(B, n)`` tensor or a list of 1D tensors (variable ``n``).
    Returns:
        ``(B,)`` — per-prompt mean logit across the indexed set.
    """
    if isinstance(idxs, t.Tensor):
        return t.gather(logits, -1, idxs).mean(-1)
    return t.stack([logits[i].index_select(-1, a).mean() for i, a in enumerate(idxs)])


def _faith_loss_margin(logits: t.Tensor, batch) -> t.Tensor:
    """Paper-faithful 2-class CE: ``softplus(mean_wrong - mean_correct).mean()``.

    Pulls correct-answer logits above wrong-answer logits at the answer position,
    averaging over any multi-token answer / distractor sets.
    """
    pair = t.stack(
        [_gather_mean(logits, batch.answers),
         _gather_mean(logits, batch.wrong_answers)],
        -1,
    )
    target = t.zeros(pair.size(0), dtype=t.long, device=pair.device)
    return F.cross_entropy(pair, target)


def _complete_loss_margin(logits: t.Tensor, batch) -> t.Tensor:
    """Paper-faithful 2-class CE against uniform 0.5/0.5 on (correct, wrong).

    Forces the (mean_correct, mean_wrong) margin to zero under reversed masks
    — i.e. destroys the binary preference without flattening the entire vocab.
    """
    pair = t.stack(
        [_gather_mean(logits, batch.answers),
         _gather_mean(logits, batch.wrong_answers)],
        -1,
    )
    return F.cross_entropy(pair, t.full_like(pair, 0.5))


def _faith_loss_dispatch(
    logits: t.Tensor,
    batch,
    clean_targets: Dict[BatchKey, t.Tensor],
    loss_mode: str,
) -> t.Tensor:
    if loss_mode == "answer_margin_ce":
        return _faith_loss_margin(logits, batch)
    return _faith_loss(logits, clean_targets[batch.key])


def _complete_loss_dispatch(
    rev_logits: t.Tensor,
    batch,
    loss_mode: str,
) -> t.Tensor:
    if loss_mode == "answer_margin_ce":
        return _complete_loss_margin(rev_logits, batch)
    n_classes = rev_logits.shape[-1]
    uniform_target = t.full_like(rev_logits, 1.0 / n_classes)
    return F.cross_entropy(
        rev_logits.reshape(-1, n_classes),
        uniform_target.reshape(-1, n_classes),
    )


def _sample_edge_masks(
    edge_logits: Dict[str, nn.Parameter],
    gs_temp: float,
    reverse: bool,
    hard: bool,
) -> Dict[str, t.Tensor]:
    """Sample edge masks and convert to AutoCircuit convention.

    DiscoGP: mask=1 means keep.  AutoCircuit: mask=1 means ablate.
    So: ``AC_mask = 1 - disco_mask``.
    When ``reverse=True``: ``AC_mask = disco_mask`` (kept edges get ablated).
    """
    masks: Dict[str, t.Tensor] = {}
    for mod_name, logit in edge_logits.items():
        disco_mask = gumbel_sigmoid(logit, gs_temp=gs_temp, hard=hard)
        if reverse:
            disco_mask = 1.0 - disco_mask
        masks[mod_name] = 1.0 - disco_mask
    return masks


def _forward_with_masks(
    model: PatchableModel,
    input_batch: t.Tensor,
    patch_outs: t.Tensor,
    masks: Dict[str, t.Tensor],
    dest_wrapper_dict: Dict[str, PatchWrapperImpl],
    out_slice: tuple,
) -> t.Tensor:
    """Run a forward pass with custom edge masks installed on the wrappers.

    Temporarily overrides ``wrapper.patch_mask`` and ``wrapper.mask_fn`` (set to
    ``None`` so the mask tensor is used directly without further transformation),
    then restores the originals immediately after the forward call.

    Args:
        model: The patchable model.
        input_batch: Clean input tokens.
        patch_outs: Pre-computed zero-ablation source activations.
        masks: Dict from module name to AC-convention mask tensors.
        dest_wrapper_dict: Dest wrapper lookup.
        out_slice: Model output slice.

    Returns:
        Output logits at ``out_slice``.
    """
    saved: Dict[str, dict] = {}
    with patch_mode(model, patch_outs):
        for mod_name, wrapper in dest_wrapper_dict.items():
            saved[mod_name] = {
                "patch_mask": wrapper.patch_mask,
                "mask_fn": wrapper.mask_fn,
            }
            del wrapper.patch_mask
            wrapper.patch_mask = masks[mod_name]  # type: ignore[assignment]
            wrapper.mask_fn = None

        out = model(input_batch)[out_slice]

        for mod_name, wrapper in dest_wrapper_dict.items():
            del wrapper.patch_mask
            wrapper.patch_mask = saved[mod_name]["patch_mask"]
            wrapper.mask_fn = saved[mod_name]["mask_fn"]

    return out


def _save_train_graph(
    series: Dict[str, List[float]],
    title: str,
    filename: str,
) -> None:
    """Write a Plotly training graph to an HTML file."""
    try:
        fig = go.Figure()
        for name, values in series.items():
            fig.add_trace(go.Scatter(y=values, name=name))
        fig.update_layout(title=title, xaxis_title="Step")
        fig.write_html(filename)
    except Exception:
        pass
