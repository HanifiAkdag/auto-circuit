"""Gumbel-sigmoid and lambda-scheduling utilities for the DiscoGP pruning algorithm.

Ported from the original DiscoGP implementation:
  - gumbel_sigmoid: Yu et al. (2024/2025), discogp_lm.py:68-84
  - schedule_epoch_lambda: Yu et al. (2024/2025), utils.py (schedule_epoch_lambda)

Reference: https://arxiv.org/abs/2407.03779
"""

import torch
import torch.nn as nn


def gumbel_sigmoid(
    logits: torch.Tensor,
    gs_temp: float = 1.0,
    hard: bool = True,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Sample a Bernoulli-like gate via a Gumbel-sigmoid relaxation.

    Adds Gumbel noise to ``logits``, applies a sigmoid scaled by ``gs_temp``,
    then optionally binarises the result via a straight-through estimator.

    The Gumbel noise is drawn as::

        noise = -log(-log(U1) / -log(U0))   where U0, U1 ~ Uniform(0, 1)

    This gives a reparameterisation of the difference of two Gumbel draws, which
    together with the sigmoid produces the Concrete/BinConcrete distribution.

    Args:
        logits: Un-normalised mask log-odds of shape ``(*)``.
        gs_temp: Temperature that controls the sharpness of the relaxation.
            Low temperatures (e.g. 0.01 for weight masks) push the samples
            toward hard 0/1.  Edge masks typically use 1.0.
        hard: If ``True``, apply the straight-through estimator: the sample is
            rounded to {0, 1} in the forward pass, but gradients flow through
            the continuous sigmoid value in the backward pass.  When ``False``,
            the raw sigmoid output is returned — useful for extracting
            continuous importance scores for ``PruneScores``.
        eps: Small constant for numerical stability inside the log.

    Returns:
        A tensor of the same shape as ``logits`` with values in ``[0, 1]``.
        With ``hard=True``, the forward values are exactly 0 or 1.
    """
    uniform = logits.new_empty([2] + list(logits.shape)).uniform_(0, 1)
    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
    res = torch.sigmoid((logits + noise) / gs_temp)
    if hard:
        res = ((res > 0.5).type_as(res) - res).detach() + res

    return res


def schedule_epoch_lambda(
    epoch: int,
    lambda_0: float,
    max_times: float = 1.0,
    min_times: float = 1.0,
    n_epoch_warmup: int = 0,
    n_epoch_cooldown: int = 0,
) -> float:
    """Compute the scheduled sparsity / completeness coefficient at a given epoch.

    The schedule has three phases:

    1. **Warm-up** (``0 <= epoch < n_epoch_warmup``): linearly ramp from
       ``lambda_0`` up to ``lambda_0 * max_times``.
    2. **Cool-down** (``n_epoch_warmup <= epoch < n_epoch_warmup + n_epoch_cooldown``):
       linearly decay from ``lambda_0 * max_times`` down to ``lambda_0 * min_times``.
    3. **Constant** (``epoch >= n_epoch_warmup + n_epoch_cooldown``): stay at
       ``lambda_0 * min_times``.

    When both ``n_epoch_warmup`` and ``n_epoch_cooldown`` are 0 (the default),
    the function simply returns ``lambda_0`` for every epoch.

    Args:
        epoch: Current training epoch (0-indexed).
        lambda_0: Base coefficient value.
        max_times: Peak multiplier applied to ``lambda_0`` at the end of warm-up.
        min_times: Final multiplier applied to ``lambda_0`` after cool-down.
        n_epoch_warmup: Number of warm-up epochs.
        n_epoch_cooldown: Number of cool-down epochs.

    Returns:
        The coefficient value for the current epoch.
    """
    if epoch < n_epoch_warmup:
        # Linear ramp-up
        return lambda_0 + lambda_0 * (max_times - 1) * epoch / max(n_epoch_cooldown, 1)
    elif epoch < n_epoch_warmup + n_epoch_cooldown:
        # Linear ramp-down
        return (
            lambda_0 * max_times
            - lambda_0 * (max_times - min_times) * (epoch - n_epoch_warmup) / max(n_epoch_cooldown, 1)
        )
    else:
        # Flat plateau
        return lambda_0 * min_times
