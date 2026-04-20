"""L0 Hard Concrete utilities for Edge Pruning (Bhaskar et al., NeurIPS 2024).

Training-time HC sampling already exists in auto_circuit.utils.tensor_ops.sample_hard_concrete.
This module adds evaluation-time discretization and related utilities that are specific
to the Edge-Pruning algorithm's binary search thresholding approach.

Reference: https://arxiv.org/abs/2406.16778
Based on: Edge-Pruning/src/modeling/l0.py
"""
import math

import torch as t

# Constants matching Edge-Pruning and AutoCircuit's existing HC implementation
LIMIT_LEFT = -0.1
LIMIT_RIGHT = 1.1
TEMPERATURE = 2 / 3
EPS = 1e-6
FACTOR = 0.8  # Scaling factor for deterministic sigmoid (from EP l0.py line 10)


def cdf_stretched_concrete(x: float, log_alpha: t.Tensor) -> t.Tensor:
    """CDF of the stretched concrete distribution at a given point.

    Computes P(z <= x) for the stretched concrete distribution parameterized by
    log_alpha. Used to compute expected number of zeros via P(z <= 0).

    Matches Edge-Pruning l0.py:cdf_stretched_concrete exactly.

    Args:
        x: The point at which to evaluate the CDF.
        log_alpha: The log-alpha parameters of the distribution.

    Returns:
        The CDF value at x for each element in log_alpha.
    """
    x_01 = (x - LIMIT_LEFT) / (LIMIT_RIGHT - LIMIT_LEFT)
    intermediate = math.log(x_01) - math.log(1 - x_01)
    prob_unclamped = t.sigmoid(TEMPERATURE * intermediate - log_alpha)
    return prob_unclamped.clamp(EPS, 1 - EPS)


def deterministic_z(log_alpha: t.Tensor) -> t.Tensor:
    """Convert log_alpha parameters to deterministic masks for evaluation.

    Uses the binary search discretization from the Edge-Pruning paper (Appendix A):
    1. Compute expected number of non-zero entries from the CDF
    2. Create a soft mask via sigmoid(log_alpha * FACTOR / TEMPERATURE)
    3. Zero out the bottom-k entries matching the expected number of zeros

    Matches Edge-Pruning l0.py:deterministic_z_from_log_alpha exactly.

    Args:
        log_alpha: The log-alpha parameters of the distribution.

    Returns:
        Deterministic mask values in [0, 1], same shape as log_alpha.
    """
    size = log_alpha.numel()

    # Expected number of non-zero entries: sum(1 - P(z <= 0))
    expected_num_nonzeros = t.sum(1 - cdf_stretched_concrete(0, log_alpha))
    expected_num_zeros = size - expected_num_nonzeros
    num_zeros = int(t.round(expected_num_zeros).item())

    # Create soft mask
    soft_mask = t.sigmoid(log_alpha / TEMPERATURE * FACTOR).reshape(-1)

    if num_zeros > 0:
        if soft_mask.ndim == 0:
            soft_mask = t.tensor(0.0, device=log_alpha.device)
        else:
            k = min(num_zeros, soft_mask.numel())
            _, indices = t.topk(soft_mask, k=k, largest=False)
            soft_mask[indices] = 0.0

    return soft_mask.reshape(log_alpha.shape)


def get_linear_schedule_with_warmup(
    optimizer: t.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> t.optim.lr_scheduler.LambdaLR:
    """Linear warmup then linear decay LR schedule.

    Matches HuggingFace's get_linear_schedule_with_warmup used in Edge-Pruning.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of steps for linear LR warmup (0 to base_lr).
        num_training_steps: Total number of training steps (LR decays to 0).

    Returns:
        A PyTorch LR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
