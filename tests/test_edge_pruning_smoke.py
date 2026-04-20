"""Quick smoke test for the Edge Pruning integration."""
import torch

# Test 1: l0_utils imports and basic functionality
from auto_circuit.utils.l0_utils import (
    cdf_stretched_concrete,
    deterministic_z,
    get_linear_schedule_with_warmup,
)

log_alpha = torch.tensor([10.0, -10.0, 0.0, 5.0, -5.0])
cdf_val = cdf_stretched_concrete(0, log_alpha)
det_z = deterministic_z(log_alpha)
print(f"[l0_utils] CDF at 0: {cdf_val}")
print(f"[l0_utils] Deterministic z: {det_z}")
print(f"[l0_utils] High log_alpha → keep (high z): {det_z[0]:.4f}")
print(f"[l0_utils] Low log_alpha  → ablate (low z): {det_z[1]:.4f}")

# Test 2: Verify against Edge-Pruning's implementation
# For log_alpha=10: sigmoid(10 / (2/3) * 0.8) = sigmoid(12) ≈ 1.0
# For log_alpha=-10: sigmoid(-10 / (2/3) * 0.8) = sigmoid(-12) ≈ 0.0
assert det_z[0].item() > 0.9, f"High log_alpha should give high z, got {det_z[0]}"
assert det_z[1].item() < 0.1, f"Low log_alpha should give low z, got {det_z[1]}"
print("[l0_utils] ✓ Sanity checks passed")

# Test 3: Scheduler
opt = torch.optim.Adam([torch.tensor(1.0, requires_grad=True)], lr=0.1)
sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=100)
assert sched is not None
print("[l0_utils] ✓ Scheduler created")

# Test 4: edge_pruning imports
from auto_circuit.prune_algos.edge_pruning import edge_pruning_prune_scores
print("[edge_pruning] ✓ Module imported")

# Test 5: Registration
from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT, EDGE_PRUNING_PRUNE_ALGO
assert "Edge Pruning" in PRUNE_ALGO_DICT, f"EP not in dict. Keys: {list(PRUNE_ALGO_DICT.keys())}"
print(f"[prune_algos] ✓ Edge Pruning registered (key='{EDGE_PRUNING_PRUNE_ALGO.key}')")

# Test 6: sample_hard_concrete equivalence check
from auto_circuit.utils.tensor_ops import sample_hard_concrete
torch.manual_seed(42)
la = torch.tensor([5.0, -5.0, 0.0])
z = sample_hard_concrete(la, batch_size=10000)
mean_z = z.mean(dim=0)
# log_alpha=5 (positive) → HC samples biased toward 1
# log_alpha=-5 → samples biased toward 0
print(f"[sample_HC] Mean z for log_alpha=[5, -5, 0]: {mean_z}")
assert mean_z[0] > 0.7, f"Positive log_alpha should give high mean, got {mean_z[0]}"
assert mean_z[1] < 0.3, f"Negative log_alpha should give low mean, got {mean_z[1]}"
print("[sample_HC] ✓ HC sampling behaves as expected")

print("\n=== ALL TESTS PASSED ===")
