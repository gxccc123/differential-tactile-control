"""
Smoke test for StructuredTactileEncoder.

Tests:
  1. forward with [B, T, D]  input
  2. forward with [B, T*D]   input (flat)
  3. Both produce identical outputs (shape & values)
  4. Prints all key intermediate tensor shapes

Run from /root/data/dtc/:
    /root/data/conda_envs/vitacformer_new/bin/python tests/test_structured_tactile_encoder.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from models.structured_tactile_encoder import StructuredTactileEncoder

# ── Config matching live codebase ─────────────────────────────────────────────
B               = 4
T               = 18
D               = 120
STEM_DIM        = 64
HIDDEN_DIM      = 256


def print_shape(name: str, tensor: torch.Tensor) -> None:
    print(f"  {name:<20s}: {tuple(tensor.shape)}")


def run_case(label: str, tactile: torch.Tensor, encoder: StructuredTactileEncoder) -> dict:
    print(f"\n{'='*60}")
    print(f"  Case: {label}")
    print(f"{'='*60}")
    print_shape("input tactile", tactile)

    with torch.no_grad():
        debug = encoder.forward_debug(tactile)

    print_shape("h_t (stem out)",     debug["h_t"])
    print_shape("s_t (smooth state)", debug["s_t"])
    print_shape("v_t (velocity)",     debug["v_t"])
    print_shape("r_t (residual)",     debug["r_t"])
    print_shape("s_agg (pooled)",     debug["s_agg"])
    print_shape("t_v  (vel token)",   debug["t_v"])
    print_shape("t_r  (res token)",   debug["t_r"])
    print_shape("state_token  t^s",   debug["state_token"])
    print_shape("dynamic_token t^dyn",debug["dynamic_token"])

    # basic shape assertions
    assert debug["h_t"].shape           == (B, T, STEM_DIM),   f"h_t shape wrong: {debug['h_t'].shape}"
    assert debug["s_t"].shape           == (B, T, STEM_DIM),   f"s_t shape wrong"
    assert debug["v_t"].shape           == (B, T, STEM_DIM),   f"v_t shape wrong"
    assert debug["r_t"].shape           == (B, T, STEM_DIM),   f"r_t shape wrong"
    assert debug["state_token"].shape   == (B, HIDDEN_DIM),    f"state_token shape wrong"
    assert debug["dynamic_token"].shape == (B, HIDDEN_DIM),    f"dynamic_token shape wrong"

    # v_t[0] must be zero (causal, d_0 = 0)
    assert debug["v_t"][:, 0, :].abs().max().item() == 0.0,    "v_t[:,0,:] should be zero"

    # r_t should have near-zero mean across time (residual from mean)
    r_mean = debug["r_t"].mean(dim=1).abs().max().item()
    assert r_mean < 1e-5, f"r_t mean should be ~0, got {r_mean:.2e}"

    print(f"  ✓  All assertions passed for: {label}")
    return debug


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder = StructuredTactileEncoder(
        tactile_dim=D,
        tac_history_cnt=T,
        stem_dim=STEM_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    encoder.eval()

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nStructuredTactileEncoder parameters: {n_params/1e3:.1f}K")

    # ── Case 1: 3-D input [B, T, D] ──────────────────────────────────
    tactile_3d = torch.randn(B, T, D, device=device)
    dbg1 = run_case("[B, T, D]  input", tactile_3d, encoder)

    # ── Case 2: flat input [B, T*D] ──────────────────────────────────
    tactile_flat = tactile_3d.reshape(B, T * D)
    dbg2 = run_case("[B, T*D]   input (flat)", tactile_flat, encoder)

    # ── Case 3: outputs must be identical ────────────────────────────
    print(f"\n{'='*60}")
    print("  Case: [B,T,D] vs [B,T*D] produce identical results")
    print(f"{'='*60}")
    for key in ("state_token", "dynamic_token"):
        diff = (dbg1[key] - dbg2[key]).abs().max().item()
        assert diff < 1e-6, f"  Mismatch in {key}: max diff = {diff:.2e}"
        print(f"  {key:<20s}: max diff = {diff:.2e}  ✓")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ALL SMOKE TESTS PASSED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
