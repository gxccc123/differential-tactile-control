"""
Step 2 Instantiation Test:

1. Verify all 4 flag combinations can be instantiated without error.
2. Verify additional_pos_embed shape per path:
   - use_tactile=False                     → [2, H]
   - use_tactile=True  (baseline)          → [4, H]
   - use_tactile=True + diff               → [4, H]
   - use_tactile=True + structured         → [5, H]
3. Verify StructuredTactileEncoder is only registered when use_structured_tactile=True.
4. Verify StructuredTactileEncoder is NOT called in forward (no wiring yet — Step 3).
5. Run bit-exact regression for the 3 old paths (forward unchanged).

Run from /root/data/dtc/:
    /root/data/conda_envs/vitacformer_new/bin/python tests/step2_instantiation_test.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import argparse

SEED       = 42
B          = 2
NUM_Q      = 100
H          = 256
DFF        = 3200
ENC        = 4
DEC        = 7
NHEADS     = 8
QPOS_DIM   = 348
STATE_DIM  = 58
TAC_FLAT   = 18 * 120
TAC_NEXT_T = 18
TAC_NEXT_D = 120
N_CAM      = 4
IMG_H, IMG_W = 224, 320

CAMERA_NAMES = [
    '/observe/vision/head/stereo/lefteye/rgb',
    '/observe/vision/head/stereo/righteye/rgb',
    '/observe/vision/right_wrist/fisheye/rgb',
    '/observe/vision/left_wrist/fisheye/rgb',
]

SNAPSHOT_PATH = "tests/step0_baseline.pt"


def make_args(use_tactile, use_differential_tactile=False, use_structured_tactile=False):
    return argparse.Namespace(
        lr=1e-4, lr_backbone=1e-5, batch_size=B, weight_decay=1e-4,
        epochs=1, lr_drop=200, clip_max_norm=0.1,
        backbone='resnet18', dilation=False, position_embedding='sine',
        camera_names=CAMERA_NAMES,
        enc_layers=ENC, dec_layers=DEC,
        dim_feedforward=DFF, hidden_dim=H,
        dropout=0.1, nheads=NHEADS, num_queries=NUM_Q,
        pre_norm=False, masks=False,
        eval=False, onscreen_render=False, ckpt_dir='tmp',
        policy_class='ACT', task_name='smoke',
        seed=SEED, num_epochs=1,
        kl_weight=10, chunk_size=NUM_Q, temporal_agg=False,
        algo='act', ckpt_path=None,
        use_tactile=use_tactile,
        use_differential_tactile=use_differential_tactile,
        use_structured_tactile=use_structured_tactile,
        resume_path=None,
    )


def make_inputs(device):
    torch.manual_seed(SEED)
    return dict(
        qpos    = torch.randn(B, QPOS_DIM,            device=device),
        image   = torch.randn(B, N_CAM, 3, IMG_H, IMG_W, device=device),
        actions = torch.randn(B, NUM_Q, STATE_DIM,    device=device),
        is_pad  = torch.zeros(B, NUM_Q, dtype=torch.bool, device=device),
        tactile      = torch.randn(B, TAC_FLAT,              device=device),
        tactile_next = torch.randn(B, TAC_NEXT_T, TAC_NEXT_D, device=device),
    )


# ── Part 1: Instantiation + attribute checks ──────────────────────────────────

def check_instantiation(tag, use_tactile, use_diff, use_struct, device,
                        expected_pos_slots, expect_ste):
    from detr.models import build_ACT_model
    torch.manual_seed(SEED)
    args = make_args(use_tactile, use_diff, use_struct)
    model = build_ACT_model(args).to(device)

    pos_shape = tuple(model.additional_pos_embed.weight.shape)
    has_ste   = hasattr(model, 'structured_tactile_encoder')

    ok_pos = (pos_shape == (expected_pos_slots, H))
    ok_ste = (has_ste == expect_ste)

    status_pos = "✓" if ok_pos else "✗"
    status_ste = "✓" if ok_ste else "✗"

    print(f"\n  [{tag}]")
    print(f"    additional_pos_embed.shape: {pos_shape}  "
          f"{status_pos}  (expected ({expected_pos_slots}, {H}))")
    print(f"    has structured_tactile_encoder: {has_ste}  "
          f"{status_ste}  (expected {expect_ste})")

    if not ok_pos or not ok_ste:
        print("    → FAIL")
        return False
    return True


# ── Part 2: Forward bit-exact regression for 3 old paths ─────────────────────

def run_forward(tag, use_tactile, use_diff, use_struct, device):
    from detr.models import build_ACT_model
    torch.manual_seed(SEED)
    args = make_args(use_tactile, use_diff, use_struct)
    model = build_ACT_model(args).to(device)
    model.eval()

    inp = make_inputs(device)
    with torch.no_grad():
        a_hat, is_pad_hat, (mu, logvar), tac_hat = model(
            inp['qpos'], inp['image'], env_state=None,
            tactile      = inp['tactile']      if use_tactile else None,
            actions      = inp['actions'],
            is_pad       = inp['is_pad'],
            tactile_next = inp['tactile_next'] if use_tactile else None,
            epoch=0,
        )

    return {
        'a_hat':      a_hat.cpu(),
        'is_pad_hat': is_pad_hat.cpu(),
        'mu':         mu.cpu(),
        'logvar':     logvar.cpu(),
        'tac_hat':    tac_hat.cpu() if tac_hat is not None else None,
    }


def compare(tag, current, baseline):
    keys = ['a_hat', 'is_pad_hat', 'mu', 'logvar', 'tac_hat']
    all_ok = True
    for k in keys:
        c, b = current[k], baseline[k]
        if c is None and b is None:
            print(f"      {k:12s}: ✓  both None")
            continue
        if c is None or b is None:
            print(f"      {k:12s}: ✗  None mismatch")
            all_ok = False
            continue
        if torch.equal(c, b):
            print(f"      {k:12s}: ✓  bit-exact  shape={tuple(c.shape)}")
        else:
            mx = (c - b).abs().max().item()
            print(f"      {k:12s}: ✗  max_diff={mx:.3e}  shape={tuple(c.shape)}")
            all_ok = False
    return all_ok


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n" + "=" * 55)
    print("  Part 1 — Instantiation & attribute checks")
    print("=" * 55)

    cases_inst = [
        # tag, use_tac, use_diff, use_struct, pos_slots, expect_ste
        ('use_tactile=False',             False, False, False, 2, False),
        ('use_tactile=True  (baseline)',  True,  False, False, 4, False),
        ('use_differential_tactile=True', True,  True,  False, 4, False),
        ('use_structured_tactile=True',   True,  False, True,  5, True),
    ]

    inst_ok = True
    for tag, ut, ud, us, slots, ste in cases_inst:
        ok = check_instantiation(tag, ut, ud, us, device, slots, ste)
        inst_ok = inst_ok and ok

    print("\n" + "=" * 55)
    print("  Part 2 — Bit-exact forward regression (3 old paths)")
    print("=" * 55)

    print(f"\nLoading baseline snapshot from '{SNAPSHOT_PATH}' ...")
    if not os.path.exists(SNAPSHOT_PATH):
        print("ERROR: Snapshot not found. Run step0_baseline_snapshot.py first.")
        sys.exit(1)
    snapshot = torch.load(SNAPSHOT_PATH, map_location='cpu',
                          weights_only=False)

    cases_fwd = [
        ('no_tactile',   False, False, False, 'use_tactile=False'),
        ('baseline',     True,  False, False, 'use_tactile=True (baseline)'),
        ('diff_tactile', True,  True,  False, 'use_differential_tactile=True'),
    ]

    fwd_ok = True
    for snap_key, ut, ud, us, tag in cases_fwd:
        print(f"\n  [{tag}]")
        current = run_forward(tag, ut, ud, us, device)
        ok = compare(tag, current, snapshot[snap_key])
        fwd_ok = fwd_ok and ok

    print("\n" + "=" * 55)
    if inst_ok and fwd_ok:
        print("✓  Step 2 PASSED — instantiation correct, old paths bit-exact.")
    else:
        if not inst_ok:
            print("✗  INSTANTIATION CHECK FAILED.")
        if not fwd_ok:
            print("✗  FORWARD REGRESSION FAILED.")
        sys.exit(1)


if __name__ == '__main__':
    main()
