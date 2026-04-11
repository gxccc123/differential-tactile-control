"""
Step 1 Regression Test: verify all existing paths produce bit-exact output
after transformer.py parameterization.

Loads the Step 0 baseline snapshot and compares tensor values with allclose
(atol=0, rtol=0 → exact match), since no weights changed and default
n_tactile_tokens=1 should reproduce identical computation graphs.

Run from /root/data/dtc/:
    /root/data/conda_envs/vitacformer_new/bin/python tests/step1_regression_test.py
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


def make_args(use_tactile, use_differential_tactile):
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
        use_structured_tactile=False,
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


def run_case(tag, use_tactile, use_differential_tactile, device):
    from detr.models import build_ACT_model
    torch.manual_seed(SEED)
    args = make_args(use_tactile, use_differential_tactile)
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
        c = current[k]
        b = baseline[k]
        if c is None and b is None:
            print(f"    {k:12s}: ✓  both None")
            continue
        if c is None or b is None:
            print(f"    {k:12s}: ✗  one is None! current={c is None}, baseline={b is None}")
            all_ok = False
            continue
        if c.shape != b.shape:
            print(f"    {k:12s}: ✗  shape mismatch {c.shape} vs {b.shape}")
            all_ok = False
            continue
        if torch.equal(c, b):
            print(f"    {k:12s}: ✓  bit-exact  shape={tuple(c.shape)}")
        else:
            max_diff = (c - b).abs().max().item()
            print(f"    {k:12s}: ✗  NOT exact  max_diff={max_diff:.3e}  shape={tuple(c.shape)}")
            all_ok = False
    return all_ok


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading baseline snapshot from '{SNAPSHOT_PATH}' ...")
    if not os.path.exists(SNAPSHOT_PATH):
        print("ERROR: Snapshot not found. Run step0_baseline_snapshot.py first.")
        sys.exit(1)
    snapshot = torch.load(SNAPSHOT_PATH, map_location='cpu')

    print("\n=== Step 1: Bit-exact regression test ===\n")

    cases = [
        ('no_tactile',    False, False, 'use_tactile=False'),
        ('baseline',      True,  False, 'use_tactile=True (baseline)'),
        ('diff_tactile',  True,  True,  'use_differential_tactile=True'),
    ]

    overall = True
    for snap_key, use_tac, use_diff, tag in cases:
        print(f"[{tag}]")
        current = run_case(tag, use_tac, use_diff, device)
        ok = compare(tag, current, snapshot[snap_key])
        overall = overall and ok
        print()

    print("=" * 50)
    if overall:
        print("✓  ALL PATHS BIT-EXACT — Step 1 passed.")
    else:
        print("✗  REGRESSION DETECTED — Step 1 FAILED. Stop and investigate.")
        sys.exit(1)


if __name__ == '__main__':
    main()
