"""
Step 0: Save baseline snapshot of all 3 existing paths.

Captures forward outputs (shapes + values) with a fixed seed.
This file serves as the regression baseline for Step 1 bit-exact check.

Run from /root/data/dtc/:
    /root/data/conda_envs/vitacformer_new/bin/python tests/step0_baseline_snapshot.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import argparse

# ── fixed hyper-params (match live codebase) ──────────────────────────────────
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
TAC_FLAT   = 18 * 120      # 2160
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

SAVE_PATH = "tests/step0_baseline.pt"


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
        use_structured_tactile=False,   # new flag (not yet used, just here for compat)
        resume_path=None,
    )


def make_inputs(device):
    torch.manual_seed(SEED)
    return dict(
        qpos    = torch.randn(B, QPOS_DIM,          device=device),
        image   = torch.randn(B, N_CAM, 3, IMG_H, IMG_W, device=device),
        actions = torch.randn(B, NUM_Q, STATE_DIM,  device=device),
        is_pad  = torch.zeros(B, NUM_Q, dtype=torch.bool, device=device),
        tactile      = torch.randn(B, TAC_FLAT,            device=device),
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

    result = {
        'a_hat':       a_hat.cpu(),
        'is_pad_hat':  is_pad_hat.cpu(),
        'mu':          mu.cpu(),
        'logvar':      logvar.cpu(),
        'tac_hat':     tac_hat.cpu() if tac_hat is not None else None,
    }

    print(f"\n  [{tag}]")
    print(f"    a_hat:      {tuple(result['a_hat'].shape)}")
    print(f"    is_pad_hat: {tuple(result['is_pad_hat'].shape)}")
    print(f"    mu:         {tuple(result['mu'].shape)}")
    print(f"    logvar:     {tuple(result['logvar'].shape)}")
    print(f"    tac_hat:    {tuple(result['tac_hat'].shape) if result['tac_hat'] is not None else None}")
    print(f"    a_hat[0,0,:3]: {result['a_hat'][0, 0, :3].tolist()}")

    return result


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("\n=== Step 0: Saving baseline snapshot ===")

    snapshot = {}

    snapshot['no_tactile'] = run_case(
        'use_tactile=False',
        use_tactile=False, use_differential_tactile=False,
        device=device,
    )

    snapshot['baseline'] = run_case(
        'use_tactile=True (baseline)',
        use_tactile=True, use_differential_tactile=False,
        device=device,
    )

    snapshot['diff_tactile'] = run_case(
        'use_differential_tactile=True',
        use_tactile=True, use_differential_tactile=True,
        device=device,
    )

    torch.save(snapshot, SAVE_PATH)
    print(f"\n✓  Snapshot saved → {SAVE_PATH}")


if __name__ == '__main__':
    main()
