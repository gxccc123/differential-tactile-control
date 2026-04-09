"""
Smoke test for DifferentialTactileEncoder integration.

Verifies:
  a. use_differential_tactile=False  → original ViTacFormer path forward OK
  b. use_differential_tactile=True   → diff tactile branch forward OK
  c. Key tensor shapes are printed for both cases

Run from /root/data/dtc/:
    python test_diff_tactile.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import argparse

# ── constants matching the live codebase ──────────────────────────────────────
B               = 2      # batch size
NUM_QUERIES     = 100    # must be 100 (asserted in DETRVAE)
HIDDEN_DIM      = 256
DIM_FEEDFORWARD = 3200
ENC_LAYERS      = 4
DEC_LAYERS      = 7
NHEADS          = 8
QPOS_DIM        = 348
TAC_HISTORY     = 18
TAC_DIM         = 120
TAC_DIM_ALL     = TAC_HISTORY * TAC_DIM   # 2160
STATE_DIM       = 58
N_CAM           = 4
IMG_H, IMG_W    = 224, 320

CAMERA_NAMES = [
    '/observe/vision/head/stereo/lefteye/rgb',
    '/observe/vision/head/stereo/righteye/rgb',
    '/observe/vision/right_wrist/fisheye/rgb',
    '/observe/vision/left_wrist/fisheye/rgb',
]


def make_args(use_differential_tactile: bool) -> argparse.Namespace:
    args = argparse.Namespace(
        lr=1e-4,
        lr_backbone=1e-5,
        batch_size=B,
        weight_decay=1e-4,
        epochs=1,
        lr_drop=200,
        clip_max_norm=0.1,
        backbone='resnet18',
        dilation=False,
        position_embedding='sine',
        camera_names=CAMERA_NAMES,
        enc_layers=ENC_LAYERS,
        dec_layers=DEC_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        hidden_dim=HIDDEN_DIM,
        dropout=0.1,
        nheads=NHEADS,
        num_queries=NUM_QUERIES,
        pre_norm=False,
        masks=False,
        # imitate_episodes compat args (not used by build)
        eval=False,
        onscreen_render=False,
        ckpt_dir='smoke_test_tmp',
        policy_class='ACT',
        task_name='smoke',
        seed=0,
        num_epochs=1,
        kl_weight=10,
        chunk_size=NUM_QUERIES,
        temporal_agg=False,
        algo='act',
        ckpt_path=None,
        use_tactile=True,
        use_differential_tactile=use_differential_tactile,
        resume_path=None,
    )
    return args


def run_forward(use_differential_tactile: bool, device: torch.device):
    tag = "diff_tac=True " if use_differential_tactile else "diff_tac=False"
    print(f"\n{'='*60}")
    print(f"  Case: {tag}")
    print(f"{'='*60}")

    from detr.models import build_ACT_model
    args = make_args(use_differential_tactile)
    model = build_ACT_model(args).to(device)
    model.eval()

    # ── dummy inputs ─────────────────────────────────────────────────────────
    qpos    = torch.randn(B, QPOS_DIM,         device=device)
    image   = torch.randn(B, N_CAM, 3, IMG_H, IMG_W, device=device)
    actions = torch.randn(B, NUM_QUERIES, STATE_DIM,  device=device)
    is_pad  = torch.zeros(B, NUM_QUERIES, dtype=torch.bool, device=device)
    tactile      = torch.randn(B, TAC_DIM_ALL, device=device)   # [B, 2160]
    tactile_next = torch.randn(B, TAC_HISTORY, TAC_DIM, device=device)  # [B, 18, 120]

    print(f"  Input  tactile      : {tuple(tactile.shape)}")

    with torch.no_grad():
        a_hat, is_pad_hat, (mu, logvar), tac_hat = model(
            qpos, image, env_state=None,
            tactile=tactile,
            actions=actions,
            is_pad=is_pad,
            tactile_next=tactile_next,
            epoch=0,
        )

    print(f"  Output a_hat        : {tuple(a_hat.shape)}")
    print(f"  Output tac_hat      : {tuple(tac_hat.shape) if tac_hat is not None else None}")
    print(f"  Output mu           : {tuple(mu.shape)}")
    print(f"  Output logvar       : {tuple(logvar.shape)}")

    # verify diff encoder internals if active
    if use_differential_tactile:
        diff_enc = model.tactile_diff_encoder
        raw_token = model.input_proj_tactile(tactile)
        diff_token = diff_enc(tactile)
        fused = model.tactile_fusion(torch.cat([raw_token, diff_token], dim=-1))
        print(f"  [diff] raw_token    : {tuple(raw_token.shape)}")
        print(f"  [diff] diff_token   : {tuple(diff_token.shape)}")
        print(f"  [diff] fused_input  : {tuple(fused.shape)}")

    print(f"  ✓  Forward OK for {tag}")
    return True


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ok_baseline = run_forward(use_differential_tactile=False, device=device)
    ok_diff     = run_forward(use_differential_tactile=True,  device=device)

    print(f"\n{'='*60}")
    if ok_baseline and ok_diff:
        print("  ALL SMOKE TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
