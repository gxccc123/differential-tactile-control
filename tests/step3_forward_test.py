"""
Step 3 Forward Test:

Part 1 — Shape smoke-test for all 4 paths (including new structured tactile).
Part 2 — Bit-exact regression for the 3 old paths against Step-0 baseline.
Part 3 — Token-layout verification for the structured path
         (checks that t^s / t^dyn / tac_pred are assembled correctly
          by inspecting n_tactile_tokens wiring inside transformer.forward).

Run from /root/data/dtc/:
    /root/data/conda_envs/vitacformer_new/bin/python tests/step3_forward_test.py
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
        'mu':         mu.cpu() if mu is not None else None,
        'logvar':     logvar.cpu() if logvar is not None else None,
        'tac_hat':    tac_hat.cpu() if tac_hat is not None else None,
    }


# ── Part 1: Shape smoke-test ──────────────────────────────────────────────────

def check_shapes(tag, result, expect_tac_hat):
    ok = True

    def chk(name, val, expected_shape):
        nonlocal ok
        if val is None:
            if expected_shape is None:
                print(f"    {name:12s}: ✓  None")
            else:
                print(f"    {name:12s}: ✗  got None, expected {expected_shape}")
                ok = False
            return
        if val.shape == torch.Size(expected_shape):
            print(f"    {name:12s}: ✓  {tuple(val.shape)}")
        else:
            print(f"    {name:12s}: ✗  got {tuple(val.shape)}, expected {expected_shape}")
            ok = False

    print(f"\n  [{tag}]")
    chk("a_hat",      result['a_hat'],      (B, NUM_Q, STATE_DIM))
    chk("is_pad_hat", result['is_pad_hat'], (B, NUM_Q, 1))
    chk("mu",         result['mu'],         (B, 32))
    chk("logvar",     result['logvar'],     (B, 32))
    chk("tac_hat",    result['tac_hat'],    (B, TAC_NEXT_T, TAC_NEXT_D) if expect_tac_hat else None)
    return ok


# ── Part 2: Bit-exact regression ─────────────────────────────────────────────

def compare_exact(tag, current, baseline):
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
            print(f"      {k:12s}: ✓  bit-exact  {tuple(c.shape)}")
        else:
            mx = (c - b).abs().max().item()
            print(f"      {k:12s}: ✗  max_diff={mx:.3e}  {tuple(c.shape)}")
            all_ok = False
    return all_ok


# ── Part 3: Token-layout probe for structured path ────────────────────────────

def probe_structured_token_layout(device):
    """
    Monkey-patch Transformer.forward to capture the assembled src tensor
    (just after token stacking, before encoder) and verify:
      - src[0] comes from t_s  (structured_tactile_encoder output slot 0)
      - src[1] comes from t_dyn (slot 1)
      - For the hs call (pred_action=True, n_tactile_tokens=2):
          src[0]=t_s, src[1]=t_dyn, src[2]=tactile_pred, src[3]=latent, src[4]=proprio
    """
    from detr.models import build_ACT_model
    from detr.models.transformer import Transformer

    torch.manual_seed(SEED)
    args = make_args(True, False, True)
    model = build_ACT_model(args).to(device)
    model.eval()
    inp = make_inputs(device)

    captured = {}

    orig_forward = Transformer.forward

    def patched_forward(self_tr, src, mask, query_embed, pos_embed,
                        latent_input=None, proprio_input=None,
                        additional_pos_embed=None, tactile=None,
                        tactile_pred=None, tactile_dyn=None):

        # Identify which call this is by query_embed size (18 = tactile, 100 = action)
        key = "hs_tactile" if query_embed.shape[0] == 18 else "hs"

        # Store input tokens before assembly
        captured[f"{key}_tactile"]     = tactile.detach().cpu() if tactile is not None else None
        captured[f"{key}_tactile_dyn"] = tactile_dyn.detach().cpu() if tactile_dyn is not None else None
        captured[f"{key}_tactile_pred"]= tactile_pred.detach().cpu() if tactile_pred is not None else None

        result = orig_forward(self_tr, src, mask, query_embed, pos_embed,
                              latent_input, proprio_input, additional_pos_embed,
                              tactile, tactile_pred, tactile_dyn)
        return result

    Transformer.forward = patched_forward

    with torch.no_grad():
        model(
            inp['qpos'], inp['image'], env_state=None,
            tactile      = inp['tactile'],
            actions      = inp['actions'],
            is_pad       = inp['is_pad'],
            tactile_next = inp['tactile_next'],
            epoch=0,
        )

    Transformer.forward = orig_forward  # restore

    # Also get the actual encoder outputs for comparison
    t_s_ref, t_dyn_ref = None, None
    orig_ste = model.structured_tactile_encoder.forward

    def ste_capture(tactile):
        nonlocal t_s_ref, t_dyn_ref
        t_s_ref, t_dyn_ref = orig_ste(tactile)
        return t_s_ref, t_dyn_ref

    model.structured_tactile_encoder.forward = ste_capture
    with torch.no_grad():
        model(
            inp['qpos'], inp['image'], env_state=None,
            tactile      = inp['tactile'],
            actions      = inp['actions'],
            is_pad       = inp['is_pad'],
            tactile_next = inp['tactile_next'],
            epoch=0,
        )
    model.structured_tactile_encoder.forward = orig_ste

    t_s_ref   = t_s_ref.detach().cpu()
    t_dyn_ref = t_dyn_ref.detach().cpu()

    print("\n  [Structured token layout probe]")
    ok = True

    def assert_equal(name, a, b):
        nonlocal ok
        if a is None:
            print(f"    {name}: ✗  got None")
            ok = False
            return
        if torch.equal(a, b):
            print(f"    {name}: ✓  matches reference  shape={tuple(a.shape)}")
        else:
            mx = (a - b).abs().max().item()
            print(f"    {name}: ✗  max_diff={mx:.3e}")
            ok = False

    def assert_none(name, val):
        nonlocal ok
        if val is None:
            print(f"    {name}: ✓  None (as expected)")
        else:
            print(f"    {name}: ✗  expected None, got shape {tuple(val.shape)}")
            ok = False

    # hs_tactile call: tactile=t_s, tactile_dyn=t_dyn, tactile_pred=None
    assert_equal("hs_tactile.tactile  == t_s  ", captured['hs_tactile_tactile'],     t_s_ref)
    assert_equal("hs_tactile.tactile_dyn == t_dyn", captured['hs_tactile_tactile_dyn'], t_dyn_ref)
    assert_none ("hs_tactile.tactile_pred is None", captured['hs_tactile_tactile_pred'])

    # hs call: tactile=t_s, tactile_dyn=t_dyn, tactile_pred=<not None>
    assert_equal("hs.tactile  == t_s  ", captured['hs_tactile'],     t_s_ref)
    assert_equal("hs.tactile_dyn == t_dyn", captured['hs_tactile_dyn'], t_dyn_ref)
    if captured['hs_tactile_pred'] is not None:
        print(f"    hs.tactile_pred shape: ✓  {tuple(captured['hs_tactile_pred'].shape)}  (non-None as expected)")
    else:
        print(f"    hs.tactile_pred: ✗  got None, expected non-None")
        ok = False

    return ok


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Part 1: shape test ──────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  Part 1 — Shape smoke-test (all 4 paths)")
    print("=" * 58)

    cases = [
        (False, False, False, False, 'use_tactile=False'),
        (True,  False, False, True,  'use_tactile=True (baseline)'),
        (True,  True,  False, True,  'use_differential_tactile=True'),
        (True,  False, True,  True,  'use_structured_tactile=True'),
    ]

    shape_ok = True
    for ut, ud, us, expect_tac, tag in cases:
        res = run_forward(tag, ut, ud, us, device)
        ok  = check_shapes(tag, res, expect_tac)
        shape_ok = shape_ok and ok

    # ── Part 2: bit-exact regression (3 old paths) ─────────────────────────
    print("\n" + "=" * 58)
    print("  Part 2 — Bit-exact regression (3 old paths)")
    print("=" * 58)

    print(f"\nLoading baseline from '{SNAPSHOT_PATH}' ...")
    if not os.path.exists(SNAPSHOT_PATH):
        print("ERROR: Snapshot not found. Run step0_baseline_snapshot.py first.")
        sys.exit(1)
    snapshot = torch.load(SNAPSHOT_PATH, map_location='cpu', weights_only=False)

    regression_cases = [
        ('no_tactile',   False, False, False, 'use_tactile=False'),
        ('baseline',     True,  False, False, 'use_tactile=True (baseline)'),
        ('diff_tactile', True,  True,  False, 'use_differential_tactile=True'),
    ]

    regression_ok = True
    for snap_key, ut, ud, us, tag in regression_cases:
        print(f"\n  [{tag}]")
        current = run_forward(tag, ut, ud, us, device)
        ok = compare_exact(tag, current, snapshot[snap_key])
        regression_ok = regression_ok and ok

    # ── Part 3: token layout probe ──────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  Part 3 — Token layout probe (structured path)")
    print("=" * 58)
    layout_ok = probe_structured_token_layout(device)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    if shape_ok and regression_ok and layout_ok:
        print("✓  Step 3 PASSED — all shapes correct, old paths bit-exact,")
        print("   token layout verified.")
    else:
        if not shape_ok:      print("✗  SHAPE CHECK FAILED.")
        if not regression_ok: print("✗  REGRESSION FAILED — old paths not bit-exact.")
        if not layout_ok:     print("✗  TOKEN LAYOUT PROBE FAILED.")
        sys.exit(1)


if __name__ == '__main__':
    main()
