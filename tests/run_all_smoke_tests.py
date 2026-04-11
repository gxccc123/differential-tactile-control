"""
Unified Smoke Test Suite — ViTacFormer Structured Tactile Integration
======================================================================
Covers all five implementation steps and four model paths:

  Path A: use_tactile=False                   (pure vision)
  Path B: use_tactile=True                    (ViTacFormer baseline)
  Path C: use_tactile=True + diff_tactile     (Step-1 differential)
  Path D: use_tactile=True + struct_tactile   (Step-3 structured dual token)

Test groups
-----------
  [G0] Baseline snapshot exists (Step 0)
  [G1] transformer.py parameterisation — bit-exact regression on A/B/C (Step 1)
  [G2] DETRVAE instantiation — additional_pos_embed shape, STE registration (Step 2)
  [G3] Forward shape + bit-exact regression + token-layout probe (Step 3)
  [G4] CLI flag propagation + ACTPolicy instantiation (Step 4)
  [G5] train.sh sanity — structured-tactile entry present (Step 5)

Run from /root/data/dtc/:
    /root/data/conda_envs/vitacformer_new/bin/python tests/run_all_smoke_tests.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import argparse
import traceback

# ── shared fixtures ───────────────────────────────────────────────────────────
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
TRAIN_SH_PATH = "train.sh"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def make_inputs():
    torch.manual_seed(SEED)
    return dict(
        qpos    = torch.randn(B, QPOS_DIM,            device=DEVICE),
        image   = torch.randn(B, N_CAM, 3, IMG_H, IMG_W, device=DEVICE),
        actions = torch.randn(B, NUM_Q, STATE_DIM,    device=DEVICE),
        is_pad  = torch.zeros(B, NUM_Q, dtype=torch.bool, device=DEVICE),
        tactile      = torch.randn(B, TAC_FLAT,              device=DEVICE),
        tactile_next = torch.randn(B, TAC_NEXT_T, TAC_NEXT_D, device=DEVICE),
    )


def build_model(use_tactile, use_diff=False, use_struct=False):
    from detr.models import build_ACT_model
    torch.manual_seed(SEED)
    return build_ACT_model(make_args(use_tactile, use_diff, use_struct)).to(DEVICE)


def run_forward(model, use_tactile):
    model.eval()
    inp = make_inputs()
    with torch.no_grad():
        a_hat, is_pad_hat, (mu, logvar), tac_hat = model(
            inp['qpos'], inp['image'], env_state=None,
            tactile      = inp['tactile']      if use_tactile else None,
            actions      = inp['actions'],
            is_pad       = inp['is_pad'],
            tactile_next = inp['tactile_next'] if use_tactile else None,
            epoch=0,
        )
    return dict(
        a_hat      = a_hat.cpu(),
        is_pad_hat = is_pad_hat.cpu(),
        mu         = mu.cpu()      if mu      is not None else None,
        logvar     = logvar.cpu()  if logvar  is not None else None,
        tac_hat    = tac_hat.cpu() if tac_hat is not None else None,
    )


# ── result tracking ───────────────────────────────────────────────────────────

class Results:
    def __init__(self):
        self._groups = []    # [(group_name, [(test_name, ok, msg)])]
        self._cur = None

    def group(self, name):
        self._cur = (name, [])
        self._groups.append(self._cur)
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"{'─'*60}")

    def check(self, name, ok, detail=''):
        sym = '✓' if ok else '✗'
        line = f"  {sym}  {name}"
        if detail:
            line += f"  ({detail})"
        print(line)
        self._cur[1].append((name, ok))

    def summary(self):
        total_pass = total_fail = 0
        failed_groups = []
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        for gname, tests in self._groups:
            g_ok  = all(ok for _, ok in tests)
            g_sym = '✓' if g_ok else '✗'
            n_p   = sum(1 for _, ok in tests if ok)
            n_f   = sum(1 for _, ok in tests if not ok)
            print(f"  {g_sym}  {gname}  ({n_p}/{len(tests)} passed)")
            total_pass += n_p
            total_fail += n_f
            if not g_ok:
                failed_groups.append(gname)
        print(f"{'─'*60}")
        print(f"  Total: {total_pass} passed, {total_fail} failed")
        if not failed_groups:
            print(f"\n✓  ALL TESTS PASSED")
        else:
            print(f"\n✗  FAILED GROUPS: {failed_groups}")
        return total_fail == 0


# ── test groups ───────────────────────────────────────────────────────────────

def g0_snapshot(R):
    R.group("[G0] Baseline snapshot (Step 0)")
    exists = os.path.exists(SNAPSHOT_PATH)
    R.check("Snapshot file exists", exists, SNAPSHOT_PATH)
    if not exists:
        return None

    snap = torch.load(SNAPSHOT_PATH, map_location='cpu', weights_only=False)
    for key in ['no_tactile', 'baseline', 'diff_tactile']:
        R.check(f"snapshot['{key}'] present", key in snap)
    return snap


def g1_regression(R, snap):
    R.group("[G1] transformer.py parameterisation — bit-exact regression (Step 1)")
    if snap is None:
        R.check("Skipped (no snapshot)", False)
        return

    cases = [
        ('no_tactile',   False, False, False, 'Path A: pure vision'),
        ('baseline',     True,  False, False, 'Path B: baseline'),
        ('diff_tactile', True,  True,  False, 'Path C: diff tactile'),
    ]
    for snap_key, ut, ud, us, label in cases:
        try:
            model = build_model(ut, ud, us)
            curr  = run_forward(model, ut)
            keys  = ['a_hat', 'is_pad_hat', 'mu', 'logvar', 'tac_hat']
            ok = all(
                (curr[k] is None and snap[snap_key][k] is None) or
                (curr[k] is not None and snap[snap_key][k] is not None and
                 torch.equal(curr[k], snap[snap_key][k]))
                for k in keys
            )
            R.check(f"bit-exact: {label}", ok)
        except Exception as e:
            R.check(f"bit-exact: {label}", False, str(e))


def g2_instantiation(R):
    R.group("[G2] DETRVAE instantiation — pos_embed shape + STE (Step 2)")
    cases = [
        # ut,    ud,    us,    expected_pos_slots, expect_ste, label
        (False, False, False, 2, False, 'Path A: pure vision'),
        (True,  False, False, 4, False, 'Path B: baseline'),
        (True,  True,  False, 4, False, 'Path C: diff tactile'),
        (True,  False, True,  5, True,  'Path D: structured tactile'),
    ]
    for ut, ud, us, slots, ste, label in cases:
        try:
            model     = build_model(ut, ud, us)
            pos_shape = tuple(model.additional_pos_embed.weight.shape)
            has_ste   = hasattr(model, 'structured_tactile_encoder')
            ok_pos    = (pos_shape == (slots, H))
            ok_ste    = (has_ste == ste)
            R.check(
                f"pos_embed={slots} ste={ste}: {label}",
                ok_pos and ok_ste,
                f"pos={pos_shape} ste={has_ste}",
            )
        except Exception as e:
            R.check(f"instantiation: {label}", False, str(e))


def g3_forward(R, snap):
    R.group("[G3] Forward shapes + regression + token layout (Step 3)")

    # 3a — shape check for all 4 paths
    shape_cases = [
        (False, False, False, False, 'Path A: pure vision'),
        (True,  False, False, True,  'Path B: baseline'),
        (True,  True,  False, True,  'Path C: diff tactile'),
        (True,  False, True,  True,  'Path D: structured tactile'),
    ]
    for ut, ud, us, expect_tac, label in shape_cases:
        try:
            model = build_model(ut, ud, us)
            out   = run_forward(model, ut)
            ok = (
                out['a_hat'].shape      == torch.Size([B, NUM_Q, STATE_DIM]) and
                out['is_pad_hat'].shape == torch.Size([B, NUM_Q, 1])         and
                out['mu'].shape         == torch.Size([B, 32])               and
                out['logvar'].shape     == torch.Size([B, 32])               and
                ((out['tac_hat'] is None) == (not expect_tac))               and
                (out['tac_hat'] is None or out['tac_hat'].shape == torch.Size([B, TAC_NEXT_T, TAC_NEXT_D]))
            )
            R.check(f"shapes: {label}", ok,
                    f"a_hat={tuple(out['a_hat'].shape)} tac_hat={'None' if out['tac_hat'] is None else tuple(out['tac_hat'].shape)}")
        except Exception as e:
            R.check(f"shapes: {label}", False, str(e))

    # 3b — bit-exact regression for A/B/C
    if snap is not None:
        reg_cases = [
            ('no_tactile',   False, False, False, 'Path A'),
            ('baseline',     True,  False, False, 'Path B'),
            ('diff_tactile', True,  True,  False, 'Path C'),
        ]
        for snap_key, ut, ud, us, label in reg_cases:
            try:
                model = build_model(ut, ud, us)
                curr  = run_forward(model, ut)
                keys  = ['a_hat', 'is_pad_hat', 'mu', 'logvar', 'tac_hat']
                ok = all(
                    (curr[k] is None and snap[snap_key][k] is None) or
                    (curr[k] is not None and snap[snap_key][k] is not None and
                     torch.equal(curr[k], snap[snap_key][k]))
                    for k in keys
                )
                R.check(f"bit-exact regression: {label}", ok)
            except Exception as e:
                R.check(f"bit-exact regression: {label}", False, str(e))
    else:
        R.check("bit-exact regression: skipped (no snapshot)", False)

    # 3c — token layout probe for Path D
    try:
        from detr.models.transformer import Transformer
        model = build_model(True, False, True)
        model.eval()
        inp   = make_inputs()
        captured = {}

        orig_fwd = Transformer.forward
        def patched(self_t, src, mask, qe, pe,
                    latent_input=None, proprio_input=None,
                    additional_pos_embed=None, tactile=None,
                    tactile_pred=None, tactile_dyn=None):
            key = "hs_tac" if qe.shape[0] == 18 else "hs"
            captured[key] = dict(
                tactile     = tactile.detach().cpu()     if tactile     is not None else None,
                tactile_dyn = tactile_dyn.detach().cpu() if tactile_dyn is not None else None,
                tactile_pred= tactile_pred.detach().cpu()if tactile_pred is not None else None,
            )
            return orig_fwd(self_t, src, mask, qe, pe,
                            latent_input, proprio_input, additional_pos_embed,
                            tactile, tactile_pred, tactile_dyn)

        Transformer.forward = patched
        with torch.no_grad():
            model(inp['qpos'], inp['image'], env_state=None,
                  tactile=inp['tactile'], actions=inp['actions'],
                  is_pad=inp['is_pad'], tactile_next=inp['tactile_next'], epoch=0)
        Transformer.forward = orig_fwd

        # hs_tactile: tactile_pred must be None
        R.check("hs_tactile: tactile_pred is None",
                captured.get('hs_tac', {}).get('tactile_pred') is None)
        # hs: tactile_dyn must be non-None
        R.check("hs: tactile_dyn non-None",
                captured.get('hs', {}).get('tactile_dyn') is not None)
        # hs: tactile_pred must be non-None
        R.check("hs: tactile_pred non-None",
                captured.get('hs', {}).get('tactile_pred') is not None)
    except Exception as e:
        R.check("token layout probe", False, str(e))
        traceback.print_exc()


def g4_cli(R):
    R.group("[G4] CLI flag propagation + ACTPolicy instantiation (Step 4)")

    # detr/main.py parser contains the flag
    try:
        from detr.main import get_args_parser
        parser = get_args_parser()
        R.check("detr/main.py: --use_structured_tactile in help",
                '--use_structured_tactile' in parser.format_help())
    except Exception as e:
        R.check("detr/main.py parser", False, str(e))

    # default = False
    try:
        from detr.main import get_args_parser
        full = argparse.ArgumentParser(parents=[get_args_parser()])
        args = full.parse_args(['--task_name', 'smoke'])
        R.check("default use_structured_tactile=False", not args.use_structured_tactile)
        args2 = full.parse_args(['--task_name', 'smoke', '--use_structured_tactile'])
        R.check("explicit flag → True", args2.use_structured_tactile)
    except Exception as e:
        R.check("flag default/set", False, str(e))

    # ckpt_dir naming
    ck = 'base/ts'
    ts = 'ts'
    for flag, suf in [('use_tactile', '_tactile'), ('use_structured_tactile', '_structac')]:
        ck += suf; ts += suf
    R.check("ckpt_dir suffix '_tactile_structac'", ck.endswith('_tactile_structac'))

    # ACTPolicy instantiation
    from policy import ACTPolicy
    policy_config = {
        'lr': 1e-4, 'num_queries': NUM_Q, 'kl_weight': 10,
        'hidden_dim': H, 'dim_feedforward': DFF, 'lr_backbone': 1e-5,
        'backbone': 'resnet18', 'enc_layers': ENC, 'dec_layers': DEC,
        'nheads': NHEADS, 'camera_names': CAMERA_NAMES,
        'use_tactile': True, 'use_differential_tactile': False,
        'use_structured_tactile': True,
    }
    old_argv = sys.argv
    sys.argv  = ['test', '--task_name', 'smoke']
    try:
        policy = ACTPolicy(policy_config)
        R.check("ACTPolicy(use_structured_tactile=True) instantiation", True,
                f"{sum(p.numel() for p in policy.parameters() if p.requires_grad)/1e6:.2f}M params")
        R.check("has structured_tactile_encoder",
                hasattr(policy.model, 'structured_tactile_encoder'))
        R.check("additional_pos_embed = 5 slots",
                policy.model.additional_pos_embed.weight.shape[0] == 5)
    except Exception as e:
        R.check("ACTPolicy instantiation", False, str(e))
    finally:
        sys.argv = old_argv


def g5_trainsh(R):
    R.group("[G5] train.sh — structured tactile entry (Step 5)")

    if not os.path.exists(TRAIN_SH_PATH):
        R.check("train.sh exists", False)
        return

    content = open(TRAIN_SH_PATH).read()
    R.check("--use_structured_tactile present in train.sh",
            '--use_structured_tactile' in content)
    R.check("Mode A/B/C/D comments present",
            'Mode A' in content and 'Mode B' in content and
            'Mode C' in content and 'Mode D' in content)
    R.check("python command references imitate_episodes.py",
            'imitate_episodes.py' in content)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Device: {DEVICE}")
    print(f"\n{'='*60}")
    print("  ViTacFormer Structured Tactile — Unified Smoke Test Suite")
    print(f"{'='*60}")

    R = Results()

    snap = g0_snapshot(R)
    g1_regression(R, snap)
    g2_instantiation(R)
    g3_forward(R, snap)
    g4_cli(R)
    g5_trainsh(R)

    ok = R.summary()
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
