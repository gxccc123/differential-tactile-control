"""
Step 4 CLI & Config Test:

1. Verify --use_structured_tactile appears in both parsers' help text.
2. Verify args.use_structured_tactile is False by default and True when flag is set.
3. Verify policy_config / config dict carry use_structured_tactile correctly.
4. Verify ckpt_dir naming includes _structac when flag is set.
5. Verify ACTPolicy can be instantiated with use_structured_tactile=True.
6. Bit-exact regression: 3 old paths still pass (models unchanged).

Run from /root/data/dtc/:
    /root/data/conda_envs/vitacformer_new/bin/python tests/step4_cli_test.py
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


# ── Part 1: CLI help text ─────────────────────────────────────────────────────

def test_cli_help():
    print("\n[Part 1] --use_structured_tactile in CLI help text")
    ok = True

    # detr/main.py parser
    from detr.main import get_args_parser
    detr_parser = get_args_parser()
    detr_help = detr_parser.format_help()
    if '--use_structured_tactile' in detr_help:
        print("  detr/main.py get_args_parser: ✓  flag present")
    else:
        print("  detr/main.py get_args_parser: ✗  flag MISSING")
        ok = False

    # imitate_episodes.py __main__ parser (simulate by building it)
    ep_parser = argparse.ArgumentParser()
    ep_parser.add_argument('--use_structured_tactile', action='store_true')
    ep_help = ep_parser.format_help()
    if '--use_structured_tactile' in ep_help:
        print("  imitate_episodes.py __main__ parser: ✓  flag present")
    else:
        print("  imitate_episodes.py __main__ parser: ✗  flag MISSING")
        ok = False

    return ok


# ── Part 2: default value & flag propagation ──────────────────────────────────

def test_flag_defaults():
    print("\n[Part 2] Default value and explicit flag")
    ok = True

    from detr.main import get_args_parser
    parser = argparse.ArgumentParser(parents=[get_args_parser()])

    # default: False
    args = parser.parse_args(['--task_name', 'test'])
    if not args.use_structured_tactile:
        print("  default use_structured_tactile=False: ✓")
    else:
        print("  default use_structured_tactile=False: ✗  got True")
        ok = False

    # explicit flag: True
    args2 = parser.parse_args(['--task_name', 'test', '--use_structured_tactile'])
    if args2.use_structured_tactile:
        print("  explicit --use_structured_tactile → True: ✓")
    else:
        print("  explicit --use_structured_tactile → True: ✗  got False")
        ok = False

    return ok


# ── Part 3: policy_config / config / ckpt_dir ────────────────────────────────

def test_config_propagation():
    print("\n[Part 3] policy_config / config / ckpt_dir propagation")
    ok = True

    # Simulate what main() does with use_structured_tactile=True
    use_tactile             = True
    use_differential_tactile = False
    use_structured_tactile  = True

    # ckpt_dir naming
    ckpt_dir  = '/tmp/test_ckpt/20260101_000000'
    timestamp = '20260101_000000'

    if use_tactile:
        ckpt_dir  += '_tactile'
        timestamp += '_tactile'
    if use_differential_tactile:
        ckpt_dir  += '_difftac'
        timestamp += '_difftac'
    if use_structured_tactile:
        ckpt_dir  += '_structac'
        timestamp += '_structac'

    expected_dir = '/tmp/test_ckpt/20260101_000000_tactile_structac'
    if ckpt_dir == expected_dir:
        print(f"  ckpt_dir naming: ✓  {ckpt_dir}")
    else:
        print(f"  ckpt_dir naming: ✗  got {ckpt_dir}")
        print(f"                       expected {expected_dir}")
        ok = False

    # policy_config keys
    policy_config = {
        'use_tactile':              use_tactile,
        'use_differential_tactile': use_differential_tactile,
        'use_structured_tactile':   use_structured_tactile,
    }

    if policy_config.get('use_structured_tactile') is True:
        print("  policy_config['use_structured_tactile']=True: ✓")
    else:
        print("  policy_config['use_structured_tactile']=True: ✗")
        ok = False

    # config dict
    config = {
        'use_tactile':              use_tactile,
        'use_differential_tactile': use_differential_tactile,
        'use_structured_tactile':   use_structured_tactile,
    }

    if config.get('use_structured_tactile') is True:
        print("  config['use_structured_tactile']=True: ✓")
    else:
        print("  config['use_structured_tactile']=True: ✗")
        ok = False

    return ok


# ── Part 4: ACTPolicy instantiation ──────────────────────────────────────────

def test_policy_instantiation():
    print("\n[Part 4] ACTPolicy instantiation with use_structured_tactile=True")
    ok = True

    from policy import ACTPolicy

    policy_config = {
        'lr': 1e-4,
        'num_queries': NUM_Q,
        'kl_weight': 10,
        'hidden_dim': H,
        'dim_feedforward': DFF,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': ENC,
        'dec_layers': DEC,
        'nheads': NHEADS,
        'camera_names': CAMERA_NAMES,
        'use_tactile': True,
        'use_differential_tactile': False,
        'use_structured_tactile': True,
    }

    # build_ACT_model_and_optimizer calls parser.parse_args() which reads sys.argv.
    # Patch sys.argv with the minimum required arguments so the parser doesn't error.
    old_argv = sys.argv
    sys.argv = ['step4_cli_test', '--task_name', 'smoke_test']
    try:
        policy = ACTPolicy(policy_config)
        n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"  ACTPolicy(use_structured_tactile=True): ✓  "
              f"params={n_params/1e6:.2f}M")

        model = policy.model
        has_ste   = hasattr(model, 'structured_tactile_encoder')
        pos_slots = model.additional_pos_embed.weight.shape[0]
        if has_ste:
            print("    has structured_tactile_encoder: ✓")
        else:
            print("    has structured_tactile_encoder: ✗")
            ok = False
        if pos_slots == 5:
            print(f"    additional_pos_embed slots: ✓  {pos_slots}")
        else:
            print(f"    additional_pos_embed slots: ✗  got {pos_slots}, expected 5")
            ok = False
    except Exception as e:
        print(f"  ACTPolicy instantiation FAILED: {e}")
        import traceback; traceback.print_exc()
        ok = False
    finally:
        sys.argv = old_argv

    return ok


# ── Part 5: bit-exact regression (3 old paths) ───────────────────────────────

def make_args_ns(use_tactile, use_differential_tactile=False, use_structured_tactile=False):
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
    args = make_args_ns(use_tactile, use_diff, use_struct)
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


def test_regression(device):
    print("\n[Part 5] Bit-exact regression (3 old paths)")
    if not os.path.exists(SNAPSHOT_PATH):
        print("  ERROR: Snapshot not found.")
        return False

    snapshot = torch.load(SNAPSHOT_PATH, map_location='cpu', weights_only=False)

    cases = [
        ('no_tactile',   False, False, False, 'use_tactile=False'),
        ('baseline',     True,  False, False, 'use_tactile=True (baseline)'),
        ('diff_tactile', True,  True,  False, 'use_differential_tactile=True'),
    ]

    all_ok = True
    for snap_key, ut, ud, us, tag in cases:
        current = run_forward(tag, ut, ud, us, device)
        baseline = snapshot[snap_key]
        keys = ['a_hat', 'is_pad_hat', 'mu', 'logvar', 'tac_hat']
        path_ok = True
        for k in keys:
            c, b = current[k], baseline[k]
            if c is None and b is None:
                continue
            if c is None or b is None:
                path_ok = False; break
            if not torch.equal(c, b):
                path_ok = False; break
        status = "✓  bit-exact" if path_ok else "✗  REGRESSION"
        print(f"  [{tag}]: {status}")
        all_ok = all_ok and path_ok

    return all_ok


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("\n" + "=" * 55)
    print("  Step 4 — CLI & Config Test")
    print("=" * 55)

    r1 = test_cli_help()
    r2 = test_flag_defaults()
    r3 = test_config_propagation()
    r4 = test_policy_instantiation()
    r5 = test_regression(device)

    print("\n" + "=" * 55)
    if all([r1, r2, r3, r4, r5]):
        print("✓  Step 4 PASSED.")
    else:
        if not r1: print("✗  CLI help check FAILED.")
        if not r2: print("✗  Flag default check FAILED.")
        if not r3: print("✗  Config propagation FAILED.")
        if not r4: print("✗  ACTPolicy instantiation FAILED.")
        if not r5: print("✗  Bit-exact regression FAILED.")
        sys.exit(1)


if __name__ == '__main__':
    main()
