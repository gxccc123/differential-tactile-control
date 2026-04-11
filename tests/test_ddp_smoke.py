"""
DDP smoke test for imitate_episodes.py

Verifies:
  G-DDP-1  DDP initialises and each process binds to the correct GPU
  G-DDP-2  DistributedSampler shards the dataset (each rank sees a subset)
  G-DDP-3  model forward pass succeeds on every rank
  G-DDP-4  backward + optimizer.step() updates parameters (for all 4 paths)
  G-DDP-5  only rank 0 prints / saves checkpoints

The test is launched by torchrun, not plain python:

  /root/data/conda_envs/vitacformer_new/bin/torchrun \\
      --nproc_per_node=2 tests/test_ddp_smoke.py

If only one GPU is available, export SINGLE_GPU=1 and run with python directly:

  SINGLE_GPU=1 /root/data/conda_envs/vitacformer_new/bin/python \\
      tests/test_ddp_smoke.py
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── helpers imported directly from imitate_episodes ──────────────────────────
from imitate_episodes import (
    setup_ddp, cleanup_ddp,
    _get_saveable_state_dict, _load_state_dict_into_policy,
    _build_optimizer_from_config,
)
from policy import ACTPolicy


# ─────────────────────────────────────────────────────────────────────────────
# Minimal policy config that covers one of the four model paths.
# We test one path per invocation; the chosen path is controlled by an env var:
#   TACTILE_PATH=none|baseline|diff|structured
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_NAMES = [
    '/observe/vision/head/stereo/lefteye/rgb',
    '/observe/vision/head/stereo/righteye/rgb',
    '/observe/vision/right_wrist/fisheye/rgb',
    '/observe/vision/left_wrist/fisheye/rgb',
]

HIDDEN_DIM  = 512
CHUNK_SIZE  = 100   # must match DETRVAE's num_queries assertion

def _make_policy_config(tactile_path: str) -> dict:
    return {
        'lr':                        1e-4,
        'lr_backbone':               1e-5,
        'num_queries':               CHUNK_SIZE,
        'kl_weight':                 10,
        'hidden_dim':                HIDDEN_DIM,
        'dim_feedforward':           3200,
        'backbone':                  'resnet18',
        'enc_layers':                4,
        'dec_layers':                7,
        'nheads':                    8,
        'camera_names':              CAMERA_NAMES,
        'use_tactile':               tactile_path != 'none',
        'use_differential_tactile':  tactile_path == 'diff',
        'use_structured_tactile':    tactile_path == 'structured',
    }


def _make_fake_batch(device: torch.device, use_tactile: bool):
    """Return a minimal synthetic batch that matches the forward_pass signature."""
    B         = 2
    N_cam     = len(CAMERA_NAMES)
    H, W      = 480, 640
    T_act     = CHUNK_SIZE  # must match num_queries
    QPOS_DIM  = 348         # 6 frames * 58 joint dims, hardcoded in detr_vae.py
    ACT_DIM   = 58          # state_dim
    T_tac     = 18
    D_tac     = 120

    image    = torch.randn(B, N_cam, 3, H, W,   device=device)
    qpos     = torch.randn(B, QPOS_DIM,          device=device)
    action   = torch.randn(B, T_act, ACT_DIM,    device=device)
    is_pad   = torch.zeros(B, T_act, dtype=torch.bool, device=device)

    if use_tactile:
        tactile      = torch.randn(B, T_tac * D_tac, device=device)
        tactile_next = torch.randn(B, T_tac, D_tac,  device=device)
        return image, qpos, action, is_pad, tactile, tactile_next
    return image, qpos, action, is_pad, None, None


# ─────────────────────────────────────────────────────────────────────────────

def run_smoke(rank: int, world_size: int, tactile_path: str,
              tmp_dir: str, single_gpu: bool):

    is_main = (rank == 0)

    # ── G-DDP-1: DDP init and GPU binding ────────────────────────────────────
    if not single_gpu:
        setup_ddp(rank)

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    if is_main:
        bound = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        print(f"[G-DDP-1] rank={rank} / world={world_size} bound to device={bound}  PASS")

    # ── G-DDP-2: DistributedSampler sharding ─────────────────────────────────
    dummy_ds = TensorDataset(torch.arange(100))
    if world_size > 1:
        sampler = DistributedSampler(dummy_ds, num_replicas=world_size,
                                     rank=rank, shuffle=False)
    else:
        sampler = None

    loader = DataLoader(dummy_ds, batch_size=10,
                        sampler=sampler, shuffle=(sampler is None))

    indices_seen = []
    for (batch,) in loader:
        indices_seen.extend(batch.tolist())

    if world_size > 1:
        # Each rank should see exactly 100 // world_size distinct indices
        expected_per_rank = 100 // world_size
        assert len(indices_seen) == expected_per_rank, (
            f"rank={rank} expected {expected_per_rank} samples, "
            f"got {len(indices_seen)}")
        if is_main:
            print(f"[G-DDP-2] DistributedSampler: each rank sees "
                  f"{expected_per_rank}/{100} samples  PASS")
    else:
        assert len(indices_seen) == 100
        if is_main:
            print(f"[G-DDP-2] Single-GPU: all 100 samples seen  PASS")

    # ── G-DDP-3 & G-DDP-4: forward + backward + step ─────────────────────────
    # Patch sys.argv so build_ACT_model_and_optimizer can call parse_known_args
    # without hitting argparse errors.
    old_argv = sys.argv[:]
    sys.argv = ['smoke_test', '--task_name', 'smoke', '--batch_size', '2',
                '--num_epochs', '1', '--lr', '1e-4', '--seed', '0',
                '--policy_class', 'ACT', '--kl_weight', '10',
                f'--chunk_size', str(CHUNK_SIZE),
                '--hidden_dim', str(HIDDEN_DIM),
                '--dim_feedforward', '3200']
    if tactile_path != 'none':
        sys.argv.append('--use_tactile')
    if tactile_path == 'diff':
        sys.argv.append('--use_differential_tactile')
    if tactile_path == 'structured':
        sys.argv.append('--use_structured_tactile')

    policy_config = _make_policy_config(tactile_path)
    policy = ACTPolicy(policy_config)
    sys.argv = old_argv

    policy.to(device)

    if world_size > 1:
        policy.model = DDP(policy.model, device_ids=[rank],
                           find_unused_parameters=True)
        optimizer = _build_optimizer_from_config(policy_config, policy.model)
    else:
        optimizer = policy.configure_optimizers()

    # Snapshot a parameter before step()
    probe_param = next(policy.parameters()).detach().clone()

    use_tactile = policy_config['use_tactile']
    image, qpos, action, is_pad, tactile, tactile_next = _make_fake_batch(
        device, use_tactile)

    policy.train()
    optimizer.zero_grad()

    if use_tactile:
        out = policy(qpos, image, action, is_pad, device,
                     tactile, tactile_next, epoch=0)
    else:
        out = policy(qpos, image, action, is_pad, device)

    # ── G-DDP-3: forward ─────────────────────────────────────────────────────
    assert 'loss' in out, "forward dict missing 'loss' key"
    loss = out['loss']
    assert loss.isfinite(), f"loss is not finite: {loss.item()}"
    if is_main:
        print(f"[G-DDP-3] path={tactile_path}  forward OK  "
              f"loss={loss.item():.4f}  PASS")

    # ── G-DDP-4: backward + optimizer step ───────────────────────────────────
    loss.backward()
    optimizer.step()

    probe_after = next(policy.parameters()).detach().clone()
    params_changed = not torch.equal(probe_param, probe_after)
    assert params_changed, "parameters did not change after optimizer.step()"
    if is_main:
        print(f"[G-DDP-4] path={tactile_path}  backward + step OK  "
              f"params_changed={params_changed}  PASS")

    # ── G-DDP-5: rank-0 saves, others skip ───────────────────────────────────
    ckpt_path = os.path.join(tmp_dir, f'smoke_{rank}.ckpt')
    if is_main:
        sd = _get_saveable_state_dict(policy)
        torch.save(sd, ckpt_path)
        # Verify the saved keys have no DDP 'module.' artefact
        assert all(k.startswith('model.') for k in sd), \
            "saved state dict contains unexpected key prefix"
        # Verify it can be loaded back into a fresh non-DDP policy
        fresh_policy_config = _make_policy_config(tactile_path)
        old_argv2 = sys.argv[:]
        sys.argv = ['smoke_test', '--task_name', 'smoke', '--batch_size', '2',
                    '--num_epochs', '1', '--lr', '1e-4', '--seed', '0',
                    '--policy_class', 'ACT', '--kl_weight', '10',
                    f'--chunk_size', str(CHUNK_SIZE),
                    '--hidden_dim', str(HIDDEN_DIM),
                    '--dim_feedforward', '3200']
        fresh_policy = ACTPolicy(fresh_policy_config)
        sys.argv = old_argv2
        _load_state_dict_into_policy(fresh_policy, sd)
        print(f"[G-DDP-5] rank=0 saved + reloaded checkpoint OK  PASS")
    else:
        assert not os.path.exists(ckpt_path), \
            f"non-rank-0 process (rank={rank}) should not save checkpoint"
        print(f"[G-DDP-5] rank={rank} correctly skipped checkpoint save  PASS")

    if world_size > 1:
        dist.barrier()

    if is_main:
        print(f"\n=== DDP smoke test PASSED (path={tactile_path}, "
              f"world_size={world_size}) ===\n")

    if not single_gpu and dist.is_initialized():
        cleanup_ddp()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    single_gpu = bool(int(os.environ.get('SINGLE_GPU', '0')))
    tactile_path = os.environ.get('TACTILE_PATH', 'structured')

    if single_gpu:
        rank       = 0
        world_size = 1
    else:
        rank       = int(os.environ.get('RANK',       0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_smoke(rank=rank, world_size=world_size,
                  tactile_path=tactile_path,
                  tmp_dir=tmp_dir, single_gpu=single_gpu)
