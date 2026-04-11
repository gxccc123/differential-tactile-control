#!/usr/bin/env bash

# ── GPU count ─────────────────────────────────────────────────────────────────
# Set NUM_GPUS=1 for single-GPU training (equivalent to plain `python`).
# Set NUM_GPUS=2/4/8 for multi-GPU DDP training via torchrun.
NUM_GPUS=2

# ── Tactile representation selector ──────────────────────────────────────────
# Uncomment EXACTLY ONE of the four lines below.
#
# Mode A: ViTacFormer baseline (raw single tactile token)
# TACTILE_FLAGS="--use_tactile"
#
# Mode B: Differential tactile (first-order diff, single fused token)
# TACTILE_FLAGS="--use_tactile --use_differential_tactile"
#
# Mode C: Structured tactile (dual token: t^s + t^dyn)
TACTILE_FLAGS="--use_tactile --use_structured_tactile"
#
# Mode D: Pure vision (no tactile)
# TACTILE_FLAGS=""
# ─────────────────────────────────────────────────────────────────────────────

/root/data/conda_envs/vitacformer_new/bin/torchrun \
    --nproc_per_node=$NUM_GPUS \
    imitate_episodes.py \
    --task_name flip_book \
    --ckpt_dir ckpt_dir/flip_book \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
    --num_epochs 2000 --lr 1e-4 \
    --seed 0 \
    $TACTILE_FLAGS \
    --use_swanlab \
    --swanlab_project ViTacFormer-DTC \
    --swanlab_log_interval 10
    # --swanlab_mode local
    # --resume_path /path/to/checkpoint.ckpt
