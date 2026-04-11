# ── Tactile representation selector ─────────────────────────────────────────
# Exactly ONE of the three modes below should be active at a time.
#
#  Mode A: ViTacFormer baseline (raw tactile token)
#          → keep --use_tactile, comment out the two diff/struct flags
#
#  Mode B: Differential tactile (first-order diff, single fused token)
#          → keep --use_tactile --use_differential_tactile
#
#  Mode C: Structured tactile (dual token: t^s + t^dyn)   ← new
#          → keep --use_tactile --use_structured_tactile
#          → comment out --use_differential_tactile
#
#  Mode D: Pure vision (no tactile)
#          → comment out --use_tactile and both tactile flags
# ─────────────────────────────────────────────────────────────────────────────

/root/data/conda_envs/vitacformer_new/bin/python imitate_episodes.py \
--task_name flip_book \
--ckpt_dir ckpt_dir/flip_book \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-4 \
--seed 0 \
--use_tactile \
--use_structured_tactile \
# --use_differential_tactile \
--use_swanlab \
--swanlab_project ViTacFormer-DTC \
--swanlab_log_interval 10
# --swanlab_mode local
# --resume_path ~