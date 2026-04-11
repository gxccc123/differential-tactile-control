# Differential Tactile Control for Long-Horizon Dexterous Manipulation

本项目探索长时精细灵巧操作中的**微分触觉控制**方法，基于 [ViTacFormer](https://roboverseorg.github.io/ViTacFormerPage/) 工程底座逐步演进。

---

## 核心问题

长时精细操作并不是一个单纯的"视觉与触觉如何融合"的问题，而是一个更深层的结构性矛盾：

$$\text{长时任务主线} + \text{若干短时关键精细接触窗口}$$

- **长时主线要求一致性（consistency）**：系统必须在较长时间尺度上维持任务推进，不因局部噪声或短时波动而偏航
- **精细窗口要求响应性（responsiveness）**：系统必须对极短但决定成败的关键接触时刻高度敏感——插入最后几毫米、旋拧刚刚咬牙、卡扣闭合瞬间、柔性件边缘搭上

**基本判断**：raw tactile 本身不是最理想的控制变量。更合适的做法是将触觉改写为一组具有明确控制语义的结构化变量，并让视觉在当前交互条件下被选择性调用，而不是被动地全量吞入。

---

## 方法主线

> 长时精细灵巧操作的关键，不是 raw tactile，而是**接触状态、接触动态与接触异常**；视觉的作用不是与触觉逐点对齐，而是在当前交互条件下提供最相关的上下文；控制主体不是显式高低层分解，而是统一交互状态中的双时间尺度动力学。

### 结构化触觉表示

将原始触觉信号 $x_t$ 经编码后分解为三类量：

$$x_t \xrightarrow{\phi} h_t \rightarrow (s_t,\; v_t,\; r_t)$$

| 变量 | 名称 | 语义 | 控制角色 |
|------|------|------|---------|
| $s_t$ | Contact State | 接触主线、稳态压力分布、阶段推进 | 支撑长时任务主线 |
| $v_t$ | Contact Dynamics | 接触如何变化：建立、滑移 onset、受力恶化、对齐改善 | 驱动关键窗口局部修正 |
| $r_t$ | Contact Irregularity | 平滑状态无法解释的局部异常：stick-slip、微振动、瞬时 burst | 捕捉高频接触异常 |

**计算方式**（feature space）：

$$s_t = \mathrm{Smooth}(h_t), \qquad v_t = h_t - h_{t-1}, \qquad r_t = h_t - s_t$$

### 四模块框架

```
┌─────────────────────────────────────────────────────────────┐
│  模块一  Differential Tactile Encoder                        │
│          raw tactile → h_t → (s_t, v_t, r_t)               │
├─────────────────────────────────────────────────────────────┤
│  模块二  Interaction-Conditioned Context Selection           │
│          c_t = f(q_t, a_{t-1}, s_t, v_t, r_t)              │
│          Ṽ_t = Retrieve(V_t, c_t)                           │
│          视觉不做硬空间对齐，而是按交互条件动态选取          │
├─────────────────────────────────────────────────────────────┤
│  模块三  Unified Interaction State Backbone                  │
│          z^main ← 吸收 Ṽ_t, q_t, s_t   （慢、稳、抗噪）    │
│          z^win  ← 吸收 Ṽ_t, q_t, v_t, r_t （快、敏感）     │
│          ω_t = f(v_t, r_t, q_t, z^main)  窗口强度           │
├─────────────────────────────────────────────────────────────┤
│  模块四  Mainline–Refinement Output Heads                    │
│          u^main = f_m(z^main)                               │
│          Δu = f_ref(z^win, Ṽ_t, q_t, v_t, r_t)             │
│          u_t = u^main + ω_t · Δu                            │
└─────────────────────────────────────────────────────────────┘
```

**关键特性**：这不是显式的高层规划器 + 低层修正器，而是在同一个统一系统内部建模两种时间尺度的状态演化——$s_t$ 维持长时主线，$v_t, r_t$ 驱动关键接触窗口中的局部精细控制。

---

## 当前实现状态

本仓库是上述框架的**工程演进底座**，目前已完成前两轮改进并支持多卡训练：

### 四条模型路径

| 模式 | 说明 | `train.sh` 开关 | token 数 |
|------|------|-----------------|----------|
| **Pure Vision** | 无触觉，纯视觉 baseline | `TACTILE_FLAGS=""` | 0 |
| **ViTacFormer Baseline** | raw tactile 单 token | `--use_tactile` | 1 |
| **Differential Tactile** | 一阶差分 + raw 融合 token（第一轮） | `--use_tactile --use_differential_tactile` | 1 |
| **Structured Tactile** | $s_t + v_t + r_t$ 双 token（t^s + t^dyn）| `--use_tactile --use_structured_tactile` | 2 |

Structured Tactile 路径将 $(s_t, v_t, r_t)$ 聚合为两个独立 token 注入 ViTacFormer 主 Transformer：

```
token layout: [t^s | t^dyn | tac_pred | latent | proprio | visual_patches...]
```

### 项目架构

```
dtc/
├── imitate_episodes.py                    # 训练入口（含 DDP 支持）
├── policy.py                              # ACTPolicy 包装
├── train.sh                               # 一键训练脚本
├── detr/
│   ├── main.py                            # 模型构建 & CLI 参数
│   └── models/
│       ├── detr_vae.py                    # DETRVAE（含触觉分支）
│       └── transformer.py                 # 支持可变数量触觉 token
├── models/
│   └── structured_tactile_encoder.py      # s_t / v_t / r_t 编码器
├── dataset/                               # 数据集加载
├── tests/
│   ├── test_ddp_smoke.py                  # DDP 验证（单卡 & 多卡）
│   ├── run_all_smoke_tests.py             # 全量回归测试
│   └── ...
└── differential_tactile_long_horizon_notes.tex  # 方法主线草稿
```

---

## 环境安装

```bash
conda create -n vitacformer python=3.8.10
conda activate vitacformer
pip install torch torchvision
pip install opencv-python matplotlib tqdm einops h5py ipython
pip install transforms3d zarr transformers swanlab
cd dataset/ha_data && pip install -e .
cd detr && pip install -e .
```

---

## 训练

### 快速开始

```bash
conda activate vitacformer
bash train.sh
```

### `train.sh` 核心参数

```bash
# GPU 数量：1 = 单卡，2/4/8 = 多卡 DDP
NUM_GPUS=2

# 触觉模式（四选一，取消对应行注释）
# TACTILE_FLAGS=""                                          # Pure Vision
# TACTILE_FLAGS="--use_tactile"                            # ViTacFormer Baseline
# TACTILE_FLAGS="--use_tactile --use_differential_tactile" # Differential Tactile
TACTILE_FLAGS="--use_tactile --use_structured_tactile"     # Structured Tactile（当前默认）
```

### 手动启动命令

```bash
# 单卡
torchrun --nproc_per_node=1 imitate_episodes.py \
    --task_name flip_book --ckpt_dir ckpt_dir/flip_book \
    --policy_class ACT --kl_weight 10 --chunk_size 100 \
    --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
    --num_epochs 2000 --lr 1e-4 --seed 0 \
    --use_tactile --use_structured_tactile \
    --use_swanlab --swanlab_project ViTacFormer-DTC

# 2 卡
torchrun --nproc_per_node=2 imitate_episodes.py [同上参数]

# 4 卡
torchrun --nproc_per_node=4 imitate_episodes.py [同上参数]
```

### 断点续训

```bash
# 在启动命令末尾加入
--resume_path /path/to/ckpt_dir/policy_epoch_XX_loss_YY.ckpt
```

---

## 多卡训练（DDP）

通过 `torchrun` 启动 PyTorch DDP，无需额外依赖。

| 设计点 | 实现 |
|--------|------|
| 模型并行 | `DDP(policy.model, find_unused_parameters=True)` |
| 数据分片 | `DistributedSampler`，每 epoch `set_epoch` 重新 shuffle |
| Optimizer 重建 | DDP 包装后显式重建，backbone / non-backbone 双 lr group |
| 日志 / 保存 | 仅 rank 0 执行 SwanLab 记录与 checkpoint 保存 |
| Checkpoint 格式 | 剥离 `module.` 前缀，单卡可直接加载 |
| 目录同步 | rank 0 建目录 → `barrier()` → 其他进程继续 |

> **注**：`find_unused_parameters=True` 是必须的，因为 DETRVAE 内部含条件分支（CVAE encoder、tactile_pred head），部分参数在特定 step 不参与前向计算。

---

## 实验跟踪（SwanLab）

```bash
# 云端记录（默认）
--use_swanlab --swanlab_project ViTacFormer-DTC

# 本地记录（无网络环境）
--use_swanlab --swanlab_mode local
```

SwanLab 记录：step loss、epoch loss、lr schedule、best loss、checkpoint 事件、数据集大小。

---

## Checkpoint 目录结构

```
ckpt_dir/flip_book/<timestamp>_tactile_structac/
├── normalize.pkl                        # 数据归一化参数
├── dataset_stats.pkl                    # 数据集统计
├── policy_epoch_0_loss_XX.ckpt          # 每 5 epoch 保存一次
├── policy_epoch_5_loss_XX.ckpt
├── ...
├── policy_last.ckpt                     # 最终 epoch
├── policy_best.ckpt                     # 最优 epoch（按 train loss）
└── swanlab/                             # SwanLab 本地日志
```

---

## 测试

```bash
# DDP smoke test（验证多卡初始化 / 分片 / forward / backward / rank 0 保存）
# 单卡
SINGLE_GPU=1 TACTILE_PATH=structured \
    /root/data/conda_envs/vitacformer_new/bin/python tests/test_ddp_smoke.py

# 2 卡
TACTILE_PATH=structured \
    /root/data/conda_envs/vitacformer_new/bin/torchrun \
    --nproc_per_node=2 tests/test_ddp_smoke.py

# 全量回归测试（四条路径）
/root/data/conda_envs/vitacformer_new/bin/python tests/run_all_smoke_tests.py
```

`TACTILE_PATH` 可选值：`none` | `baseline` | `diff` | `structured`

---

## 后续扩展方向

下列模块在 `.tex` 草稿中已设计，尚未纳入当前工程：

- **Interaction-Conditioned Context Selection**：用 $c_t = f(q_t, a_{t-1}, s_t, v_t, r_t)$ 从视觉特征池中动态检索最相关上下文，代替全局 token 拼接
- **Unified Interaction State Backbone**：双状态通道 $(z^{main}, z^{win})$ + 窗口强度 $\omega_t$，在统一系统中建模长时主线与精细窗口的双时间尺度动力学
- **Mainline–Refinement Output Heads**：$u_t = u^{main} + \omega_t \cdot \Delta u$，窗口修正头可进一步参数化为条件 flow model
- **Directional Tactile Derivatives**：$v_t = [v_t^n, v_t^\tau, v_t^\omega]$，将法向 / 切向 / 旋转分量显式分离
- **Cross-Finger Coupling**：$\Delta v_t^{i,j} = v_t^i - v_t^j$，捕捉多指协同接触的相对变化

---

## 改动摘要

| 改动 | 涉及文件 |
|------|---------|
| `DifferentialTactileEncoder`（一阶差分触觉） | `detr/models/detr_vae.py` |
| `StructuredTactileEncoder`（$s_t / v_t / r_t$ 分解） | `models/structured_tactile_encoder.py` |
| 双触觉 token 注入 Transformer（可变 token 数参数化） | `detr/models/transformer.py` |
| DETRVAE 四路径前向分支 | `detr/models/detr_vae.py` |
| 四条路径 CLI 开关 | `detr/main.py`, `imitate_episodes.py` |
| PyTorch DDP 多卡支持 | `imitate_episodes.py`, `policy.py` |
| SwanLab 实验记录 | `imitate_episodes.py` |
| `torchrun` 启动脚本 | `train.sh` |

---

## 原始项目

本项目基于 [ViTacFormer](https://roboverseorg.github.io/ViTacFormerPage/) 演进，原始论文：

```bibtex
@misc{heng2025vitacformerlearningcrossmodalrepresentation,
      title={ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation},
      author={Liang Heng and Haoran Geng and Kaifeng Zhang and Pieter Abbeel and Jitendra Malik},
      year={2025},
      eprint={2506.15953},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.15953},
}
```
