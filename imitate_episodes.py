import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from utils import unnormalize_image, normalize_action, denormalize_action, normalize_obs_lowdim, denormalize_obs_lowdim, normalize_tactile, denormalize_tactile, normalize_tactile_next, denormalize_tactile_next, apply_joint_mask
from policy import ACTPolicy
# from visualize_episodes import save_videos
from dataset.ha_pipelinev2_dataset import HaPipelineV2DatasetD020
from dataset.data import data
from dataset.data_tactile import data_tactile
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm, trange

import IPython
e = IPython.embed

import torchvision.utils as vutils
import os
from PIL import Image
import torchvision.transforms.functional as TF

import cv2


def to_python_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().float().mean().item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def init_swanlab_run(config, timestamp):
    if not config.get('use_swanlab', False):
        return None

    print(f"[SwanLab] Initializing tracking in {config.get('swanlab_mode', 'cloud')} mode...")

    try:
        import swanlab
    except ImportError:
        print('[SwanLab] swanlab is not installed, skip experiment tracking.')
        return None

    use_differential_tactile = config.get('use_differential_tactile', False)
    run_name = config.get('swanlab_run_name') or timestamp
    init_kwargs = {
        'project':           config.get('swanlab_project', 'ViTacFormer-DTC'),
        'experiment_name':   run_name,
        'config':            config,
        'logdir':            os.path.join(config['ckpt_dir'], 'swanlab'),
        'mode':              config.get('swanlab_mode', 'cloud'),
        'tags': [
            config.get('task_name', 'unknown_task'),
            config.get('policy_class', 'unknown_policy'),
            'tactile'      if config.get('use_tactile')              else 'vision_only',
            'diff_tactile' if use_differential_tactile               else 'raw_tactile',
        ],
    }

    description = config.get('swanlab_description')
    if description:
        init_kwargs['description'] = description

    try:
        run = swanlab.init(**init_kwargs)
        print('[SwanLab] Tracking initialized.')
        return run
    except Exception as e:
        print(f'[SwanLab] Initialization failed, skip tracking: {e}')
        return None


def log_swanlab_metrics(run, metrics):
    if run is None:
        return

    import swanlab

    sanitized = {}
    for key, value in metrics.items():
        if value is None:
            continue
        sanitized[key] = to_python_scalar(value)

    if sanitized:
        swanlab.log(sanitized)


def finish_swanlab_run(run):
    if run is None:
        return

    import swanlab
    print('[SwanLab] Finishing run...')
    swanlab.finish()
    print('[SwanLab] Run finished.')


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    use_tactile = args['use_tactile']
    use_differential_tactile = args.get('use_differential_tactile', False)
    resume_path = args['resume_path']
    use_swanlab = args.get('use_swanlab', False)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ckpt_dir = os.path.join(ckpt_dir, timestamp)

    if use_tactile:
        ckpt_dir = ckpt_dir + "_tactile"
        timestamp = timestamp + "_tactile"
    if use_differential_tactile:
        ckpt_dir = ckpt_dir + "_difftac"
        timestamp = timestamp + "_difftac"


    os.makedirs(ckpt_dir, exist_ok=True)

    episode_len = 10000
    camera_names = ['/observe/vision/head/stereo/lefteye/rgb','/observe/vision/head/stereo/righteye/rgb','/observe/vision/right_wrist/fisheye/rgb','/observe/vision/left_wrist/fisheye/rgb']

    # fixed parameters
    state_dim = 58
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'use_tactile': use_tactile,
                         'use_differential_tactile': use_differential_tactile,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        # 'real_robot': not is_sim,
        'use_tactile': use_tactile,
        'use_differential_tactile': use_differential_tactile,
        'resume_path': resume_path,
        'use_swanlab':             use_swanlab,
        'swanlab_project':         args.get('swanlab_project', 'ViTacFormer-DTC'),
        'swanlab_run_name':        args.get('swanlab_run_name', None),
        'swanlab_mode':            args.get('swanlab_mode', 'cloud'),
        'swanlab_description':     args.get('swanlab_description', None),
        'swanlab_log_interval':    args.get('swanlab_log_interval', 10),
        'lr_config': {
            'policy': 'CosineAnnealing',
            'warmup': 'linear',
            'warmup_iters': 1000,
            'warmup_ratio': 1.0 / 10,
            'min_lr_ratio': 1e-1,
        }
    }

    norm_stats_cache = os.path.join(ckpt_dir, 'dataset_stats.pkl')

    data['train']['norm_stats_cache'] = norm_stats_cache
    data['val']['norm_stats_cache'] = norm_stats_cache
    data_tactile['train']['norm_stats_cache'] = norm_stats_cache
    data_tactile['val']['norm_stats_cache'] = norm_stats_cache


    if not use_tactile:
        train_dataset = HaPipelineV2DatasetD020(**data['train'])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=36, prefetch_factor=1)

        # val_dataset = HaPipelineV2DatasetD020(**data['val'])
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=False, num_workers=36, prefetch_factor=1)
    else:
        train_dataset = HaPipelineV2DatasetD020(**data_tactile['train'])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=36, prefetch_factor=1)

        # val_dataset = HaPipelineV2DatasetD020(**data_tactile['val'])
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=False, num_workers=36, prefetch_factor=1)

    normalizer = train_dataset.get_normalizer()



    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'normalize.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(normalizer, f)

    best_ckpt_info = train_bc(train_dataloader, normalizer, train_dataset, timestamp, config)
    best_epoch, best_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, loss {best_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def forward_pass(data, policy, normalizer, device, use_tactile, epoch=0):
    image_data = data["image"]               # [B, N_cam, 3, H, W]
    qpos_data = data["lowdim"]               # [B, T1, D1]
    action_data = data["action"]            # [B, T, D_action]
    is_pad = data["action_mask"]            # [B, T]

    # normalize
    qpos_data_norm = normalize_obs_lowdim(qpos_data, normalizer)  # [B, T1, D1]
    action_data_norm = normalize_action(action_data, normalizer)  # [B, T, D_action]

    # === apply masking to hand joint
    hand_mask = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    # right_hand mask
    qpos_data_norm = apply_joint_mask(qpos_data_norm, hand_mask, start_index=7)

    # left_hand mask
    qpos_data_norm = apply_joint_mask(qpos_data_norm, hand_mask, start_index=35)

    # flatten
    B, T1, D1 = qpos_data_norm.shape
    qpos_data_norm = qpos_data_norm.view(B, T1 * D1)  # → [B, T1 * D1]

    # move to device
    qpos_data_norm = qpos_data_norm.to(device)
    image_data = image_data.to(device)
    action_data_norm = action_data_norm.to(device)
    is_pad = is_pad.to(device)

    if use_tactile:
        tactile = data["tactile"]                          # [B, T2, D2]
        tactile_norm = normalize_tactile(tactile, normalizer)  # normalize
        B, T2, D2 = tactile_norm.shape
        tactile_norm = tactile_norm.view(B, T2 * D2)                # → [B, T2 * D2]
        tactile_norm = tactile_norm.to(device)                     

        tactile_next = data["tactile_next"]                          # [B, T2, D2]
        tactile_next_norm = normalize_tactile_next(tactile_next, normalizer)  # normalize
        tactile_next_norm = tactile_next_norm.to(device)                     


        return policy(qpos_data_norm, image_data, action_data_norm, is_pad, device, tactile_norm, tactile_next_norm, epoch)

    return policy(qpos_data_norm, image_data, action_data_norm, is_pad, device)


def train_bc(train_dataloader, normalizer, dataset, timestamp, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    use_tactile = config['use_tactile']
    resume_path = config.get('resume_path', None)
    swanlab_log_interval = max(1, config.get('swanlab_log_interval', 10))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(seed)

    start_epoch = 0
    global_step = 0
    best_loss = np.inf
    best_ckpt_info = None

    from transformers import get_cosine_schedule_with_warmup

    policy = make_policy(policy_class, policy_config)
    policy.to(device)
    optimizer = make_optimizer(policy_class, policy)

    # === 构建 scheduler ===
    total_iters = num_epochs * len(train_dataloader)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['lr_config']['warmup_iters'],
        num_training_steps=total_iters,
    )

    train_history = []
    global_step = 0
    best_loss = np.inf
    best_ckpt_info = None

    swanlab_run = init_swanlab_run(config, timestamp)

    log_swanlab_metrics(swanlab_run, {
        'meta/train_dataset_size': len(train_dataloader.dataset),
        'meta/train_num_batches':  len(train_dataloader),
        'meta/batch_size':         train_dataloader.batch_size,
    })

    # === resume ===
    if resume_path is not None and os.path.exists(resume_path):
        print(f"[Resume] Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        policy.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        best_loss = checkpoint.get('best_loss', checkpoint.get('min_val_loss', np.inf))
        best_ckpt_info = checkpoint.get('best_ckpt_info', None)

    try:
        for epoch in tqdm(range(start_epoch, num_epochs)):
            print(f'\nEpoch {epoch}')

            # training
            policy.train()
            optimizer.zero_grad()
            epoch_history = []
            with tqdm(train_dataloader, desc=f"Train Epoch {epoch}", leave=False) as tepoch:
                for _, data in enumerate(tepoch):
                    data = dataset.postprocess(data, device, use_tactile)
                    forward_dict = forward_pass(data, policy, normalizer, device, use_tactile, epoch)
                    loss = forward_dict['loss']
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    detached = detach_dict(forward_dict)
                    epoch_history.append(detached)
                    train_history.append(detached)

                    current_lr = optimizer.param_groups[0]['lr']
                    tepoch.set_postfix(
                        loss=loss.item(),
                        lr=f"{current_lr:.2e}",
                        refresh=False
                    )

                    global_step += 1

                    if global_step % swanlab_log_interval == 0:
                        step_metrics = {
                            f'train/step_{k}': v for k, v in detached.items()
                        }
                        step_metrics.update({
                            'train/lr':          current_lr,
                            'train/epoch':       epoch,
                            'train/global_step': global_step,
                        })
                        log_swanlab_metrics(swanlab_run, step_metrics)

            epoch_summary = compute_dict_mean(epoch_history)
            epoch_train_loss = epoch_summary['loss']
            print(f'Train loss: {epoch_train_loss:.5f}')

            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

            if epoch_train_loss < best_loss:
                best_loss = epoch_train_loss
                best_ckpt_info = (epoch, best_loss, deepcopy(policy.state_dict()))

            epoch_metrics = {
                f'train/epoch_{k}': v for k, v in epoch_summary.items()
            }
            epoch_metrics.update({
                'train/epoch':       epoch,
                'train/global_step': global_step,
                'train/lr':          optimizer.param_groups[0]['lr'],
                'train/best_loss':   best_loss,
            })
            log_swanlab_metrics(swanlab_run, epoch_metrics)

            if epoch % 5 == 0:
                ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_loss_{epoch_train_loss:.3f}.ckpt')
                torch.save({
                    'model':         policy.state_dict(),
                    'optimizer':     optimizer.state_dict(),
                    'scheduler':     scheduler.state_dict(),
                    'epoch':         epoch,
                    'global_step':   global_step,
                    'best_loss':     best_loss,
                    'min_val_loss':  best_loss,
                    'best_ckpt_info': best_ckpt_info,
                }, ckpt_path)
                log_swanlab_metrics(swanlab_run, {
                    'checkpoint/epoch':     epoch,
                    'checkpoint/best_loss': best_loss,
                })
    finally:
        finish_swanlab_run(swanlab_run)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    if best_ckpt_info is None:
        best_ckpt_info = (num_epochs - 1, float('inf'), deepcopy(policy.state_dict()))

    best_epoch, best_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_loss_{best_loss}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, best loss {best_loss:.6f} at epoch {best_epoch}')

    return best_ckpt_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_tactile', action='store_true')
    parser.add_argument('--use_differential_tactile', action='store_true')
    parser.add_argument('--resume_path', type=str, default=None, help='path to resume checkpoint')
    parser.add_argument('--use_swanlab', action='store_true')
    parser.add_argument('--swanlab_project', type=str, default='ViTacFormer-DTC', help='SwanLab project name')
    parser.add_argument('--swanlab_run_name', type=str, default=None, help='SwanLab experiment name')
    parser.add_argument('--swanlab_mode', type=str, default='cloud', choices=['cloud', 'local', 'offline', 'disabled'], help='SwanLab logging mode')
    parser.add_argument('--swanlab_description', type=str, default=None, help='SwanLab experiment description')
    parser.add_argument('--swanlab_log_interval', type=int, default=10, help='log metrics to SwanLab every N steps')

    main(vars(parser.parse_args()))
