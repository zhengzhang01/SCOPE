import os
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
import time
import random
from typing import *
import itertools
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import io
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.version
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import utils3d
import click
from tqdm import tqdm, trange
import mlflow
torch.backends.cudnn.benchmark = False      # Varying input size, make sure cudnn benchmark is disabled

from torch.utils.data import DataLoader, ConcatDataset
from scope.dataset.tartanair import TartanAirPoint
from scope.dataset.pointodyssey import PointOdysseyPoint
from scope.dataset.spring import SpringPoint
from scope.dataset.vkitti2 import VKitti2Point
from scope.dataset.lightwheel import LightWheelPoint
from scope.dataset.hypersim import HypersimPoint
from scope.dataset.gtaim import GTAIMPoint
from scope.dataset.mvssynth import MVSSynthPoint
from scope.dataset.US4K import US4KPoint
from scope.dataset.gtasfm import GTASFMPoint
from scope.dataset.IRS import IRSPoint
from scope.dataset.midair import MidAirPoint

from scope.train.losses import (
    get_flat_metrics,
    compute_losses,
)
from scope.train.utils import build_optimizer, build_lr_scheduler, adjust_config
from scope.utils.geometry_torch import intrinsics_to_fov
from scope.utils.checkpoints import SCOPE_CHECKPOINT_REPO_ID, resolve_checkpoint_path
from scope.dataset.transform import generate_pointmap
from scope.utils.tools import key_average, recursive_replace, CallbackOnException, flatten_nested_dict
from scope.test.metrics import compute_metrics
DEFAULT_CHECKPOINT = "auto"


def _step_model_checkpoint(workspace: str, step: int) -> Path:
    return Path(workspace, 'checkpoint', f'step_{step:08d}.pt')


def _step_optimizer_checkpoint(workspace: str, step: int) -> Path:
    return Path(workspace, 'checkpoint', f'step_{step:08d}_optimizer.pt')


def _step_ema_checkpoint(workspace: str, step: int) -> Path:
    return Path(workspace, 'checkpoint', f'step_{step:08d}_ema.pt')


def _first_existing_path(*paths: Path) -> Optional[Path]:
    return next((path for path in paths if path.exists()), None)


SMOKE_TRAIN_CONFIG = {
    "model_name": "scope",
    "model": {
        "encoder": "dinov2_vitl14",
        "remap_output": "exp",
        "intermediate_layers": 4,
        "dim_upsample": [256, 128, 64],
        "dim_times_res_block_hidden": 2,
        "num_res_blocks": 2,
        "num_tokens_range": [16, 16],
        "last_conv_channels": 32,
        "last_conv_size": 1,
        "num_frames": 24,
        "pe": "rope",
        "pe_stretch_prob": 0.5,
    },
    "optimizer": {
        "type": "AdamW",
        "params": [
            {"params": {"include": ["*"], "exclude": ["*backbone.*"]}, "lr": 1e-4},
            {"params": {"include": ["*backbone.*"]}, "lr": 1e-5},
        ],
    },
    "lr_scheduler": {
        "type": "SequentialLR",
        "params": {
            "schedulers": [
                {"type": "LambdaLR", "params": {"lr_lambda": ["1.0", "max(0.0, min(1.0, (epoch - 1000) / 1000))"]}},
                {"type": "StepLR", "params": {"step_size": 25000, "gamma": 0.5}},
            ],
            "milestones": [2000],
        },
    },
    "low_resolution_training_steps": 50000,
    "loss": {
        "synthetic": {
            "global": {
                "function": "affine_invariant_global_loss",
                "weight": 1.0,
                "params": {
                    "align_resolution": 16,
                    "beta": 0.0,
                    "trunc": 1.0,
                    "sparsity_aware": False,
                    "align_method": "roe",
                    "use_downsample": True,
                    "video_align": False,
                    "shift_mode": "z_only",
                },
            },
            "patch_4": {
                "function": "affine_invariant_local_loss",
                "weight": 1.0,
                "params": {
                    "level": 4,
                    "align_resolution": 8,
                    "num_patches": 4,
                    "beta": 0.0,
                    "trunc": 1.0,
                    "sparsity_aware": False,
                    "align_method": "roe",
                    "shift_mode": "xyz",
                },
            },
            "mask": {
                "function": "mask_l2_loss",
                "weight": 1.0,
            },
        }
    },
}


def get_loss_value(loss):
    return loss.item() if hasattr(loss, 'item') else float(loss)

@click.command()
@click.option('--config', 'config_path', type=str, default='configs/train/scope.json')
@click.option('--workspace', type=str, default='workspace/scope_train', help='Path to the workspace')
@click.option('--checkpoint', 'checkpoint_path', type=str, default=DEFAULT_CHECKPOINT, help=f'Path to a pretrained checkpoint, Hugging Face repo id, "auto" to use/download the released checkpoint from {SCOPE_CHECKPOINT_REPO_ID}, "latest" to resume, or a step number.')
@click.option('--batch_size_forward', type=int, default=1, help='Batch size for each forward pass on each device')
@click.option('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
@click.option('--enable_gradient_checkpointing', type=bool, default=True, help='Use gradient checkpointing in backbone')
@click.option('--enable_mixed_precision', type=bool, default=False, help='Use mixed precision training. Backbone is converted to FP16')
@click.option('--enable_ema', type=bool, default=True, help='Maintain an exponential moving average of the model weights')
@click.option('--epochs', type=int, default=10, help='Number of epochs to train the model')
@click.option('--save_every', type=int, default=3000, help='Save checkpoint every n iterations')
@click.option('--log_every', type=int, default=1000, help='Log metrics every n iterations')
@click.option('--vis_every', type=int, default=0, help='Visualize every n iterations')
@click.option('--num_vis_images', type=int, default=3, help='Number of images to visualize, must be a multiple of divided batch size')
@click.option('--enable_mlflow', type=bool, default=True, help='Log metrics to MLFlow')
@click.option('--seed', type=int, default=0, help='Random seed')
@click.option('--images_per_sample', type=int, default=16, help='Number of images per sample')
@click.option('--cj_p', type=float, default=0.5, help='Probability of applying color jitter')
@click.option('--cj_s', type=float, default=13.0, help='Strength of color jitter')
@click.option('--g_p', type=float, default=0.5, help='Probability of applying gaussian noise')
@click.option('--g_s', type=float, default=13.0, help='Strength of gaussian noise')
@click.option('--sample_interval', type=int, default=3, help='Sample interval for datasets')
@click.option('--img_size', type=int, default=728, help='Image size for training')
@click.option('--smoke-test', is_flag=True, help='Run a single synthetic training step without external datasets.')
def main(
    config_path: str,
    workspace: str,
    checkpoint_path: str,
    batch_size_forward: int,
    gradient_accumulation_steps: int,
    enable_gradient_checkpointing: bool,
    enable_mixed_precision: bool,
    enable_ema: bool,
    epochs: int,
    save_every: int,
    log_every: int,
    vis_every: int,
    num_vis_images: int,
    enable_mlflow: bool,
    seed: Optional[int],
    images_per_sample: int,
    cj_p: float,
    cj_s: float,
    g_p: float,
    g_s: float,
    sample_interval: int,
    img_size: int,
    smoke_test: bool,
):
    if smoke_test and torch.cuda.is_available():
        enable_mixed_precision = True

    # Load config
    if smoke_test:
        config = json.loads(json.dumps(SMOKE_TRAIN_CONFIG))
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
    dataset_roots = config.get('data', {}).get('dataset_roots', {})
    dataset_metadata = config.get('data', {}).get('metadata', {})

    def _dataset_root(name: str) -> Optional[str]:
        value = dataset_roots.get(name)
        return value if value else None

    def _dataset_metadata(name: str, key: str, default=None):
        return dataset_metadata.get(name, {}).get(key, default)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='fp16' if enable_mixed_precision else None,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ]
    )
    device = accelerator.device
    batch_size_total = batch_size_forward * gradient_accumulation_steps * accelerator.num_processes

    # Log config
    if accelerator.is_main_process:
        if enable_mlflow:
            try:
                mlflow.log_params({
                    **click.get_current_context().params,
                    'batch_size_total': batch_size_total,
                })
            except:
                print('Failed to log config to MLFlow')
        Path(workspace).mkdir(parents=True, exist_ok=True)
        with Path(workspace).joinpath('config.json').open('w') as f:
            json.dump(config, f, indent=4)
        writer = SummaryWriter(workspace)

    # Set seed
    if seed is not None:
        set_seed(seed, device_specific=True)

    # Initialize model
    print('Initialize model')
    with accelerator.local_main_process_first():
        from scope.model import import_model_class, normalize_model_name
        model_name = normalize_model_name(config.get('model_name', 'scope'))
        config['model_name'] = model_name
        ScopeModel = import_model_class(model_name)
        model = ScopeModel(**config['model'])
    count_total_parameters = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {count_total_parameters}')

    # Set up EMA model
    if enable_ema and accelerator.is_main_process:
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: 0.999 * averaged_model_parameter + 0.001 * model_parameter
        ema_model = torch.optim.swa_utils.AveragedModel(model, device=accelerator.device, avg_fn=ema_avg_fn)

    # Set gradient checkpointing
    if enable_gradient_checkpointing:
        model.enable_gradient_checkpointing()
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
    
    # Initialize custom datasets
    size = (img_size, img_size)
    datasets = []

    def _build_smoke_batch(batch_size: int, frames_per_sample: int, image_size: int, device: torch.device):
        h = w = image_size
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=device),
            torch.linspace(0.0, 1.0, w, device=device),
            indexing='ij',
        )
        base_depth = 1.0 + 0.25 * xx + 0.15 * yy
        image_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        image_frames = []
        pointmaps = []
        valid_masks = []
        sky_masks = []
        intrinsics = []
        camera_poses = []
        for b in range(batch_size):
            frame_stack = []
            point_stack = []
            valid_stack = []
            sky_stack = []
            intr_stack = []
            pose_stack = []
            for t in range(frames_per_sample):
                r = torch.remainder(xx + 0.07 * (b + t), 1.0)
                g = torch.remainder(yy + 0.05 * (b + t), 1.0)
                bch = torch.remainder(0.5 * xx + 0.5 * yy + 0.03 * t, 1.0)
                frame = torch.stack([r, g, bch], dim=0).float()
                frame = (frame - image_mean) / image_std
                depth = base_depth + 0.03 * t + 0.02 * b
                K = torch.tensor(
                    [[1.25, 0.0, 0.5], [0.0, 1.25, 0.5], [0.0, 0.0, 1.0]],
                    device=device,
                    dtype=torch.float32,
                )
                point_map = generate_pointmap(depth, K)
                valid = depth > 0
                sky = torch.zeros_like(valid)
                frame_stack.append(frame)
                point_stack.append(point_map)
                valid_stack.append(valid)
                sky_stack.append(sky)
                intr_stack.append(K)
                pose_stack.append(torch.eye(4, device=device, dtype=torch.float32))
            image_frames.append(torch.stack(frame_stack, dim=0))
            pointmaps.append(torch.stack(point_stack, dim=0))
            valid_masks.append(torch.stack(valid_stack, dim=0))
            sky_masks.append(torch.stack(sky_stack, dim=0))
            intrinsics.append(torch.stack(intr_stack, dim=0))
            camera_poses.append(torch.stack(pose_stack, dim=0))

        batch = {
            'image': torch.stack(image_frames, dim=0),
            'pointmap': torch.stack(pointmaps, dim=0),
            'valid_mask': torch.stack(valid_masks, dim=0),
            'sky_mask': torch.stack(sky_masks, dim=0),
            'intrinsics': torch.stack(intrinsics, dim=0),
            'camera_poses': torch.stack(camera_poses, dim=0),
        }
        batch['depth'] = batch['pointmap'][:, :, 2]
        return batch

    if smoke_test:
        print('Running smoke-test training path')
        datasets = []
        batch = _build_smoke_batch(
            batch_size=batch_size_forward,
            frames_per_sample=config['model']['num_frames'],
            image_size=56,
            device=device,
        )
        train_loader = [batch]
        config = adjust_config(
            config=config,
            datasets=[type('SmokeDataset', (), {'__len__': lambda self: 1})()],
            batch_size_forward=batch_size_forward,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_processes=accelerator.num_processes,
            images_per_sample=config['model']['num_frames'],
            epochs=1,
        )
    else:
        datasets = []

    if not smoke_test:
        # TartanAirPoint dataset
        tartan_dataset = TartanAirPoint(
            filelist_path='scope/dataset/splits/tartanair.txt',
            mode='train',
            images_per_sample=images_per_sample,
            size=size,
            sample_interval=(sample_interval * 3),
            duplicate_times=1,
            disparity=False,
            cj_p=cj_p, cj_s=cj_s, g_p=g_p, g_s=g_s,
        )
        datasets.append(tartan_dataset)

        # PointOdysseyPoint dataset
        pointodyssey_dataset = PointOdysseyPoint(
            filelist_path='scope/dataset/splits/pointodyssey.txt',
            mode='train',
            images_per_sample=images_per_sample,
            size=size,
            sample_interval=sample_interval,
            duplicate_times=1,
            disparity=False,
            cj_p=cj_p, cj_s=cj_s, g_p=g_p, g_s=g_s,
        )
        datasets.append(pointodyssey_dataset)

        # SpringPoint dataset
        spring_dataset = SpringPoint(
            filelist_path='scope/dataset/splits/spring.txt',
            mode='train',
            images_per_sample=images_per_sample,
            size=size,
            sample_interval=2,
            duplicate_times=6,
            disparity=False,
            cj_p=cj_p, cj_s=cj_s, g_p=g_p, g_s=g_s,
            cam_data_base=_dataset_metadata('Spring', 'cam_data_base'),
        )
        datasets.append(spring_dataset)

        # VKitti2Point dataset
        vkitti_dataset = VKitti2Point(
            filelist_path='scope/dataset/splits/vkitti.txt',
            mode='train',
            images_per_sample=images_per_sample,
            size=size,
            sample_interval=sample_interval,
            duplicate_times=1,
            disparity=False,
            cj_p=cj_p, cj_s=cj_s, g_p=g_p, g_s=g_s,
        )
        datasets.append(vkitti_dataset)

        # LightWheelPoint dataset
        lightwheel_dataset = LightWheelPoint(
            filelist_path='scope/dataset/splits/lightwheel.txt',
            mode='train',
            images_per_sample=images_per_sample,
            size=size,
            sample_interval=sample_interval,
            duplicate_times=1,
            disparity=False,
            cj_p=cj_p, cj_s=cj_s, g_p=g_p, g_s=g_s,
            info_pickle_paths=_dataset_metadata('LightWheel', 'info_pickle_paths', []),
        )
        datasets.append(lightwheel_dataset)

        # HypersimPoint dataset
        hypersim_dataset = HypersimPoint(
            filelist_path='scope/dataset/splits/hypersim/all.txt',
            mode='train',
            images_per_sample=images_per_sample,
            size=size,
            sample_interval=1,
            duplicate_times=1,
            disparity=False,
            cj_p=cj_p, cj_s=cj_s, g_p=g_p, g_s=g_s,
            metadata_root=_dataset_root('Hypersim'),
        )
        datasets.append(hypersim_dataset)

        # GTAIMPoint dataset
        gtaim_dataset = GTAIMPoint(
            filelist_path='scope/dataset/splits/GTAIM.txt',
            mode='train',
            images_per_sample=images_per_sample,
            size=size,
            sample_interval=(sample_interval * 2),
            duplicate_times=1,
            disparity=False,
            cj_p=cj_p, cj_s=cj_s, g_p=g_p, g_s=g_s,
        )
        datasets.append(gtaim_dataset)

        # MVSSynthPoint dataset
        mvssynth_dataset = MVSSynthPoint(
            filelist_path='scope/dataset/splits/mvssynth.txt',
            mode='train',
            images_per_sample=images_per_sample,
            size=size,
            sample_interval=sample_interval,
            duplicate_times=1,
            disparity=False,
            cj_p=cj_p, cj_s=cj_s, g_p=g_p, g_s=g_s,
        )
        datasets.append(mvssynth_dataset)

        # US4KPoint dataset
        us4k_dataset = US4KPoint(
            filelist_path='scope/dataset/splits/US4k.txt',
            mode='train',
            images_per_sample=images_per_sample,
            size=size,
            sample_interval=(sample_interval * 2),
            duplicate_times=1,
            disparity=False,
            cj_p=cj_p, cj_s=cj_s, g_p=g_p, g_s=g_s,
        )
        datasets.append(us4k_dataset)

        # GTASFMPoint dataset
        gtasfm_dataset = GTASFMPoint(
            data_dir=_dataset_root('GTASFM'),
            mode='train',
            images_per_sample=images_per_sample,
            size=size,
            sample_interval=1,
            duplicate_times=1,
            disparity=False,
            cj_p=cj_p, cj_s=cj_s, g_p=g_p, g_s=g_s,
        )
        datasets.append(gtasfm_dataset)

        irs_dataset = IRSPoint(
            filelist_path='scope/dataset/splits/IRS.txt',
            mode='train',
            images_per_sample=images_per_sample,
            size=size,
            sample_interval=(sample_interval * 6),
            duplicate_times=1,
            disparity=False,
            cj_p=cj_p, cj_s=cj_s, g_p=g_p, g_s=g_s,
            dataset_root=_dataset_root('IRS'),
        )
        datasets.append(irs_dataset)

        midair_dataset = MidAirPoint(
            filelist_path='scope/dataset/splits/midair.txt',
            mode='train',
            images_per_sample=images_per_sample,
            size=size,
            sample_interval=(sample_interval * 7),
            duplicate_times=1,
            disparity=False,
            cj_p=cj_p, cj_s=cj_s, g_p=g_p, g_s=g_s,
        )
        datasets.append(midair_dataset)
    
    if not smoke_test:
        # Create DataLoader with combined dataset
        train_dataset = ConcatDataset(datasets)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_forward,
            shuffle=True,
            num_workers=32,
            pin_memory=False,
            drop_last=False
        )
        print(f"accelerator.num_processes: {accelerator.num_processes}")
        config = adjust_config(
            config=config, 
            datasets=datasets,
            batch_size_forward=batch_size_forward,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_processes=accelerator.num_processes,
            images_per_sample=images_per_sample,
            epochs=epochs
        )
    else:
        train_loader = [batch]
        print(f"accelerator.num_processes: {accelerator.num_processes}")
    
    # Initalize optimizer & lr scheduler
    optimizer = build_optimizer(model, config['optimizer'])
    lr_scheduler = build_lr_scheduler(optimizer, config['lr_scheduler'])

    count_grouped_parameters = [sum(p.numel() for p in param_group['params'] if p.requires_grad) for param_group in optimizer.param_groups]
    for i, count in enumerate(count_grouped_parameters):
        print(f'- Group {i}: {count} parameters')

    # Attempt to load checkpoint
    checkpoint: Dict[str, Any]
    with accelerator.local_main_process_first():
        ckpt_str = str(checkpoint_path) if checkpoint_path is not None else ''
        if checkpoint_path is None or ckpt_str == '' or ckpt_str.lower() in {'none', 'scratch'}:
            checkpoint = None
        elif ckpt_str.lower() == 'auto':
            checkpoint_file = resolve_checkpoint_path(ckpt_str)
            checkpoint_path = str(checkpoint_file)
            print(f'Load checkpoint: {checkpoint_file}')
            checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
        elif ckpt_str == 'latest':
            checkpoint_file = Path(workspace, 'checkpoint', 'latest.pt')
            if checkpoint_file.exists():
                print(f'Load checkpoint: {checkpoint_file}')
                checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
                i_step = checkpoint['step']
                checkpoint_model_path = _first_existing_path(
                    _step_model_checkpoint(workspace, i_step),
                    Path(workspace, 'checkpoint', f'{i_step:08d}.pt'),
                )
                if 'model' not in checkpoint and checkpoint_model_path is not None:
                    print(f'Load model checkpoint: {checkpoint_model_path}')
                    checkpoint['model'] = torch.load(checkpoint_model_path, map_location='cpu', weights_only=True)['model']
                checkpoint_optimizer_path = _first_existing_path(
                    _step_optimizer_checkpoint(workspace, i_step),
                    Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt'),
                )
                if 'optimizer' not in checkpoint and checkpoint_optimizer_path is not None:
                    print(f'Load optimizer checkpoint: {checkpoint_optimizer_path}')
                    checkpoint.update(torch.load(checkpoint_optimizer_path, map_location='cpu', weights_only=True))
                if enable_ema and accelerator.is_main_process:
                    checkpoint_ema_model_path = _first_existing_path(
                        _step_ema_checkpoint(workspace, i_step),
                        Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt'),
                    )
                    if 'ema_model' not in checkpoint and checkpoint_ema_model_path is not None:
                        print(f'Load EMA model checkpoint: {checkpoint_ema_model_path}')
                        checkpoint['ema_model'] = torch.load(checkpoint_ema_model_path, map_location='cpu', weights_only=True)['model']
            else:
                checkpoint = None
        elif ckpt_str.endswith('.pt'):
            # - Load specific checkpoint file
            checkpoint_file = resolve_checkpoint_path(ckpt_str)
            checkpoint_path = str(checkpoint_file)
            print(f'Load checkpoint: {checkpoint_file}')
            checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
        elif '/' in ckpt_str:
            checkpoint_file = resolve_checkpoint_path(ckpt_str)
            checkpoint_path = str(checkpoint_file)
            print(f'Load checkpoint: {checkpoint_file}')
            checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
        elif checkpoint_path is not None:
            # - Load by step number
            i_step = int(ckpt_str)
            checkpoint = {'step': i_step}
            checkpoint_model_path = _first_existing_path(
                _step_model_checkpoint(workspace, i_step),
                Path(workspace, 'checkpoint', f'{i_step:08d}.pt'),
            )
            if checkpoint_model_path is not None:
                print(f'Load model checkpoint: {checkpoint_model_path}')
                checkpoint['model'] = torch.load(checkpoint_model_path, map_location='cpu', weights_only=True)['model']
            checkpoint_optimizer_path = _first_existing_path(
                _step_optimizer_checkpoint(workspace, i_step),
                Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt'),
            )
            if checkpoint_optimizer_path is not None:
                print(f'Load optimizer checkpoint: {checkpoint_optimizer_path}')
                checkpoint.update(torch.load(checkpoint_optimizer_path, map_location='cpu', weights_only=True))
            if enable_ema and accelerator.is_main_process:
                checkpoint_ema_model_path = _first_existing_path(
                    _step_ema_checkpoint(workspace, i_step),
                    Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt'),
                )
                if checkpoint_ema_model_path is not None:
                    print(f'Load EMA model checkpoint: {checkpoint_ema_model_path}')
                    checkpoint['ema_model'] = torch.load(checkpoint_ema_model_path, map_location='cpu', weights_only=True)['model']
        else:
            checkpoint = None

    if checkpoint is None:
        # Initialize model weights
        print('Initialize model weights')
        with accelerator.local_main_process_first():
            model.init_weights()
        initial_step = 0
    else:
        model.load_state_dict(checkpoint['model'], strict=False)
        if str(checkpoint_path).endswith('.pt'):
            initial_step = 0
        elif 'step' in checkpoint:
            initial_step = checkpoint['step'] + 1
        else:
            initial_step = 0
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if enable_ema and accelerator.is_main_process and 'ema_model' in checkpoint:
            ema_model.module.load_state_dict(checkpoint['ema_model'], strict=False)
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        del checkpoint
        
    # Set initial epoch based on initial_step
    initial_epoch = 0 if smoke_test else initial_step // len(train_loader)
    for dataset in datasets:
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(initial_epoch)
    
    # Prepare with accelerator
    if smoke_test:
        model, optimizer = accelerator.prepare(model, optimizer)
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    num_iterations = 1 if smoke_test else epochs * len(train_loader)
    
    # CRITICAL FIX: Set static graph for DDP
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        print(f"Process {accelerator.process_index}: Setting static graph for DDP")
        model._set_static_graph()

    # Register communication hook for ROCm
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        try:
            from scope.model.utils import sync_ddp_hook
            model.register_comm_hook(None, sync_ddp_hook)
        except:
            print("Could not register custom sync_ddp_hook")
        
    def _write_bytes_retry_loop(save_path: Path, data: bytes):
        while True:
            try:
                save_path.write_bytes(data)
                break
            except Exception as e:
                print('Error while saving checkpoint, retrying in 1 minute: ', e)
                time.sleep(60)

    # Ready to train
    records = []
    model.train()
    
    # Get some batches for visualization
    if accelerator.is_main_process and vis_every > 0:
        batches_for_vis = []
        num_vis_images = min(num_vis_images, len(train_loader) * batch_size_forward)
        num_vis_images = num_vis_images // batch_size_forward * batch_size_forward
        train_iter = iter(train_loader)
        for _ in range(num_vis_images // batch_size_forward):
            try:
                batch = next(train_iter)
                batches_for_vis.append(batch)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
                batches_for_vis.append(batch)

        
    # Reset seed to avoid training on the same data when resuming training
    if seed is not None:
        set_seed(seed + initial_step, device_specific=True)

    # Training loop
    with tqdm(initial=initial_step, total=num_iterations, desc='Training', disable=not accelerator.is_main_process) as pbar, \
         ThreadPoolExecutor(max_workers=1) as save_checkpoint_executor:
        
        i_step = initial_step
        for epoch in range(initial_epoch, 1 if smoke_test else epochs):
            if not smoke_test:
                # Update datasets with current epoch
                for dataset in datasets:
                    if hasattr(dataset, 'set_epoch'):
                        dataset.set_epoch(epoch)

                train_iter = iter(train_loader)
            else:
                train_iter = iter(train_loader)

            for batch_idx in range(len(train_loader)):
                if i_step >= num_iterations:
                    break

                # Load batch
                try:
                    sample = next(train_iter)
                except StopIteration:
                    if smoke_test:
                        break
                    train_iter = iter(train_loader)
                    sample = next(train_iter)

                if smoke_test:
                    img = sample['image']
                    pointmap = sample['pointmap'].permute(0, 1, 3, 4, 2)
                    valid_mask = sample['valid_mask']
                    sky_mask = sample['sky_mask']
                    intrinsics = sample['intrinsics']
                    camera_poses = sample.get('camera_poses', None)
                    gt_focal = 1 / (1 / (intrinsics[..., 0, 0] / (2 * intrinsics[..., 0, 2])) ** 2 + 1 / (intrinsics[..., 1, 1] / (2 * intrinsics[..., 1, 2])) ** 2) ** 0.5
                else:
                    img, depth, valid_mask = sample['image'], sample['depth'], sample['valid_mask']
                    sky_mask, pointmap = sample['sky_mask'], sample['pointmap'].permute(0, 1, 3, 4, 2)
                    intrinsics = sample['intrinsics']
                    camera_poses = sample.get('camera_poses', None)
                    gt_focal = 1 / (1 / (intrinsics[..., 0, 0] / (2 * intrinsics[..., 0, 2])) ** 2 + 1 / (intrinsics[..., 1, 1] / (2 * intrinsics[..., 1, 2])) ** 2) ** 0.5
                
                with accelerator.accumulate(model):
                    # Forward
                    num_tokens = config['model']['num_tokens_range'][0]
                    if i_step <= config.get('low_resolution_training_steps', 0):
                        num_tokens = config['model']['num_tokens_range'][0]
                    else:
                        num_tokens = accelerate.utils.broadcast_object_list([random.randint(*config['model']['num_tokens_range'])])[0]
                    
                    # with torch.autocast(device_type=accelerator.device.type, dtype=torch.float16, enabled=enable_mixed_precision):
                    with accelerator.autocast():
                        output = model(img, num_tokens=num_tokens)
                    pred_points, pred_mask = output['points'], output['mask']

                    total_loss, loss_dict, misc_dict, gt_metric_scale, gt_metric_shift = compute_losses(
                        config,
                        pred_points,
                        pred_mask,
                        pointmap,
                        valid_mask,
                        sky_mask,
                        gt_focal,
                        camera_poses,
                        loss_category="synthetic",
                        i_step=i_step,
                        num_iterations=num_iterations,
                    )
                    if accelerator.is_main_process and writer is not None:
                        writer.add_scalar('train/loss', get_loss_value(total_loss), i_step)
                        writer.add_scalar('train/num_tokens', get_loss_value(num_tokens), i_step)
                        writer.add_scalar('train/pred_min', get_loss_value(pred_points[..., 2].min()), i_step)
                        writer.add_scalar('train/pred_max', get_loss_value(pred_points[..., 2].max()), i_step)
                        
                        # Log the new loss components
                        for k, v in loss_dict.items():
                            writer.add_scalar(f'train/loss_{k}', get_loss_value(v), i_step)
                        
                        # Log the metrics
                        for k, v in misc_dict.items():
                            if isinstance(v, dict):
                                for subk, subv in v.items():
                                    if isinstance(subv, (int, float, bool, torch.Tensor)):
                                        writer.add_scalar(f'train/metric_{k}_{subk}', get_loss_value(subv) if isinstance(subv, torch.Tensor) else subv, i_step)
                    
                    # Handle NaN loss
                    if torch.isnan(total_loss).item():
                        accelerator.print(f'NaN loss detected, skipping update')
                        optimizer.zero_grad()
                        continue
         
                    # Backward & update
                    accelerator.backward(total_loss)
                    if accelerator.sync_gradients:
                        if not enable_mixed_precision and any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
                            print(pred_points.min(), pred_points.max(), pred_points[..., 2].min(), pred_points[..., 2].max())
                            if accelerator.is_main_process:
                                pbar.write(f'NaN gradients, skip update')
                            optimizer.zero_grad()
                            continue
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                            
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()

                # EMA update            
                if enable_ema and accelerator.is_main_process and accelerator.sync_gradients:
                    ema_model.update_parameters(model)

                # Prepare metrics for logging using the helper function
                flat_metrics = get_flat_metrics(loss_dict, misc_dict)
                flat_metrics['total_loss'] = total_loss.item()
                        
                # Record metrics
                records.append(flat_metrics)

                # Log metrics
                if i_step == initial_step or i_step % log_every == 0:
                    records = [key_average(records)]
                    # records = accelerator.gather_for_metrics(records, use_gather_object=True)
                    if accelerator.is_main_process:
                        records = key_average(records)
                        if enable_mlflow:
                            try:
                                mlflow.log_metrics(records, step=i_step)
                            except Exception as e:
                                print(f'Error while logging metrics to mlflow: {e}')
                    records = []

                # Save model weight checkpoint
                if (not smoke_test) and accelerator.is_main_process and (i_step % save_every == 0) and i_step != 0:
                    # NOTE: Writing checkpoint is done in a separate thread to avoid blocking the main process
                    pbar.write(f'Save checkpoint: {i_step:08d}')
                    Path(workspace, 'checkpoint').mkdir(parents=True, exist_ok=True)

                    # Model checkpoint
                    with io.BytesIO() as f:
                        torch.save({
                            'model_config': config['model'],
                            'model': accelerator.unwrap_model(model).state_dict(),
                        }, f)
                        checkpoint_bytes = f.getvalue()
                    save_checkpoint_executor.submit(
                        _write_bytes_retry_loop, _step_model_checkpoint(workspace, i_step), checkpoint_bytes
                    )

                    # Optimizer checkpoint
                    with io.BytesIO() as f:
                        torch.save({
                            'model_config': config['model'],
                            'step': i_step,
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                        }, f)
                        checkpoint_bytes = f.getvalue()
                    save_checkpoint_executor.submit(
                        _write_bytes_retry_loop, _step_optimizer_checkpoint(workspace, i_step), checkpoint_bytes
                    )
                    
                    # EMA model checkpoint
                    if enable_ema:
                        with io.BytesIO() as f:
                            torch.save({
                                'model_config': config['model'],
                                'model': ema_model.module.state_dict(),
                            }, f)
                            checkpoint_bytes = f.getvalue()
                        save_checkpoint_executor.submit(
                            _write_bytes_retry_loop, _step_ema_checkpoint(workspace, i_step), checkpoint_bytes
                        )

                    # Latest checkpoint
                    with io.BytesIO() as f:
                        torch.save({
                            'model_config': config['model'],
                            'step': i_step,
                        }, f)
                        checkpoint_bytes = f.getvalue()
                    save_checkpoint_executor.submit(
                        _write_bytes_retry_loop, Path(workspace, 'checkpoint', 'latest.pt'), checkpoint_bytes
                    )
                
                if (not smoke_test) and accelerator.is_main_process and i_step == num_iterations - 1:
                    final_step = i_step - 1  # Last completed step
                    pbar.write(f'Training completed, saving final model weights: {final_step:08d}')
                    Path(workspace, 'checkpoint').mkdir(parents=True, exist_ok=True)

                    # Save final model weights
                    with io.BytesIO() as f:
                        torch.save({
                            'step': final_step,
                            'model_config': config['model'],
                            'model': accelerator.unwrap_model(model).state_dict(),
                        }, f)
                        checkpoint_bytes = f.getvalue()
                    save_checkpoint_executor.submit(
                        _write_bytes_retry_loop, Path(workspace, 'checkpoint', 'checkpoint.pt'), checkpoint_bytes
                    )
                    
                    # Save final EMA model weights (if enabled)
                    if enable_ema:
                        with io.BytesIO() as f:
                            torch.save({
                                'step': final_step,
                                'model_config': config['model'],
                                'model': ema_model.module.state_dict(),
                            }, f)
                            checkpoint_bytes = f.getvalue()
                        save_checkpoint_executor.submit(
                            _write_bytes_retry_loop, Path(workspace, 'checkpoint', 'checkpoint_ema.pt'), checkpoint_bytes
                        )  

                pbar.set_postfix({'loss': total_loss.item()}, refresh=False)
                pbar.update(1)
                i_step += 1
                if smoke_test:
                    break
            if smoke_test:
                break
                

if __name__ == '__main__':
    main()
