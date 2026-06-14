from typing import *
from numbers import Number
from functools import partial
from pathlib import Path
import importlib
import os
import warnings
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.version
import utils3d
from huggingface_hub import hf_hub_download
from easydict import EasyDict

# from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift
from .utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing
from ..utils.tools import timeit
from .moge import ResidualConvBlock

import torch
import torch.nn.functional as F
import numpy as np
import gc
from tqdm import tqdm
from functools import partial
from typing import Union, List, Dict, Tuple, Optional, Any
from scipy.optimize import least_squares

# Constants for video inference.
#
# IMPORTANT: Keep these configurable so we can sweep long-window settings without
# touching weights.
def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        warnings.warn(f"Invalid int env {name}={val!r}; using default {default}")
        return default


INFER_LEN = _env_int("SCOPE_INFER_LEN", 24)     # frames per chunk
OVERLAP = _env_int("SCOPE_OVERLAP", 8)          # keyframes from previous chunk
INTERP_LEN = _env_int("SCOPE_INTERP_LEN", 4)    # frames to interpolate at boundaries

if not (0 < INTERP_LEN < OVERLAP < INFER_LEN):
    raise ValueError(
        f"Invalid SCOPE inference config: INFER_LEN={INFER_LEN}, OVERLAP={OVERLAP}, INTERP_LEN={INTERP_LEN} "
        "(must satisfy 0 < INTERP_LEN < OVERLAP < INFER_LEN)."
    )

# Keyframes used to stitch chunks:
# - First (OVERLAP-INTERP_LEN): sparse anchors for scale/shift alignment.
# - Last (INTERP_LEN): the last INTERP_LEN frames for interpolation blending.
_align_len = OVERLAP - INTERP_LEN
_head_step = max(1, (INFER_LEN - INTERP_LEN) // _align_len)
KEYFRAMES = [i * _head_step for i in range(_align_len)] + list(range(INFER_LEN - INTERP_LEN, INFER_LEN))

# INFER_LEN = 96
# OVERLAP = 24
# KEYFRAMES = [0, 12, 24, 36, 48, 60, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 95]
# INTERP_LEN = 17


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    """
    Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal
    
    Args:
        uv: Image plane coordinates of shape (..., 2)
        xyz: 3D points of shape (..., 3)
        
    Returns:
        tuple: (shift, focal) optimal parameters
    """
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    """
    Solve `min |focal * xy / (z + shift) - uv|` with respect to shift
    
    Args:
        uv: Image plane coordinates of shape (..., 2)
        xyz: 3D points of shape (..., 3)
        focal: Fixed focal length
        
    Returns:
        float: optimal shift parameter
    """
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    return optim_shift


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, 
                            dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    """
    Create UV coordinates with left-top corner as (-width/diagonal, -height/diagonal) 
    and right-bottom corner as (width/diagonal, height/diagonal)
    
    Args:
        width: Image width
        height: Image height
        aspect_ratio: Optional aspect ratio override
        dtype: Tensor data type
        device: Tensor device
        
    Returns:
        torch.Tensor: Normalized UV coordinates of shape (height, width, 2)
    """
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def image_uv(height: int, width: int, left: int = None, top: int = None, right: int = None, bottom: int = None, device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """
    Get image space UV grid, ranging in [0, 1]. 

    >>> image_uv(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]

    Args:
        width (int): image width
        height (int): image height

    Returns:
        torch.Tensor: shape (height, width, 2)
    """
    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = torch.linspace((left + 0.5) / width, (right - 0.5) / width, right - left, device=device, dtype=dtype)
    v = torch.linspace((top + 0.5) / height, (bottom - 0.5) / height, bottom - top, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def intrinsics_from_focal_center(fx: torch.Tensor, fy: torch.Tensor, 
                               cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
    """
    Create camera intrinsics matrix from focal lengths and principal point
    
    Args:
        fx: Focal length in x direction
        fy: Focal length in y direction
        cx: Principal point x coordinate (typically 0.5)
        cy: Principal point y coordinate (typically 0.5)
        
    Returns:
        torch.Tensor: Camera intrinsics matrix of shape (N, 3, 3)
    """
    N = fx.shape[0]
    device, dtype = fx.device, fx.dtype
    
    zeros = torch.zeros(N, dtype=dtype, device=device)
    ones = torch.ones(N, dtype=dtype, device=device)
    
    # Create 3x3 matrices
    intrinsics = torch.stack([
        fx, zeros, cx,
        zeros, fy, cy,
        zeros, zeros, ones
    ], dim=-1).reshape(N, 3, 3)
    
    return intrinsics


def depth_to_points_batch(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Efficiently convert multiple depth maps to 3D point clouds using given intrinsics
    
    Args:
        depth: Depth maps of shape (B, H, W)
        intrinsics: Camera intrinsics of shape (B, 3, 3)
        
    Returns:
        torch.Tensor: 3D point clouds of shape (B, H, W, 3)
    """
    B, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    
    # Create UV coordinates (shared across batch)
    uv = image_uv(height=H, width=W, device=device, dtype=dtype)  # (H, W, 2)
    
    # Expand UV to homogeneous coordinates
    uv_homogeneous = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1)  # (H, W, 3)
    
    # Reshape for batch processing
    uv_flat = uv_homogeneous.reshape(-1, 3).unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 3)
    
    # Compute inverse of intrinsics matrices
    inv_intrinsics = torch.inverse(intrinsics)  # (B, 3, 3)
    
    # Apply inverse intrinsics
    points_flat = torch.bmm(uv_flat, inv_intrinsics.transpose(1, 2))  # (B, H*W, 3)
    
    # Reshape back to image dimensions
    points = points_flat.reshape(B, H, W, 3)  # (B, H, W, 3)
    
    # Scale by depth
    points = points * depth.unsqueeze(-1)  # (B, H, W, 3)
    
    return points


def recover_focal_shift(points: torch.Tensor, mask: torch.Tensor = None, focal: torch.Tensor = None, downsample_size: Tuple[int, int] = (64, 64)):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: torch.Tensor of shape (...) the estimated focal length, relative to the half diagonal of the map
    - `shift`: torch.Tensor of shape (...) Z-axis shift to translate the point map to camera space
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0
    
    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift


def recover_focal_shift_shared(points: torch.Tensor, mask: torch.Tensor = None, 
                              focal: torch.Tensor = None,
                              downsample_size: Tuple[int, int] = (64, 64)) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recover a shared focal length and shift for all frames in a batch
    
    Args:
        points: Point cloud of shape (T, H, W, 3)
        mask: Optional binary mask of shape (T, H, W)
        focal: Optional fixed focal length
        downsample_size: Size to downsample to for optimization
        
    Returns:
        tuple: (focal, shift) scalars as tensors
    """
    T, height, width = points.shape[0], points.shape[1], points.shape[2]
    device, dtype = points.device, points.dtype
    
    # Create normalized UV coordinates
    uv = normalized_view_plane_uv(width, height, dtype=dtype, device=device)
    
    # Downsample for efficiency
    points_flat = points.reshape(-1, height, width, 3).permute(0, 3, 1, 2)
    points_lr = F.interpolate(points_flat, downsample_size, mode='nearest')
    points_lr = points_lr.permute(0, 2, 3, 1).reshape(T, *downsample_size, 3)
    
    uv_lr = F.interpolate(
        uv.unsqueeze(0).permute(0, 3, 1, 2), 
        downsample_size, 
        mode='nearest'
    ).squeeze(0).permute(1, 2, 0)
    
    if mask is not None:
        mask_lr = F.interpolate(
            mask.unsqueeze(1).float(), 
            downsample_size, 
            mode='nearest'
        ).squeeze(1) > 0.5
    else:
        mask_lr = None
    
    # Convert to numpy for optimization
    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    
    # Handle focal value carefully
    if focal is not None:
        # Convert tensor to scalar value if needed
        if isinstance(focal, torch.Tensor):
            if focal.numel() == 1:  # Single value tensor
                focal_value = focal.item()
            else:
                # If it's a multi-value tensor, take the first value
                focal_value = focal[0].item()
        else:
            # It's already a scalar or other non-tensor type
            focal_value = focal
    else:
        focal_value = None
    
    # Handle mask
    mask_lr_np = None if mask_lr is None else mask_lr.cpu().numpy()
    
    # Collect all valid points from all frames
    all_points = []
    all_uvs = []
    
    for t in range(T):
        if mask_lr is None:
            frame_points = points_lr_np[t].reshape(-1, 3)
            frame_uvs = uv_lr_np.reshape(-1, 2)
        else:
            frame_mask = mask_lr_np[t]
            frame_points = points_lr_np[t][frame_mask]
            frame_uvs = uv_lr_np[frame_mask]
        
        # Only add if we have points
        if frame_points.size > 0:
            all_points.append(frame_points)
            all_uvs.append(frame_uvs)
    
    # Optimize for a single focal and shift for all frames
    if all_points:
        # Concatenate all points
        all_points_np = np.concatenate(all_points, axis=0)
        all_uvs_np = np.concatenate(all_uvs, axis=0)
        
        # Calculate optimal parameters
        try:
            if focal_value is None:
                # Calculate both shift and focal
                optim_shift_val, optim_focal_val = solve_optimal_focal_shift(all_uvs_np, all_points_np)
            else:
                # Calculate just shift with fixed focal
                optim_shift_val = solve_optimal_shift(all_uvs_np, all_points_np, focal_value)
                optim_focal_val = focal_value
        except Exception as e:
            # Fallback on error
            print(f"Optimization error: {e}")
            optim_shift_val = 0.0
            optim_focal_val = 1.0 if focal_value is None else focal_value
    else:
        # Fallback values if no valid points
        optim_shift_val = 0.0
        optim_focal_val = 1.0 if focal_value is None else focal_value
    
    # Ensure we have scalar float values
    optim_shift_val = float(optim_shift_val)
    optim_focal_val = float(optim_focal_val)
    
    # Convert to tensors - explicitly create single-element tensors
    optim_shift_tensor = torch.tensor(optim_shift_val, device=device, dtype=dtype).reshape(1)
    optim_focal_tensor = torch.tensor(optim_focal_val, device=device, dtype=dtype).reshape(1)
    
    return optim_focal_tensor, optim_shift_tensor


def compute_scale_and_shift(prediction: torch.Tensor, target: torch.Tensor, 
                           mask: torch.Tensor, scale_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute optimal scale and shift between prediction and target
    
    Args:
        prediction: Predicted values
        target: Target values
        mask: Binary mask indicating valid regions
        scale_only: Whether to compute only scale (default: False)
        
    Returns:
        tuple: (scale, shift) as tensors
    """
    # Convert to float
    prediction = prediction.float()
    target = target.float()
    mask = mask.float()
    
    # System matrix
    a_00 = torch.sum(mask * prediction * prediction)
    a_01 = torch.sum(mask * prediction)
    a_11 = torch.sum(mask)
    
    # Right hand side
    b_0 = torch.sum(mask * prediction * target)
    b_1 = torch.sum(mask * target)
    
    if scale_only:
        scale = b_0 / (a_00 + 1e-6)
        shift = torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
    else:
        # Solve the system
        det = a_00 * a_11 - a_01 * a_01
        scale = torch.ones_like(det)
        shift = torch.zeros_like(det)
        
        valid = det != 0
        if valid.any():
            scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
            shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    
    return scale, shift


def get_interpolate_frames(frame_list_pre, frame_list_post):
    """
    Interpolate between two sets of frames
    
    Args:
        frame_list_pre: First set of frames
        frame_list_post: Second set of frames
        
    Returns:
        list: Interpolated frames
    """
    assert len(frame_list_pre) == len(frame_list_post)
    min_w = 0.0
    max_w = 1.0
    step = (max_w - min_w) / (len(frame_list_pre) - 1)
    post_w_list = [min_w] + [i * step for i in range(1, len(frame_list_pre) - 1)] + [max_w]

    interpolated_frames = []
    if isinstance(frame_list_pre[0], torch.Tensor):
        weights = torch.tensor(post_w_list, device=frame_list_pre[0].device)
        for i in range(len(frame_list_pre)):
            interpolated_frames.append(
                frame_list_pre[i] * (1 - weights[i]) + frame_list_post[i] * weights[i]
            )
    else:
        for i in range(len(frame_list_pre)):
            interpolated_frames.append(
                frame_list_pre[i] * (1 - post_w_list[i]) + frame_list_post[i] * post_w_list[i]
            )
            
    return interpolated_frames


class HeadTemporal(nn.Module):
    def __init__(
        self, 
        num_features: int,
        dim_in: int, 
        dim_out: List[int], 
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        num_frames: int = 16,
        pe: str = 'ape',
        pe_stretch_prob: float = 0.0, 
    ):
        super().__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=dim_proj, kernel_size=1, stride=1, padding=0) 
            for _ in range(num_features)
        ])

        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_upsampler(in_ch + 2, out_ch),
                *(ResidualConvBlock(out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation="relu", norm=res_block_norm) 
                  for _ in range(num_res_blocks))
            ) for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
        ])

        self.output_block = nn.ModuleList([
            self._make_output_block(
                dim_upsample[-1] + 2, dim_out_, dim_times_res_block_hidden, last_res_blocks, 
                last_conv_channels, last_conv_size, res_block_norm,
            ) for dim_out_ in dim_out
        ])
        
        # Add temporal modules
        from .motion_module.motion_module import TemporalModule
        
        # Configure temporal module parameters
        motion_module_kwargs = EasyDict(
            num_attention_heads=8,
            num_transformer_block=1,
            num_attention_blocks=2,
            temporal_max_len=num_frames,
            zero_initialize=True,
            pos_embedding_type=pe,
            pe_stretch_prob=pe_stretch_prob,  
        )
        
        # Add temporal modules at strategic points similar to VideoDepthAnything
        self.motion_modules = nn.ModuleList([
            # For the lowest resolution feature (after projections)
            TemporalModule(in_channels=dim_proj, **motion_module_kwargs),
            # For intermediate feature after first upsampling
            TemporalModule(in_channels=dim_upsample[0], **motion_module_kwargs),
            # For the higher resolution features after second upsampling
            TemporalModule(in_channels=dim_upsample[1], **motion_module_kwargs),
            # For the highest resolution features after final upsampling
            TemporalModule(in_channels=dim_upsample[2], **motion_module_kwargs)
        ])
    
    def _make_upsampler(self, in_channels: int, out_channels: int):
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        )
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(self, dim_in: int, dim_out: int, dim_times_res_block_hidden: int, 
                          last_res_blocks: int, last_conv_channels: int, last_conv_size: int, 
                          res_block_norm: Literal['group_norm', 'layer_norm']):
        return nn.Sequential(
            nn.Conv2d(dim_in, last_conv_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            *(ResidualConvBlock(last_conv_channels, last_conv_channels, 
                             dim_times_res_block_hidden * last_conv_channels, 
                             activation='relu', norm=res_block_norm) 
              for _ in range(last_res_blocks)),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_conv_channels, dim_out, kernel_size=last_conv_size, 
                    stride=1, padding=last_conv_size // 2, padding_mode='replicate'),
        )
            
    def forward(self, hidden_states: List[Tuple[torch.Tensor, torch.Tensor]], 
               image: torch.Tensor, frame_length: int):
        """
        Process feature vectors across temporal dimension
        
        Args:
            hidden_states: List of (features, cls_token) tuples from backbone
            image: Input image tensor of shape (B*T, C, H, W)
            frame_length: Number of frames T
            
        Returns:
            List of output tensors
        """
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14
        
        # Process the hidden states
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
            for proj, (feat, clstoken) in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)
        
        # Restructure for temporal processing
        B, T = x.shape[0] // frame_length, frame_length
        
        # Apply first temporal module (at lowest resolution)
        x = self.motion_modules[0](
            x.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), 
            None, None
        ).permute(0, 2, 1, 3, 4).flatten(0, 1)
                
        # Upsample stage with temporal modules
        for i, block in enumerate(self.upsample_blocks):
            # Add UV coordinates for aspect ratio awareness
            uv = normalized_view_plane_uv(
                width=x.shape[-1], height=x.shape[-2], 
                aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)
            
            # Apply upsampling block with gradient checkpointing
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            
            # Apply temporal module after upsampling (except for last block)
            if i < len(self.upsample_blocks):
                x = self.motion_modules[i+1](
                    x.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), 
                    None, None
                ).permute(0, 2, 1, 3, 4).flatten(0, 1)
        
        # Final interpolation to image resolution
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        
        # Add UV coordinates
        uv = normalized_view_plane_uv(
            width=x.shape[-1], height=x.shape[-2], 
            aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device
        )
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim=1)

        # Apply output blocks with gradient checkpointing
        if isinstance(self.output_block, nn.ModuleList):
            chunk_size = 80
            if x.shape[0] > chunk_size:
                output = []
                for block in self.output_block:
                    block_outputs = []
                    for i in range(0, x.shape[0], chunk_size):
                        end_idx = min(i + chunk_size, x.shape[0])
                        chunk = x[i:end_idx]
                        chunk_output = torch.utils.checkpoint.checkpoint(block, chunk, use_reentrant=False)
                        block_outputs.append(chunk_output)
                    output.append(torch.cat(block_outputs, dim=0))
            else:
                output = [torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) 
                     for block in self.output_block]
        else:
            chunk_size = 90
            if x.shape[0] > chunk_size:
                outputs = []
                for i in range(0, x.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, x.shape[0])
                    chunk = x[i:end_idx]
                    chunk_output = torch.utils.checkpoint.checkpoint(self.output_block, chunk, use_reentrant=False)
                    outputs.append(chunk_output)
                output = torch.cat(outputs, dim=0)
            else:
                output = torch.utils.checkpoint.checkpoint(self.output_block, x, use_reentrant=False)
        
        return output


class ScopeModel(nn.Module):
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(self, 
        encoder: str = 'dinov2_vitb14', 
        intermediate_layers: Union[int, List[int]] = 4,
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        remap_output: Literal[False, True, 'linear', 'sinh', 'exp', 'sinh_exp'] = 'linear',
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        num_tokens_range: Tuple[Number, Number] = [1200, 2500],
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        mask_threshold: float = 0.5,
        num_frames: int = 16,
        pe: str = 'ape',
        pe_stretch_prob: float = 0.0, 
        **extra_kwargs
    ):
        super(ScopeModel, self).__init__()

        if extra_kwargs:
            if 'trained_area_range' in extra_kwargs:
                num_tokens_range = [extra_kwargs['trained_area_range'][0] // 14 ** 2, 
                                   extra_kwargs['trained_area_range'][1] // 14 ** 2]
                del extra_kwargs['trained_area_range']
            warnings.warn(f"The following extra model arguments are ignored: {extra_kwargs}")

        self.encoder = encoder
        self.remap_output = remap_output
        self.intermediate_layers = intermediate_layers
        self.num_tokens_range = num_tokens_range
        self.mask_threshold = mask_threshold
        self.num_frames = num_frames
        
        # Initialize backbone from DINOv2
        hub_loader = getattr(importlib.import_module(".dinov2.hub.backbones", __package__), encoder)
        self.backbone = hub_loader(pretrained=False)
        dim_feature = self.backbone.blocks[0].attn.qkv.in_features
        
        # Initialize temporal head
        self.head = HeadTemporal(
            num_features=intermediate_layers if isinstance(intermediate_layers, int) 
                        else len(intermediate_layers), 
            dim_in=dim_feature, 
            dim_out=[3, 1], 
            dim_proj=dim_proj,
            dim_upsample=dim_upsample,
            dim_times_res_block_hidden=dim_times_res_block_hidden,
            num_res_blocks=num_res_blocks,
            res_block_norm=res_block_norm,
            last_res_blocks=last_res_blocks,
            last_conv_channels=last_conv_channels,
            last_conv_size=last_conv_size,
            num_frames=num_frames,
            pe=pe,
            pe_stretch_prob=pe_stretch_prob,
        )

        # Register image normalization buffers
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)
        
        # Enable SDPA for newer PyTorch versions
        if torch.__version__ >= '2.0':
            self.enable_pytorch_native_sdpa()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path, IO[bytes]], 
                       model_kwargs: Optional[Dict[str, Any]] = None, **hf_kwargs) -> 'ScopeModel':
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to the checkpoint file or repo id.
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.
        - `hf_kwargs`: additional keyword arguments to pass to the `hf_hub_download` function.

        ### Returns:
        - A new SCOPE video model instance with the parameters loaded from the checkpoint.
        """
        if Path(pretrained_model_name_or_path).exists():
            checkpoint = torch.load(pretrained_model_name_or_path, map_location='cpu', weights_only=True)
        else:
            filename = hf_kwargs.pop("filename", "checkpoint.pt")
            cached_checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename=filename,
                **hf_kwargs
            )
            checkpoint = torch.load(cached_checkpoint_path, map_location='cpu', weights_only=True)
        model_config = checkpoint['model_config']
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model'], strict=False)
        return model

    def init_weights(self):
        "Load the backbone with pretrained dinov2 weights from torch hub"
        state_dict = torch.hub.load('facebookresearch/dinov2', self.encoder, pretrained=True).state_dict()
        self.backbone.load_state_dict(state_dict)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for backbone and temporal modules"""
        # Enable for backbone
        for i in range(len(self.backbone.blocks)):
            self.backbone.blocks[i] = wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])
        
        # Enable for temporal modules
        for i in range(len(self.head.motion_modules)):
            self.head.motion_modules[i] = wrap_module_with_gradient_checkpointing(self.head.motion_modules[i])

    def enable_pytorch_native_sdpa(self):
        """Enable PyTorch 2.0+ SDPA for faster attention computation"""
        for i in range(len(self.backbone.blocks)):
            self.backbone.blocks[i].attn = wrap_dinov2_attention_with_sdpa(self.backbone.blocks[i].attn)
    
    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        """Apply output remapping based on configuration"""
        if self.remap_output == 'linear':
            pass
        elif self.remap_output == 'sinh':
            points = torch.sinh(points)
        elif self.remap_output == 'exp':
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output == 'sinh_exp':
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points
        
    def forward(self, image: torch.Tensor, num_tokens: int) -> Dict[str, torch.Tensor]:
        """
        Forward pass for temporal model
        
        Args:
            image: Input image tensor of shape (B, T, C, H, W)
            num_tokens: Number of tokens to use for inference
            
        Returns:
            Dictionary with 'points' and 'mask' tensors
        """
        # Handle temporal dimension
        B, T, C, original_height, original_width = image.shape
        
        # Resize to expected resolution defined by num_tokens
        resize_factor = ((num_tokens * 14 ** 2) / (original_height * original_width)) ** 0.5
        resized_width = int(original_width * resize_factor)
        resized_height = int(original_height * resize_factor)

        # Reshape to (B*T, C, H, W) for processing
        image = image.flatten(0, 1).contiguous()

        image = F.interpolate(
            image, (resized_height, resized_width), 
            mode="bicubic", align_corners=False, antialias=True
        )
        
        # Ensure height and width are multiples of 14 for patch extraction
        image_14 = F.interpolate(
            image, (resized_height // 14 * 14, resized_width // 14 * 14), 
            mode="bilinear", align_corners=False, antialias=True
        )
        
        # Get intermediate layers from the backbone
        features = self.backbone.get_intermediate_layers(
            image_14, self.intermediate_layers, return_class_token=True
        )

        # Process through temporal head
        output = self.head(features, image, T)
        points, mask = output

        # Ensure fp32 precision for output
        with torch.autocast(device_type=image.device.type, dtype=torch.float32):
            # Resize to original resolution
            points = F.interpolate(
                points, (original_height, original_width), 
                mode='bilinear', align_corners=False, antialias=False
            )
            mask = F.interpolate(
                mask, (original_height, original_width), 
                mode='bilinear', align_corners=False, antialias=False
            )
            
            # Post-process points and mask
            points = points.permute(0, 2, 3, 1)
            mask = mask.squeeze(1)
            
            # Reshape back to (B, T, H, W, C) for points and (B, T, H, W) for mask
            points = points.unflatten(0, (B, T))
            mask = mask.unflatten(0, (B, T))
            
            points = self._remap_points(points)
            
        return_dict = {'points': points, 'mask': mask}
        return return_dict
    
    
    @torch.inference_mode()
    def infer(
        self, 
        frames: torch.Tensor, 
        fov_x: Union[float, torch.Tensor] = None,
        resolution_level: int = 9,
        num_tokens: int = None,
        apply_mask: bool = False,
        force_projection: bool = False,
        use_fp16: bool = False,
        frame_shared_params: bool = True,
        use_common_intrinsics: bool = False,
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Infer 3D point maps from a video
        
        Args:
            frames: Input frames tensor of shape (T, C, H, W) or (1, T, C, H, W)
            fov_x: Optional fixed horizontal field of view in degrees
            resolution_level: Resolution level [0-9] for inference quality
            num_tokens: Alternative to resolution_level, specific number of tokens
            apply_mask: Whether to apply predicted mask to outputs
            force_projection: Whether to enforce projection constraints on pointmap
            use_fp16: Whether to use half precision for inference
            frame_shared_params: Whether all frames in a chunk share intrinsics and shift
            use_common_intrinsics: Whether to use common intrinsics for all frames in final output
            device: Device to run inference on (defaults to frames.device)
        
        Returns:
            dict: Dictionary with 'points', 'depth', 'mask', and 'intrinsics' tensors
        """        
        # Handle device
        if device is None:
            device = frames.device
        
        # Handle input shape
        if frames.dim() == 5:  # (1, T, C, H, W)
            frames = frames.squeeze(0)
        
        # Get dimensions
        org_video_len = frames.shape[0]
        frame_height, frame_width = frames.shape[2:4]
        aspect_ratio = frame_width / frame_height
        
        # Calculate number of tokens based on resolution level if not provided
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))
        
        # MEMORY OPTIMIZATION: Calculate target resize dimensions based on num_tokens
        resize_factor = ((num_tokens * 14 ** 2) / (frame_height * frame_width)) ** 0.5
        resized_height = int(frame_height * resize_factor)
        resized_width = int(frame_width * resize_factor)
        
        # Convert fov_x to tensor if provided
        if fov_x is not None:
            if not isinstance(fov_x, torch.Tensor):
                fov_x = torch.tensor(fov_x, device=device).float()  # Put on target device
                if fov_x.dim() == 0:
                    fov_x = fov_x.unsqueeze(0)
        
        # Calculate padding
        frame_step = INFER_LEN - OVERLAP
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        
        # Create frame list with padding
        frame_list = [frames[i] for i in range(org_video_len)]
        frame_list += [frame_list[-1].clone()] * append_frame_len
        
        # Lists to store results
        points_list = []
        depth_list = []
        mask_list = []
        mask_prob_list = []
        intrinsics_list = []
        
        # Process frames in batches
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, frame_step), desc="Processing frames"):
            # Create batch input (on CPU)
            cur_list = [frame_list[frame_id+i].unsqueeze(0).unsqueeze(0) for i in range(INFER_LEN)]
            cur_input_cpu = torch.cat(cur_list, dim=1)  # [1, T, C, H, W]
            
            # MEMORY OPTIMIZATION: Resize on CPU
            cur_input_resized = F.interpolate(
                cur_input_cpu.flatten(0, 1),  # [T, C, H, W]
                (resized_height, resized_width),
                mode="bicubic", align_corners=False, antialias=True
            ).unflatten(0, cur_input_cpu.shape[:2])  # Back to [1, T, C, H, W]
            
            # Handle overlapping region with previous batch
            if pre_input is not None:
                cur_input_resized[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]
            
            # Now transfer to GPU
            cur_input_gpu = cur_input_resized.to(device)
            
            # Run model inference
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                outputs = self.forward(cur_input_gpu, num_tokens)
            
            # Free GPU memory for input
            del cur_input_gpu
            
            # Extract outputs (keep on GPU for further processing)
            batch_points = outputs['points'][0]  # (T, H, W, 3)
            batch_mask_prob = outputs['mask'][0]  # (T, H, W)
            batch_mask = batch_mask_prob > self.mask_threshold  # (T, H, W) as boolean

            # Calculate shared or per-frame focal and shift (on GPU)
            if frame_shared_params:
                # Shared focal and shift for all frames in batch
                batch_focal, batch_shift = recover_focal_shift_shared(
                    batch_points, 
                    mask=batch_mask, 
                    focal=fov_x,
                    downsample_size=(64, 64)
                )
                # Expand to all frames in batch
                batch_focal = batch_focal.expand(INFER_LEN)
                batch_shift = batch_shift.expand(INFER_LEN)
            else:
                # Per-frame focal and shift calculation
                batch_focal, batch_shift = recover_focal_shift(
                    batch_points, 
                    mask=batch_mask, 
                    focal=fov_x,
                    downsample_size=(64, 64)
                )
            
            # Calculate camera intrinsics (on GPU)
            batch_fx = batch_focal * 0.5 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
            batch_fy = batch_focal * 0.5 * (1 + aspect_ratio ** 2) ** 0.5
            
            batch_intrinsics = intrinsics_from_focal_center(
                fx=batch_fx, 
                fy=batch_fy,
                cx=torch.full_like(batch_fx, 0.5),
                cy=torch.full_like(batch_fy, 0.5)
            )
            
            # Get depth and apply shift (on GPU)
            batch_depth = batch_points[..., 2] + batch_shift.view(-1, 1, 1)
            
            # Apply projection constraint if requested (on GPU)
            if force_projection:
                # Create UV grid matching points resolution
                batch_points = depth_to_points_batch(
                    depth=batch_depth, 
                    intrinsics=batch_intrinsics
                )
            else:
                # Just update z coordinates with shift
                batch_points = torch.cat([
                    batch_points[..., :2],  # x, y unchanged
                    (batch_points[..., 2:] + batch_shift.view(-1, 1, 1, 1))  # z + shift
                ], dim=-1)
            
            # Apply mask if needed (on GPU)
            if apply_mask:
                batch_points = torch.where(batch_mask.unsqueeze(-1), batch_points, torch.zeros_like(batch_points))
                batch_depth = torch.where(batch_mask, batch_depth, torch.zeros_like(batch_depth))
            
            # Save current batch for next iteration's overlap region
            pre_input = cur_input_resized.cpu()
            
            # NOW move to CPU before resize operations
            batch_points_cpu = batch_points.cpu()
            batch_depth_cpu = batch_depth.cpu()
            batch_mask_cpu = batch_mask.cpu()
            batch_mask_prob_cpu = batch_mask_prob.cpu()
            batch_intrinsics_cpu = batch_intrinsics.cpu()
            
            # MEMORY OPTIMIZATION: Resize outputs back to original dimensions if needed
            if resized_height != frame_height or resized_width != frame_width:
                # Process batch before storing
                for t in range(INFER_LEN):
                    # Resize points to original dimensions
                    points_t = batch_points_cpu[t].permute(2, 0, 1)  # [3, H, W]
                    points_t_resized = F.interpolate(
                        points_t.unsqueeze(0), (frame_height, frame_width), 
                        mode='bilinear', align_corners=False
                    ).squeeze(0).permute(1, 2, 0)  # [H, W, 3]
                    
                    # Resize depth
                    depth_t_resized = F.interpolate(
                        batch_depth_cpu[t].unsqueeze(0).unsqueeze(0), (frame_height, frame_width), 
                        mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)
                    
                    # Resize mask prob
                    mask_prob_t_resized = F.interpolate(
                        batch_mask_prob_cpu[t].unsqueeze(0).unsqueeze(0), (frame_height, frame_width), 
                        mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)
                    
                    # Get mask from resized prob
                    mask_t_resized = mask_prob_t_resized > self.mask_threshold
                    
                    # Store resized outputs
                    points_list.append(points_t_resized)
                    depth_list.append(depth_t_resized)
                    mask_list.append(mask_t_resized)
                    mask_prob_list.append(mask_prob_t_resized)
                    intrinsics_list.append(batch_intrinsics_cpu[t])
            else:
                # Original dimensions - no resize needed
                for t in range(INFER_LEN):
                    points_list.append(batch_points_cpu[t])
                    depth_list.append(batch_depth_cpu[t])
                    mask_list.append(batch_mask_cpu[t])
                    mask_prob_list.append(batch_mask_prob_cpu[t])
                    intrinsics_list.append(batch_intrinsics_cpu[t])
            
            # Free GPU memory
            del batch_points, batch_depth, batch_mask, batch_mask_prob, batch_intrinsics
        
        # Clean up memory
        del frame_list, cur_input_resized, pre_input
        gc.collect()
        
        # Rest of the function remains unchanged for alignment process...
        # Alignment process for consistency between batches
        points_list_aligned = []
        depth_list_aligned = []
        mask_list_aligned = []
        mask_prob_list_aligned = []
        intrinsics_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]
        
        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                # First batch - no alignment needed
                depth_list_aligned += depth_list[:INFER_LEN]
                points_list_aligned += points_list[:INFER_LEN]
                mask_list_aligned += mask_list[:INFER_LEN]
                mask_prob_list_aligned += mask_prob_list[:INFER_LEN]
                intrinsics_list_aligned += intrinsics_list[:INFER_LEN]
                
                # Save reference frames for alignment
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id + kf_id])
            else:
                # Get current alignment frames
                curr_align = [depth_list[frame_id + i] for i in range(len(kf_align_list))]
                
                # Compute scale for depth alignment
                # Use masks for valid regions
                curr_masks = [mask_list[frame_id + i] for i in range(len(kf_align_list))]
                
                # Convert masks to float for compute_scale_and_shift
                curr_mask_float = torch.cat([m.float() for m in curr_masks], dim=0)
                ref_mask_float = torch.cat([mask_list[frame_id-INFER_LEN+kf_id].float() for kf_id in kf_align_list], dim=0)
                
                # Make mask that's only valid where both current and reference depths are valid
                combined_mask = curr_mask_float * ref_mask_float * torch.cat(ref_align, dim=0).gt(0).float()
                
                scale, shift = compute_scale_and_shift(
                    torch.cat(curr_align, dim=0),
                    torch.cat(ref_align, dim=0),
                    combined_mask,
                    scale_only=False
                )
                
                # Prepare lists for interpolation
                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = []
                for i in range(align_len, OVERLAP):
                    idx = frame_id + i
                    if idx < len(depth_list):  # Check bounds
                        # Apply scale to depth
                        new_depth = depth_list[idx] * scale + shift
                        new_depth = torch.clamp(new_depth, min=0)
                        post_depth_list.append(new_depth)
                
                pre_points_list = points_list_aligned[-INTERP_LEN:]
                post_points_list = []
                for i in range(align_len, OVERLAP):
                    idx = frame_id + i
                    if idx < len(points_list):  # Check bounds
                        # Apply scale to points
                        xy = points_list[idx][..., :2] * scale
                        z = points_list[idx][..., 2:3] * scale + shift
                        post_points_list.append(torch.cat([xy, z], dim=-1))
                
                pre_mask_list = mask_list_aligned[-INTERP_LEN:]
                post_mask_list = []
                for i in range(align_len, OVERLAP):
                    idx = frame_id + i
                    if idx < len(mask_list):  # Check bounds
                        post_mask_list.append(mask_list[idx])
                        
                pre_mask_prob_list = mask_prob_list_aligned[-INTERP_LEN:]
                post_mask_prob_list = []
                for i in range(align_len, OVERLAP):
                    idx = frame_id + i
                    if idx < len(mask_prob_list):  # Check bounds
                        post_mask_prob_list.append(mask_prob_list[idx])
                                
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)
                points_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_points_list, post_points_list)
                mask_list_aligned[-INTERP_LEN:] = get_interpolate_frames(
                    pre_mask_list, 
                    post_mask_list
                )
                mask_prob_list_aligned[-INTERP_LEN:] = get_interpolate_frames(
                    pre_mask_prob_list,
                    post_mask_prob_list
                )
                
                # Process remaining frames in the current batch
                for i in range(OVERLAP, INFER_LEN):
                    idx = frame_id + i
                    if idx >= len(depth_list):
                        break
                        
                    # Align depth
                    new_depth = depth_list[idx] * scale + shift
                    new_depth = torch.clamp(new_depth, min=0)
                    depth_list_aligned.append(new_depth)
                    
                    # Align points
                    xy = points_list[idx][..., :2] * scale
                    z = points_list[idx][..., 2:3] * scale + shift
                    new_points = torch.cat([xy, z], dim=-1)
                    points_list_aligned.append(new_points)
                    
                    # Add mask and intrinsics (unchanged by alignment)
                    mask_list_aligned.append(mask_list[idx])
                    mask_prob_list_aligned.append(mask_prob_list[idx])
                    intrinsics_list_aligned.append(intrinsics_list[idx])
                
                ref_align = ref_align[:1]  # Keep first reference frame
                for kf_id in kf_align_list[1:]:
                    idx = frame_id + kf_id
                    if idx < len(depth_list):  # Bounds check
                        # Apply scale to create new reference (scale only, no shift for scale-inv points)
                        new_depth = depth_list[idx] * scale + shift
                        new_depth = torch.clamp(new_depth, min=0)
                        ref_align.append(new_depth)

        # Use aligned lists (trim to original frame count)
        depth_list = depth_list_aligned[:org_video_len]
        points_list = points_list_aligned[:org_video_len]
        mask_list = mask_list_aligned[:org_video_len]
        mask_prob_list = mask_prob_list_aligned[:org_video_len]
        intrinsics_list = intrinsics_list_aligned[:org_video_len]
        
        # Apply common intrinsics if requested
        if use_common_intrinsics and len(intrinsics_list) > 0:
            # Move to GPU for faster processing
            gpu_intrinsics = [intr.to(device) for intr in intrinsics_list]
            gpu_depth = torch.stack([depth.to(device) for depth in depth_list])
                        
            # Find median focal lengths
            all_fx = torch.stack([intr[0, 0] for intr in gpu_intrinsics])
            all_fy = torch.stack([intr[1, 1] for intr in gpu_intrinsics])
            
            # Use median values for stability
            median_fx = all_fx.median()
            median_fy = all_fy.median()
            
            # Create common intrinsics
            common_intrinsics = intrinsics_from_focal_center(
                fx=torch.full((org_video_len,), median_fx.item(), device=device),
                fy=torch.full((org_video_len,), median_fy.item(), device=device),
                cx=torch.full((org_video_len,), 0.5, device=device),
                cy=torch.full((org_video_len,), 0.5, device=device)
            )
            
            # Reproject all points using common intrinsics
            reprojected_points = depth_to_points_batch(
                depth=gpu_depth, 
                intrinsics=common_intrinsics
            )
                
            points_list = [p.cpu() for p in reprojected_points]
                
            # Update intrinsics list
            intrinsics_list = [intr.cpu() for intr in common_intrinsics]
        
        # Create final output tensors
        points_tensor = torch.stack(points_list, dim=0)
        depth_tensor = torch.stack(depth_list, dim=0)
        mask_tensor = torch.stack(mask_list, dim=0)
        mask_prob_tensor = torch.stack(mask_prob_list, dim=0)
        intrinsics_tensor = torch.stack(intrinsics_list, dim=0)
        
        # Create return dictionary
        result_dict = {
            'points': points_tensor,
            'depth': depth_tensor,
            'mask': mask_tensor > self.mask_threshold,
            'mask_prob': mask_prob_tensor,
            'intrinsics': intrinsics_tensor
        }
        
        return result_dict
    
    @torch.inference_mode()
    def infer_simple(
        self, 
        frames: torch.Tensor, 
        resolution_level: int = 9,
        use_fp16: bool = False,
        force_projection: bool = False,
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Simplified inference for all frames at once with shared parameters
        
        Args:
            frames: Input frames tensor of shape (T, C, H, W) or (1, T, C, H, W)
            resolution_level: Resolution level [0-9] for inference quality
            use_fp16: Whether to use half precision for inference
            force_projection: Whether to enforce projection constraints on pointmap
            device: Device to run inference on (defaults to frames.device)
        
        Returns:
            dict: Dictionary with 'points', 'depth', 'mask', and 'intrinsics' tensors
        """        
        # Handle device
        if device is None:
            device = frames.device
        
        # Handle input shape
        if frames.dim() == 5:  # (1, T, C, H, W)
            frames = frames.squeeze(0)
        
        # Get dimensions
        num_frames = frames.shape[0]
        frame_height, frame_width = frames.shape[2:4]
        aspect_ratio = frame_width / frame_height
        
        # Calculate number of tokens based on resolution level
        min_tokens, max_tokens = self.num_tokens_range
        num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))
        
        # MEMORY OPTIMIZATION: Resize on CPU before sending to GPU
        # Calculate target size based on num_tokens
        resize_factor = ((num_tokens * 14 ** 2) / (frame_height * frame_width)) ** 0.5
        resized_height = int(frame_height * resize_factor)
        resized_width = int(frame_width * resize_factor)
        
        # Resize on CPU
        resized_frames = F.interpolate(
            frames, (resized_height, resized_width), 
            mode="bicubic", align_corners=False, antialias=True
        )

        adjusted_height = resized_height // 14 * 14
        adjusted_width = resized_width // 14 * 14

        # Now transfer to GPU
        input_frames = resized_frames.unsqueeze(0).to(device)  # [1, T, C, H, W]
        
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        device_idx = device.index if device.type == 'cuda' else 0
        memory_before = torch.cuda.memory_allocated(device_idx) / (1024 * 1024 * 1024)  # GB
        inference_start_time = time.time()
        # Run model inference
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            outputs = self.forward(input_frames, num_tokens)
        torch.cuda.synchronize(device)
        inference_time = time.time() - inference_start_time
        peak_memory_gb = torch.cuda.max_memory_allocated(device_idx) / (1024 * 1024 * 1024)
        memory_used_gb = peak_memory_gb - memory_before
        print(f"GPU {device_idx}: input_frames_shape: ({adjusted_height}, {adjusted_width}), Memory used: {memory_used_gb:.4f} GB")
        print(f"Inference time: {inference_time:.4f} seconds, Average per frame: {inference_time/num_frames:.4f} seconds")

        # Extract outputs (keep on GPU for further processing)
        points = outputs['points'][0]  # (T, H, W, 3)
        mask_prob = outputs['mask'][0]  # (T, H, W)
        mask = mask_prob > self.mask_threshold  # (T, H, W) as boolean
        
        # Calculate shared focal and shift for all frames (on GPU)
        focal, shift = recover_focal_shift_shared(
            points, 
            mask=mask, 
            downsample_size=(64, 64)
        )
        
        # Expand to all frames
        focal = focal.expand(num_frames)
        shift = shift.expand(num_frames)
        
        # Calculate camera intrinsics (on GPU)
        fx = focal * 0.5 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = focal * 0.5 * (1 + aspect_ratio ** 2) ** 0.5
        
        intrinsics = intrinsics_from_focal_center(
            fx=fx, 
            fy=fy,
            cx=torch.full_like(fx, 0.5),
            cy=torch.full_like(fy, 0.5)
        )
        
        # Get depth and apply shift (on GPU)
        depth = points[..., 2] + shift.view(-1, 1, 1)
        
        # Apply projection constraint if requested (on GPU)
        if force_projection:
            # Create points that strictly adhere to projection constraints
            points = depth_to_points_batch(
                depth=depth, 
                intrinsics=intrinsics
            )
        else:
            # Just update z coordinates with shift
            points = torch.cat([
                points[..., :2],  # x, y unchanged
                (points[..., 2:] + shift.view(-1, 1, 1, 1))  # z + shift
            ], dim=-1)
        
        # NOW move results to CPU before resize
        points = points.cpu()
        depth = depth.cpu()
        mask = mask.cpu()
        mask_prob = mask_prob.cpu()
        intrinsics = intrinsics.cpu()
        
        # MEMORY OPTIMIZATION: Resize outputs back to original dimensions
        if resized_height != frame_height or resized_width != frame_width:
            # Resize points
            points_reshaped = points.permute(0, 3, 1, 2)  # [T, 3, H, W]
            points_resized = F.interpolate(
                points_reshaped, (frame_height, frame_width), 
                mode='bilinear', align_corners=False
            )
            points = points_resized.permute(0, 2, 3, 1)  # [T, H, W, 3]
            
            # Resize depth
            depth = F.interpolate(
                depth.unsqueeze(1), (frame_height, frame_width), 
                mode='bilinear', align_corners=False
            ).squeeze(1)
            
            # Resize mask and mask_prob
            mask_prob = F.interpolate(
                mask_prob.unsqueeze(1), (frame_height, frame_width), 
                mode='bilinear', align_corners=False
            ).squeeze(1)
            mask = mask_prob > self.mask_threshold
        
        # Create return dictionary
        result_dict = {
            'points': points,
            'depth': depth,
            'mask': mask,
            'mask_prob': mask_prob,
            'intrinsics': intrinsics
        }
        
        return result_dict
