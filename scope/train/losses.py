from typing import *
import math

import torch
import torch.nn.functional as F
import utils3d
import numpy as np
from ..utils.geometry_torch import (
    weighted_mean, 
    harmonic_mean, 
    geometric_mean,
    mask_aware_nearest_resize,
    normalized_view_plane_uv,
    angle_diff_vec3
)
from ..utils.alignment import (
    align_points_scale_z_shift, 
    align_points_scale, 
    align_points_scale_xyz_shift,
    align_points_z_shift,
    compute_alignment_params,
)


def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def affine_invariant_global_loss(
    pred_points: torch.Tensor,  # [..., H, W, 3]
    gt_points: torch.Tensor,    # [..., H, W, 3]
    mask: torch.Tensor,         # [..., H, W]
    align_resolution: int = 64, 
    beta: float = 0.0, 
    trunc: float = 1.0, 
    sparsity_aware: bool = False,
    align_method: str = 'roe',
    use_downsample: bool = True,
    video_align: bool = False,
    shift_mode: str = 'z_only'  # Added parameter
):
    """
    Compute affine-invariant global loss between predicted and ground truth points.
    
    Args:
        pred_points: predicted points, shape [..., H, W, 3]
        gt_points: ground truth points, shape [..., H, W, 3]
        mask: valid mask, shape [..., H, W]
        align_resolution: resolution for downsampling in alignment
        beta: smoothness parameter for loss
        trunc: truncation for robust alignment
        sparsity_aware: whether to account for sparsity in loss
        align_method: 'roe' or 'median_mad'
        use_downsample: whether to downsample for alignment
        video_align: whether to compute same alignment for all frames in a video
        shift_mode: 'z_only' or 'xyz' for shift dimensionality
    
    Returns:
        loss: tensor of shape [...]
        misc: dictionary of metrics
        scale: tensor of shape [...]
    """
    device = pred_points.device
    
    # Compute alignment parameters (scale, shift)
    scale, shift = compute_alignment_params(
        pred_points, gt_points, mask,
        align_method=align_method,
        video_align=video_align,
        align_resolution=align_resolution,
        use_downsample=use_downsample,
        trunc=trunc,
        shift_mode=shift_mode
    )
    # grad_scale, grad_shift = compute_alignment_params(
    #     pred_points, gt_points, mask,
    #     align_method='median_mad',
    #     video_align=video_align,
    #     align_resolution=align_resolution,
    #     use_downsample=use_downsample,
    #     trunc=trunc,
    #     shift_mode=shift_mode
    # )
    
    # Filter valid alignments
    valid = scale > 0
    scale = torch.where(valid, scale, torch.tensor(0.0, device=device))
    shift = torch.where(valid[..., None], shift, torch.tensor(0.0, device=device))
    
    # Apply alignment to predicted points
    pred_points_aligned = scale[..., None, None, None] * pred_points + shift[..., None, None, :]
    
    # Compute weighted loss
    weight = (valid[..., None, None] & mask).float() / gt_points[..., 2].clamp(min=1e-5)
    weight = weight.clamp(max=10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))
    loss = _smooth((pred_points_aligned - gt_points).abs() * weight[..., None], beta=beta).mean(dim=(-3, -2, -1))
    
    # Sparsity-aware reweighting (optional)
    if sparsity_aware and use_downsample:
        # Calculate downsampled mask if needed for sparsity calculation
        _, ds_mask = mask_aware_nearest_resize(None, mask=mask, size=(align_resolution, align_resolution))
        sparsity = mask.float().mean(dim=(-2, -1)) / ds_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)
    
    # Compute error metrics
    err = (pred_points_aligned.detach() - gt_points).norm(dim=-1) / gt_points[..., 2]
    misc = {
        'truncated_error': weighted_mean(err.clamp(max=1.0), mask).item(),
        'delta': weighted_mean((err < 1).float(), mask).item()
    }
    
    return loss.mean(), misc, scale.detach(), shift.detach(), scale, shift


def monitoring(points: torch.Tensor):
    return {
        'std': points.std().item(),
    }


def compute_anchor_sampling_weight(
    points: torch.Tensor, 
    mask: torch.Tensor, 
    radius_2d: torch.Tensor, 
    radius_3d: torch.Tensor, 
    num_test: int = 64
) -> torch.Tensor:
    # Importance sampling balances the sampled probability of fine structures.

    height, width = points.shape[-3:-1]

    pixel_i, pixel_j = torch.meshgrid(
        torch.arange(height, device=points.device), 
        torch.arange(width, device=points.device),
        indexing='ij'
    )
    
    test_delta_i = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_delta_j = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_i, test_j = pixel_i[..., None] + test_delta_i, pixel_j[..., None] + test_delta_j                       # [height, width, num_test]
    test_mask = (test_i >= 0) & (test_i < height) & (test_j >= 0) & (test_j < width)                            # [height, width, num_test]
    test_i, test_j = test_i.clamp(0, height - 1), test_j.clamp(0, width - 1)                                    # [height, width, num_test]
    test_mask = test_mask & mask[..., test_i, test_j]                                                           # [..., height, width, num_test]
    test_points = points[..., test_i, test_j, :]                                                                # [..., height, width, num_test, 3]
    test_dist = (test_points - points[..., None, :]).norm(dim=-1)                                               # [..., height, width, num_test]

    weight = 1 / ((test_dist <= radius_3d[..., None]) & test_mask).float().sum(dim=-1).clamp_min(1)
    weight = torch.where(mask, weight, 0)
    weight = weight / weight.sum(dim=(-2, -1), keepdim=True).add(1e-7)                                          # [..., height, width]
    return weight


def affine_invariant_local_loss(
    pred_points: torch.Tensor,  # [B, H, W, 3]
    gt_points: torch.Tensor,    # [B, H, W, 3]
    gt_mask: torch.Tensor,      # [B, H, W]
    focal: torch.Tensor,        # [B]
    global_scale: torch.Tensor, # [B] or None
    level: Literal[4, 16, 64],  # scalar
    align_resolution: int = 32, # scalar
    num_patches: int = 16,      # scalar
    beta: float = 0.0,          # scalar
    trunc: float = 1.0,         # scalar
    sparsity_aware: bool = False,# boolean
    align_method: str = 'roe',  # Added parameter with default to maintain compatibility
    shift_mode: str = 'xyz'     # Added parameter with default set to 'xyz' to match original behavior
):
    device, dtype = pred_points.device, pred_points.dtype
    *batch_shape, height, width, _ = pred_points.shape
    batch_size = math.prod(batch_shape)
    # Reshape inputs to have a flat batch dimension
    pred_points = pred_points.reshape(-1, height, width, 3)      # [batch_size, H, W, 3]
    gt_points = gt_points.reshape(-1, height, width, 3)          # [batch_size, H, W, 3]
    gt_mask = gt_mask.reshape(-1, height, width)                 # [batch_size, H, W]
    focal = focal.reshape(-1)                                    # [batch_size]
    global_scale = global_scale.reshape(-1) if global_scale is not None else None  # [batch_size] or None
    
    # Sample patch anchor points indices
    radius_2d = math.ceil(0.5 / level * (height ** 2 + width ** 2) ** 0.5)  # scalar
    
    # Proper broadcasting for focal dimension
    radius_3d = 0.5 / level / focal[:, None, None] * gt_points[..., 2]  # [batch_size, H, W]
    
    # Compute importance sampling weights
    anchor_sampling_weights = compute_anchor_sampling_weight(
        gt_points, gt_mask, radius_2d, radius_3d, num_test=64)  # [batch_size, H, W]
    
    # Find indices of all valid points
    where_mask = torch.where(gt_mask)  # Tuple of 3 tensors: (batch_indices, i_indices, j_indices)
    
    # Check if we have any valid points
    if where_mask[0].shape[0] == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), {}
    
    # Ensure we don't try to sample more points than are available
    actual_num_patches = min(num_patches * batch_size, where_mask[0].shape[0])
    if actual_num_patches == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), {}
    
    # Sample anchor points using weighted sampling
    random_selection = torch.multinomial(
        anchor_sampling_weights[where_mask],  # [num_valid_points]
        actual_num_patches,                   # Number to sample
        replacement=True if actual_num_patches > where_mask[0].shape[0] else False
    )
    
    # Extract batch indices and 2D coordinates for sampled anchors
    patch_batch_idx, patch_anchor_i, patch_anchor_j = [
        indices[random_selection] for indices in where_mask]  # Each is [actual_num_patches]

    # Create grid of relative coordinates for each patch
    patch_i, patch_j = torch.meshgrid(
        torch.arange(-radius_2d, radius_2d + 1, device=device), 
        torch.arange(-radius_2d, radius_2d + 1, device=device),
        indexing='ij'
    )  # Each is [2*radius_2d+1, 2*radius_2d+1]
    
    # Add anchor coordinates to get absolute patch coordinates
    patch_i = patch_i + patch_anchor_i[:, None, None]  # [actual_num_patches, patch_h, patch_w]
    patch_j = patch_j + patch_anchor_j[:, None, None]  # [actual_num_patches, patch_h, patch_w]
    
    # Create mask for valid indices (within image boundaries)
    patch_mask = (patch_i >= 0) & (patch_i < height) & (patch_j >= 0) & (patch_j < width)
    # [actual_num_patches, patch_h, patch_w]
    
    # Clamp indices to valid image bounds
    patch_i = patch_i.clamp(0, height - 1)  # [actual_num_patches, patch_h, patch_w]
    patch_j = patch_j.clamp(0, width - 1)   # [actual_num_patches, patch_h, patch_w]
    
    # Get ground truth 3D coordinates for anchor points
    gt_patch_anchor_points = gt_points[patch_batch_idx, patch_anchor_i, patch_anchor_j]  # [actual_num_patches, 3]
    
    # Calculate 3D radius for each patch using correct batch indexing for focal
    gt_patch_radius_3d = 0.5 / level / focal[patch_batch_idx] * gt_patch_anchor_points[:, 2]  # [actual_num_patches]
    
    # Get ground truth points for all pixels in all patches
    gt_patch_points = gt_points[patch_batch_idx[:, None, None], patch_i, patch_j]  # [actual_num_patches, patch_h, patch_w, 3]
    
    # Calculate 3D distance from each point to its patch anchor
    gt_patch_dist = (gt_patch_points - gt_patch_anchor_points[:, None, None, :]).norm(dim=-1)  # [actual_num_patches, patch_h, patch_w]
    
    # Update mask to include only valid points within 3D radius
    patch_mask &= gt_mask[patch_batch_idx[:, None, None], patch_i, patch_j]  # [actual_num_patches, patch_h, patch_w]
    patch_mask &= gt_patch_dist <= gt_patch_radius_3d[:, None, None]  # [actual_num_patches, patch_h, patch_w]

    # Filter out patches with too few valid points
    MINIMUM_POINTS_PER_PATCH = 32
    nonempty = torch.where(patch_mask.sum(dim=(-2, -1)) >= MINIMUM_POINTS_PER_PATCH)  # [num_nonempty_patches]
    num_nonempty_patches = nonempty[0].shape[0]
    
    if num_nonempty_patches == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), {}
    
    # Keep only non-empty patches
    patch_batch_idx = patch_batch_idx[nonempty]        # [num_nonempty_patches]
    patch_i = patch_i[nonempty]                        # [num_nonempty_patches, patch_h, patch_w]
    patch_j = patch_j[nonempty]                        # [num_nonempty_patches, patch_h, patch_w]
    patch_mask = patch_mask[nonempty]                  # [num_nonempty_patches, patch_h, patch_w]
    gt_patch_points = gt_patch_points[nonempty]        # [num_nonempty_patches, patch_h, patch_w, 3]
    gt_patch_radius_3d = gt_patch_radius_3d[nonempty]  # [num_nonempty_patches]
    gt_patch_anchor_points = gt_patch_anchor_points[nonempty]  # [num_nonempty_patches, 3]
    
    # Get predicted points for all patches
    pred_patch_points = pred_points[patch_batch_idx[:, None, None], patch_i, patch_j]  # [num_nonempty_patches, patch_h, patch_w, 3]
    
    # Use compute_alignment_params for both downsampling and alignment
    if sparsity_aware:
        # If sparsity-aware, we need the downsampled mask for later calculation
        local_scale, local_shift, patch_lr_mask = compute_alignment_params(
            pred_points=pred_patch_points,
            gt_points=gt_patch_points,
            mask=patch_mask,
            align_method=align_method,
            video_align=False,
            align_resolution=align_resolution,
            use_downsample=True,
            trunc=trunc,
            shift_mode=shift_mode,
            weight_divisor=gt_patch_radius_3d,
            return_downsampled=True
        )
    else:
        # Standard case - just get scale and shift
        local_scale, local_shift = compute_alignment_params(
            pred_points=pred_patch_points,
            gt_points=gt_patch_points,
            mask=patch_mask,
            align_method=align_method,
            video_align=False,
            align_resolution=align_resolution,
            use_downsample=True,
            trunc=trunc,
            shift_mode=shift_mode,
            weight_divisor=gt_patch_radius_3d
        )
    
    # FIX: Properly index global_scale with patch_batch_idx for comparison
    if global_scale is not None:
        scale_differ = local_scale / global_scale[patch_batch_idx]
        patch_valid = (scale_differ > 0.1) & (scale_differ < 10.0) & (global_scale[patch_batch_idx] > 0)  # [num_nonempty_patches]
    else:
        patch_valid = local_scale > 0  # [num_nonempty_patches]
    
    # Zero out scale/shift for invalid patches
    local_scale = torch.where(patch_valid, local_scale, 0)  # [num_nonempty_patches]
    local_shift = torch.where(patch_valid[:, None], local_shift, 0)  # [num_nonempty_patches, 3]
    patch_mask &= patch_valid[:, None, None]  # [num_nonempty_patches, patch_h, patch_w]
    
    # Apply scale and shift to align predicted points with ground truth
    pred_patch_points = local_scale[:, None, None, None] * pred_patch_points + local_shift[:, None, None, :]  # [num_nonempty_patches, patch_h, patch_w, 3]
    
    # Calculate harmonic mean of ground truth depths
    gt_mean = harmonic_mean(gt_points[..., 2], gt_mask, dim=(-2, -1))  # [batch_size]
    
    # Compute per-point weights based on mask and inverse depth
    patch_weight = patch_mask.float() / gt_patch_points[..., 2].clamp_min(0.1 * gt_mean[patch_batch_idx, None, None])
    # [num_nonempty_patches, patch_h, patch_w]
    
    # Calculate smooth L1 loss
    loss = _smooth(
        (pred_patch_points - gt_patch_points).abs() * patch_weight[..., None],  # [num_nonempty_patches, patch_h, patch_w, 3]
        beta=beta
    ).mean(dim=(-3, -2, -1))  # [num_nonempty_patches]
    
    # Optionally adjust for sparse data
    if sparsity_aware:
        sparsity = patch_mask.float().mean(dim=(-2, -1)) / patch_lr_mask.float().mean(dim=(-2, -1))  # [num_nonempty_patches]
        loss = loss / (sparsity + 1e-7)  # [num_nonempty_patches]
    
    # Batch-wise loss aggregation
    per_batch_loss = torch.zeros(batch_size, dtype=dtype, device=device)  # [batch_size]
    per_batch_counts = torch.zeros(batch_size, dtype=dtype, device=device)  # [batch_size]
    
    # Accumulate losses and counts per batch
    per_batch_loss.scatter_add_(0, patch_batch_idx, loss)
    per_batch_counts.scatter_add_(0, patch_batch_idx, torch.ones_like(loss))
    
    # Avoid division by zero for batches without valid patches
    per_batch_counts = torch.maximum(per_batch_counts, torch.ones_like(per_batch_counts))
    
    # Normalize loss by expected number of patches per batch element 
    loss = per_batch_loss / (per_batch_counts * num_patches / batch_size)
    loss = loss.reshape(batch_shape)  # [*batch_shape]
    
    # Calculate error metrics normalized by patch radius
    err = (pred_patch_points.detach() - gt_patch_points).norm(dim=-1) / gt_patch_radius_3d[..., None, None]
    # [num_nonempty_patches, patch_h, patch_w]
    
    # Record metrics
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1), patch_mask).item(),  # Scalar
        'delta': weighted_mean((err < 1).float(), patch_mask).item()            # Scalar
    }

    return loss.mean(), misc

def normal_loss(points: torch.Tensor, gt_points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    device, dtype = points.device, points.dtype
    height, width = points.shape[-3:-1]

    leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
    upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
    leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
    downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
    rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

    gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
    gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
    gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
    gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
    gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

    mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
    mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
    mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
    mask_downxright = mask_leftdown & mask_rightup & mask_leftup
    mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

    loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

    loss = loss.mean() / (4 * max(points.shape[-3:-1]))

    return loss, {}


def edge_loss(points: torch.Tensor, gt_points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    device, dtype = points.device, points.dtype
    height, width = points.shape[-3:-1]

    dx = points[..., :-1, :, :] - points[..., 1:, :, :]
    dy = points[..., :, :-1, :] - points[..., :, 1:, :]
    
    gt_dx = gt_points[..., :-1, :, :] - gt_points[..., 1:, :, :]
    gt_dy = gt_points[..., :, :-1, :] - gt_points[..., :, 1:, :]

    mask_dx = mask[..., :-1, :] & mask[..., 1:, :]
    mask_dy = mask[..., :, :-1] & mask[..., :, 1:]

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(0.1), math.radians(90), math.radians(3)

    loss_dx = mask_dx * _smooth(angle_diff_vec3(dx, gt_dx).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
    loss_dy = mask_dy * _smooth(angle_diff_vec3(dy, gt_dy).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
    loss = (loss_dx.mean(dim=(-2, -1)) + loss_dy.mean(dim=(-2, -1))) / (2 * max(points.shape[-3:-1]))

    return loss, {}


def mask_l2_loss(pred_mask: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    loss = gt_mask_neg.float() * pred_mask.square() + gt_mask_pos.float() * (1 - pred_mask).square()
    loss = loss.mean(dim=(-2, -1))
    return loss.mean(), {}


def mask_bce_loss(pred_mask_prob: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    loss = (gt_mask_pos | gt_mask_neg) * F.binary_cross_entropy(pred_mask_prob, gt_mask_pos.float(), reduction='none')
    loss = loss.mean(dim=(-2, -1))
    return loss, {}

def align_predictions(pred_points, gt_metric_scale, gt_metric_shift):
    """Align predicted points using scale and shift."""
    # Apply alignment: pred_aligned = scale * pred + shift
    aligned_points = gt_metric_scale[..., None, None, None] * pred_points
    aligned_points = aligned_points + gt_metric_shift[..., None, None, :]
    return aligned_points

def gradient_loss_spatial(prediction, target, mask, gt_metric_scale=None, gt_metric_shift=None, 
                          scales=4, beta=0.0, max_grad=0.0, use_norm=True):
    """
    Multi-scale spatial gradient loss for z-coordinate (depth)
    
    Args:
        prediction: predicted point cloud [B, T, H, W, 3]
        target: ground truth point cloud [B, T, H, W, 3]
        mask: valid mask [B, T, H, W]
        gt_metric_scale: scale for alignment [B]
        gt_metric_shift: shift for alignment [B, 3]
        scales: number of scales to compute loss at
        beta: parameter for smooth L1 loss (0 for regular L1)
        max_grad: maximum gradient value for clipping
        use_norm: whether to use depth normalization (default: True)
    """
    # Align predictions if scale and shift provided
    if gt_metric_scale is None or gt_metric_shift is None:
        return torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
    
    # Check if scale is valid (positive)
    if (gt_metric_scale <= 0).any():
        return torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
    
    prediction = align_predictions(prediction, gt_metric_scale, gt_metric_shift)
    
    # Extract z-coordinate (depth)
    pred_z = prediction[..., 2]  # [B, T, H, W]
    target_z = target[..., 2]    # [B, T, H, W]
    
    valid_mask = mask > 0
    if not use_norm and torch.sum(valid_mask) > 0:
        global_norm = target_z[valid_mask].mean().clamp(min=1e-2)
    else:
        global_norm = torch.tensor(1.0, device=prediction.device)
        
    total_loss = 0
    for scale in range(scales):
        step = pow(2, scale)
        
        # Apply downsampling at current scale
        curr_pred_z = pred_z[:, :, ::step, ::step]
        curr_target_z = target_z[:, :, ::step, ::step]
        curr_mask = mask[:, :, ::step, ::step]
        
        # Skip if no valid pixels
        M = torch.sum(curr_mask)
        if M == 0:
            continue
        
        # Compute spatial gradients for both prediction and target
        # X-direction gradients
        pred_grad_x = curr_pred_z[..., :, 1:] - curr_pred_z[..., :, :-1]
        target_grad_x = curr_target_z[..., :, 1:] - curr_target_z[..., :, :-1]
        
        # Normalize gradients by target depth
        if use_norm:
            mean_depth_x = 0.5 * (curr_target_z[..., :, 1:] + curr_target_z[..., :, :-1]).clamp(min=1e-2)
            pred_grad_x = pred_grad_x / mean_depth_x
            target_grad_x = target_grad_x / mean_depth_x
        else:
            pred_grad_x = pred_grad_x / global_norm
            target_grad_x = target_grad_x / global_norm
        
        # Compute difference between gradients
        diff_x = (pred_grad_x - target_grad_x).abs()
        # diff_x = torch.minimum(diff_x, torch.tensor(max_grad, device=prediction.device))
        
        # Apply smooth L1 loss if beta > 0
        if beta > 0:
            diff_x = _smooth(diff_x, beta=beta)
            
        # Apply mask
        mask_x = torch.mul(curr_mask[..., :, 1:], curr_mask[..., :, :-1])
        grad_x = torch.mul(mask_x, diff_x)
        
        # Y-direction gradients
        pred_grad_y = curr_pred_z[..., 1:, :] - curr_pred_z[..., :-1, :]
        target_grad_y = curr_target_z[..., 1:, :] - curr_target_z[..., :-1, :]
        
        # Normalize gradients by target depth
        if use_norm:
            mean_depth_y = 0.5 * (curr_target_z[..., 1:, :] + curr_target_z[..., :-1, :]).clamp(min=1e-2)
            pred_grad_y = pred_grad_y / mean_depth_y
            target_grad_y = target_grad_y / mean_depth_y
        else:
            pred_grad_y = pred_grad_y / global_norm
            target_grad_y = target_grad_y / global_norm
        
        # Compute difference between gradients
        diff_y = (pred_grad_y - target_grad_y).abs()
        # diff_y = torch.minimum(diff_y, torch.tensor(max_grad, device=prediction.device))
        
        # Apply smooth L1 loss if beta > 0
        if beta > 0:
            diff_y = _smooth(diff_y, beta=beta)
            
        # Apply mask
        mask_y = torch.mul(curr_mask[..., 1:, :], curr_mask[..., :-1, :])
        grad_y = torch.mul(mask_y, diff_y)
        
        # Sum up losses
        scale_loss = (torch.sum(grad_x) + torch.sum(grad_y)) / (M + 1e-7)
        total_loss += scale_loss
    
    return total_loss


def gradient_loss_temporal(prediction, target, mask, gt_metric_scale=None, gt_metric_shift=None, 
                           scales=4, beta=0.0, max_grad=1.0, max_depth_diff=0.0, use_norm=True):
    """
    Multi-scale temporal gradient loss for depth
    
    Args:
        prediction: predicted point cloud [B, T, H, W, 3]
        target: ground truth point cloud [B, T, H, W, 3]
        mask: valid mask [B, T, H, W]
        gt_metric_scale: scale for alignment [B]
        gt_metric_shift: shift for alignment [B, 3]
        scales: number of scales to compute loss at
        beta: parameter for smooth L1 loss (0 for regular L1)
        max_grad: maximum gradient value for scaling
        max_depth_diff: maximum allowed absolute depth difference between frames
        use_norm: whether to use depth normalization (default: True)
    """
    # Align predictions if scale and shift provided
    if gt_metric_scale is None or gt_metric_shift is None:
        return torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
    
    # Check if scale is valid (positive)
    if (gt_metric_scale <= 0).any():
        return torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
    
    prediction = align_predictions(prediction, gt_metric_scale, gt_metric_shift)
    
    target_z = target[..., 2]
    valid_mask = mask > 0
    if not use_norm and torch.sum(valid_mask) > 0:
        global_norm = target_z[valid_mask].mean().clamp(min=1e-2)
    else:
        global_norm = torch.tensor(1.0, device=prediction.device)
    
    total_loss = 0
    for scale in range(scales):
        step = pow(2, scale)
        
        # Apply temporal downsampling at current scale
        curr_pred = prediction[:, ::step, :, :]
        curr_target = target[:, ::step, :, :]
        curr_mask = mask[:, ::step, :, :]
        
        # Skip if we have only one frame at this scale
        if curr_pred.shape[1] <= 1:
            continue
        
        # Extract depths
        pred_z = curr_pred[..., 2]  # [B, T, H, W]
        target_z = curr_target[..., 2]    # [B, T, H, W]
        
        # Filter out pixels with large depth changes between frames - parallel implementation
        depth_change_mask = torch.ones_like(curr_mask, dtype=torch.bool)  # [B, T, H, W]
        
        if target_z.shape[1] > 1 and max_depth_diff > 1e-3:
            # Calculate absolute depth changes between adjacent frames
            depth_diffs = torch.abs(target_z[:, 1:] - target_z[:, :-1])  # [B, T-1, H, W]
            
            # Create mask for pixels with large changes
            large_change = depth_diffs > max_depth_diff  # [B, T-1, H, W]
            
            # Set to False any pixel that is involved in a large depth change
            # First, handle prev frames (except last frame)
            depth_change_mask[:, :-1] = depth_change_mask[:, :-1] & ~large_change
        
        # Update mask to exclude pixels with large depth changes
        curr_mask = curr_mask & depth_change_mask
                
        # Total valid pixels
        M = torch.sum(curr_mask)
        if M == 0:
            continue

        epsilon = 1e-2
        
        # Compute temporal gradients directly - now normalized by the PREDICTION depth too
        # This prevents the model from cheating by making predictions close to zero
        if use_norm:
            pred_grad = (pred_z[:, 1:] - pred_z[:, :-1]) / target_z[:, :-1].clamp(min=epsilon)
            target_grad = (target_z[:, 1:] - target_z[:, :-1]) / target_z[:, :-1].clamp(min=epsilon)
        else:
            pred_grad = (pred_z[:, 1:] - pred_z[:, :-1]) / global_norm
            target_grad = (target_z[:, 1:] - target_z[:, :-1]) / global_norm
        
        # Compute difference between gradients
        diff_t = (pred_grad - target_grad).abs()
        
        # diff_t = torch.minimum(diff_t, torch.tensor(max_grad, device=prediction.device))
        
        # Apply smooth L1 loss if beta > 0
        if beta > 0:
            diff_t = _smooth(diff_t, beta=beta)
        
        # Apply mask for valid adjacent frames
        mask_t = torch.mul(curr_mask[:, 1:, ...], curr_mask[:, :-1, ...])
        diff_t = torch.mul(mask_t, diff_t)
        
        # Calculate loss
        scale_loss = torch.sum(diff_t) / (M + 1e-7)
        total_loss += scale_loss
    
    return total_loss


def transform_points_to_common_frame(
    points, camera_poses, valid_mask=None, ref_frame_idx=None, 
    scale=None, shift=None, downsample_factor=1
):
    """
    Transform points from multiple frames to a common reference frame's camera coordinates.
    Fully vectorized implementation that processes all time steps at once.
    
    Args:
        points: torch.Tensor - shape [B, T, H, W, 3] - points in camera coordinates
        camera_poses: torch.Tensor - shape [B, T, 4, 4] - camera-to-world transformation matrices
        valid_mask: torch.Tensor or None - shape [B, T, H, W] - valid pixel mask
        ref_frame_idx: torch.Tensor or None - shape [B] - reference frame indices (random if None)
        scale: torch.Tensor or None - shape [B, T] - global scale for alignment
        shift: torch.Tensor or None - shape [B, T, 3] - global shift for alignment
        downsample_factor: int - factor to downsample spatial dimensions (default=1, no downsampling)
        
    Returns:
        torch.Tensor - shape [B, T, H', W', 3] - points in reference frame camera coordinates
        torch.Tensor - shape [B, T, H', W'] - valid mask in reference frame
        torch.Tensor - shape [B] - reference frame indices
    """
    # Shapes
    B, T, H, W, _ = points.shape
    device, dtype = points.device, points.dtype
    
    # Create valid mask if not provided
    if valid_mask is None:
        valid_mask = torch.ones((B, T, H, W), device=device, dtype=torch.bool)
    
    # Choose random reference frame if not specified
    if ref_frame_idx is None:
        ref_frame_idx = torch.randint(0, T, (B,), device=device)
    
    # Downsampling for memory efficiency if requested
    if downsample_factor > 1:
        # Calculate downsampled dimensions
        target_H = max(1, H // downsample_factor)
        target_W = max(1, W // downsample_factor)
        
        # Reshape for batch processing
        points_flat = points.reshape(B*T, H, W, 3)  # [B*T, H, W, 3]
        masks_flat = valid_mask.reshape(B*T, H, W)  # [B*T, H, W]
        
        # Downsample
        points_ds, masks_ds = mask_aware_nearest_resize(
            points_flat, masks_flat, (target_W, target_H), return_index=False)
        
        # Reshape back
        H, W = target_H, target_W
        points = points_ds.reshape(B, T, H, W, 3)  # [B, T, H', W', 3]
        valid_mask = masks_ds.reshape(B, T, H, W)  # [B, T, H', W']
    
    # First apply scale and shift if provided (align predicted to ground truth scale)
    if scale is not None and shift is not None:
        # Apply alignment
        scale_expanded = scale.view(B, T, 1, 1, 1)  # [B, T, 1, 1, 1]
        shift_expanded = shift.view(B, T, 1, 1, 3)  # [B, T, 1, 1, 3]
        points = scale_expanded * points + shift_expanded  # [B, T, H, W, 3]
    
    # Reshape points for efficient processing
    points_reshaped = points.reshape(B, T, H*W, 3)  # [B, T, H*W, 3]
    
    # Get batch indices and reference frame indices
    batch_indices = torch.arange(B, device=device)
    
    # Get reference camera poses
    ref_poses = camera_poses[batch_indices, ref_frame_idx]  # [B, 4, 4]
    
    # Compute world-to-reference transform
    world_to_ref = torch.inverse(ref_poses)  # [B, 4, 4]
    
    # Extract rotation and translation components
    R_world_to_ref = world_to_ref[:, :3, :3]  # [B, 3, 3]
    t_world_to_ref = world_to_ref[:, :3, 3:4]  # [B, 3, 1]
    
    # Extract camera-to-world transforms for all frames
    R_cam_to_world = camera_poses[:, :, :3, :3]  # [B, T, 3, 3]
    t_cam_to_world = camera_poses[:, :, :3, 3:4]  # [B, T, 3, 1]
    
    # Transpose points for matrix multiplication
    points_transposed = points_reshaped.transpose(2, 3)  # [B, T, 3, H*W]
    
    # Transform points from camera to world coordinates
    # Using batch matmul with broadcasting
    world_points = torch.matmul(R_cam_to_world, points_transposed)  # [B, T, 3, H*W]
    world_points = world_points + t_cam_to_world  # [B, T, 3, H*W]
    
    # Expand reference transforms for broadcasting to all frames
    R_world_to_ref = R_world_to_ref.unsqueeze(1)  # [B, 1, 3, 3]
    t_world_to_ref = t_world_to_ref.unsqueeze(1)  # [B, 1, 3, 1]
    
    # Transform from world to reference frame
    ref_points = torch.matmul(R_world_to_ref, world_points)  # [B, T, 3, H*W]
    ref_points = ref_points + t_world_to_ref  # [B, T, 3, H*W]
    
    # Reshape to original format
    ref_points = ref_points.permute(0, 1, 3, 2).reshape(B, T, H, W, 3)  # [B, T, H, W, 3]
    
    return ref_points, valid_mask, ref_frame_idx

def cross_frame_global_loss(
    pred_points,          # [B, T, H, W, 3]
    gt_points,            # [B, T, H, W, 3]
    valid_mask,           # [B, T, H, W]
    camera_poses,         # [B, T, 4, 4]
    gt_metric_scale=None, # [B, T]
    gt_metric_shift=None, # [B, T, 3]
    ref_frame_idx=None,   # [B]
    downsample_factor=4,  # Downsampling factor (default=4)
    beta=0.0              # Smoothness parameter
):
    """
    Compute global loss across frames by transforming all points to a common reference frame.
    
    Args:
        pred_points: predicted points in camera coordinates
        gt_points: ground truth points in camera coordinates
        valid_mask: valid pixel mask
        camera_poses: camera-to-world transformation matrices
        gt_metric_scale: scale for alignment from global loss
        gt_metric_shift: shift for alignment from global loss
        ref_frame_idx: reference frame indices (random if None)
        downsample_factor: factor to downsample spatial dimensions (default=4)
        beta: smoothness parameter for loss
        
    Returns:
        loss: tensor of shape [B]
        misc: dictionary of metrics
    """
    device, dtype = pred_points.device, pred_points.dtype
    
    # Choose reference frame if not provided
    if ref_frame_idx is None:
        B, T = pred_points.shape[:2]
        ref_frame_idx = torch.randint(0, T, (B,), device=device)
    
    # Transform predicted and ground truth points to common reference frame
    pred_in_ref, pred_mask, _ = transform_points_to_common_frame(
        pred_points, camera_poses, valid_mask, ref_frame_idx, 
        gt_metric_scale, gt_metric_shift, downsample_factor)
    
    gt_in_ref, gt_mask, _ = transform_points_to_common_frame(
        gt_points, camera_poses, valid_mask, ref_frame_idx, 
        None, None, downsample_factor)

    # Combine masks
    combined_mask = pred_mask & gt_mask
    
    # Compute weighted L1 loss with absolute depth weighting
    depth_weight = 1.0 / torch.abs(gt_in_ref[..., 2]).clamp(min=1e-5)
    weight = combined_mask.float() * depth_weight
    
    # Normalize weights to avoid oversized gradients
    weight = weight.clamp(max=10.0 * weighted_mean(weight, combined_mask, dim=(-3, -2, -1), keepdim=True))
    
    # Calculate smooth L1 loss
    loss = _smooth((pred_in_ref - gt_in_ref).abs() * weight[..., None], beta=beta).mean(dim=(-3, -2, -1))
    
    # Compute error metrics for reporting
    err = (pred_in_ref.detach() - gt_in_ref).norm(dim=-1) / torch.abs(gt_in_ref[..., 2]).clamp(min=1e-5)
    misc = {
        'cross_frame_error': weighted_mean(err, combined_mask).item(),
        'cross_frame_delta_1': weighted_mean((err < 1.0).float(), combined_mask).item()
    }
    
    return loss.mean(), misc

def cross_frame_local_loss(
    pred_points,          # [B, T, H, W, 3]
    gt_points,            # [B, T, H, W, 3]
    valid_mask,           # [B, T, H, W]
    camera_poses,         # [B, T, 4, 4]
    gt_metric_scale=None, # [B, T]
    gt_metric_shift=None, # [B, T, 3]
    ref_frame_idx=None,   # [B]
    downsample_factor=4,  # Downsampling factor
    grid_size=4,          # Number of divisions along each dimension
    min_points_per_cell=32, # Minimum number of points required in a cell
    points_per_cell=64,   # Number of points to sample per cell
    num_cells_to_sample=32, # Number of cells to sample for loss computation
    align_method='roe',   # Alignment method: 'roe' or 'median_mad'
    beta=0.0,             # Smoothness parameter for loss
    trunc=1.0             # Truncation for robust alignment
):
    """
    Highly vectorized implementation of cross-frame local loss function.
    """
    device, dtype = pred_points.device, pred_points.dtype
    B, T = pred_points.shape[:2]
    
    # Transform points to reference frame
    pred_in_ref, pred_mask, ref_idx = transform_points_to_common_frame(
        pred_points, camera_poses, valid_mask, ref_frame_idx, 
        gt_metric_scale, gt_metric_shift, downsample_factor)
    
    gt_in_ref, gt_mask, _ = transform_points_to_common_frame(
        gt_points, camera_poses, valid_mask, ref_idx, 
        None, None, downsample_factor)
    
    # Combine masks
    combined_mask = pred_mask & gt_mask
    
    # Initialize loss and metrics tensors
    loss = torch.zeros(B, device=device, dtype=dtype)
    total_cells_used = torch.zeros(B, device=device, dtype=torch.long)
    all_errors = []
    all_deltas = []
    
    # Process all batches in parallel where possible
    for b in range(B):  # Still need a loop for each batch due to different bounding boxes
        # Get valid points for this batch
        valid_indices = torch.where(combined_mask[b])  # Returns tuple of (t_indices, h_indices, w_indices)
        
        if len(valid_indices[0]) < min_points_per_cell:
            continue  # Skip if not enough valid points
            
        # Extract valid points
        t_idx, h_idx, w_idx = valid_indices
        valid_gt = gt_in_ref[b, t_idx, h_idx, w_idx]  # [N_valid, 3]
        valid_pred = pred_in_ref[b, t_idx, h_idx, w_idx]  # [N_valid, 3]
        
        # Compute 10-90 percentile bounding box
        min_coords = torch.quantile(valid_gt, 0.1, dim=0)  # [3]
        max_coords = torch.quantile(valid_gt, 0.9, dim=0)  # [3]
        
        # Ensure minimum box size
        box_size = max_coords - min_coords  # [3]
        min_size = 0.1 * torch.max(box_size)
        max_coords = torch.maximum(max_coords, min_coords + min_size)
        
        # Create grid cell boundaries
        x_edges = torch.linspace(min_coords[0], max_coords[0], grid_size + 1, device=device)
        y_edges = torch.linspace(min_coords[1], max_coords[1], grid_size + 1, device=device)
        z_edges = torch.linspace(min_coords[2], max_coords[2], grid_size + 1, device=device)
        
        # Compute cell indices for each point
        x_idx = torch.bucketize(valid_gt[:, 0], x_edges) - 1
        y_idx = torch.bucketize(valid_gt[:, 1], y_edges) - 1
        z_idx = torch.bucketize(valid_gt[:, 2], z_edges) - 1
        
        # Clamp to valid range
        x_idx = torch.clamp(x_idx, 0, grid_size - 1)
        y_idx = torch.clamp(y_idx, 0, grid_size - 1)
        z_idx = torch.clamp(z_idx, 0, grid_size - 1)
        
        # Compute unique cell ID
        cell_ids = x_idx * (grid_size**2) + y_idx * grid_size + z_idx  # [N_valid]
        
        # Get unique cells and counts
        unique_cells, cell_point_indices, counts = torch.unique(
            cell_ids, return_inverse=True, return_counts=True)  # [n_unique_cells], [N_valid], [n_unique_cells]
        
        # Find cells with enough points
        valid_cell_mask = counts >= min_points_per_cell  # [n_unique_cells]
        valid_cells = unique_cells[valid_cell_mask]  # [n_valid_cells]
        valid_cell_counts = counts[valid_cell_mask]  # [n_valid_cells]
        
        if len(valid_cells) == 0:
            continue  # Skip if no valid cells
            
        # Sample cells if we have more than requested
        n_valid_cells = len(valid_cells)
        if n_valid_cells > num_cells_to_sample:
            perm = torch.randperm(n_valid_cells, device=device)[:num_cells_to_sample]
            valid_cells = valid_cells[perm]  # [num_cells_to_sample]
            valid_cell_counts = valid_cell_counts[perm]  # [num_cells_to_sample]
            n_sampled_cells = num_cells_to_sample
        else:
            n_sampled_cells = n_valid_cells
            
        # Create tensors to hold sampled points
        gt_cell_points = torch.zeros((n_sampled_cells, points_per_cell, 3), device=device, dtype=dtype)  # [n_cells, points_per_cell, 3]
        pred_cell_points = torch.zeros((n_sampled_cells, points_per_cell, 3), device=device, dtype=dtype)  # [n_cells, points_per_cell, 3]
        cell_point_mask = torch.zeros((n_sampled_cells, points_per_cell), device=device, dtype=torch.bool)  # [n_cells, points_per_cell]
        
        # Create indices for batch sampling
        # This is a smart way to sample different numbers of points per cell without loops
        cell_to_global_idx = {}  # Map from sampled cell ID to corresponding indices
        
        # Gather points per selected cell - this is the key to vectorizing
        for i, cell_id in enumerate(valid_cells):
            # Get indices of points in this cell
            cell_mask = (cell_ids == cell_id)
            cell_indices = torch.where(cell_mask)[0]  # [count_i]
            
            # Random sampling if needed
            count = len(cell_indices)
            if count > points_per_cell:
                # Random sampling without replacement
                perm = torch.randperm(count, device=device)[:points_per_cell]
                cell_indices = cell_indices[perm]
                count = points_per_cell
            
            # Store points
            gt_cell_points[i, :count] = valid_gt[cell_indices]
            pred_cell_points[i, :count] = valid_pred[cell_indices]
            cell_point_mask[i, :count] = True
        
        # Compute weights based on inverse depth
        cell_weight = cell_point_mask.float() / gt_cell_points[..., 2].abs().clamp(min=1e-5)
        max_weight = 10.0 * weighted_mean(cell_weight, cell_point_mask, dim=-1, keepdim=True)
        cell_weight = cell_weight.clamp(max=max_weight)
        
        # Compute alignment for all cells in parallel
        if align_method == 'roe':
            cell_scale, cell_shift = align_points_scale_xyz_shift(
                pred_cell_points, gt_cell_points, cell_weight, trunc=trunc)
        else:  # 'median_mad'
            cell_scale, cell_shift = align_points_median_mad(
                pred_cell_points, gt_cell_points, cell_point_mask,
                video_align=False, shift_mode='xyz')
        
        # Apply alignment
        aligned_pred = cell_scale[:, None, None] * pred_cell_points + cell_shift[:, None, :]
        
        cell_weight_expanded = cell_weight.unsqueeze(-1)  # [n_cells, points_per_cell, 1]
        # Compute error and metrics
        weighted_error = (aligned_pred - gt_cell_points).abs() * cell_weight_expanded  # [n_cells, points_per_cell, 3]
        
        # Compute smooth L1 loss - sum over points, then sum over cells (will divide by cell count later)
        cell_loss = _smooth(weighted_error, beta=beta).mean(dim=(-2, -1))  # [n_cells]
        
        with torch.no_grad():
            err = (aligned_pred - gt_cell_points).norm(dim=-1)  # [n_cells, points_per_cell]
            rel_err = err / gt_cell_points[..., 2].abs().clamp(min=1e-5)
        
        # Update batch statistics - sum over cells
        loss[b] = cell_loss.sum()
        total_cells_used[b] = n_sampled_cells
        
        # Collect metrics
        all_errors.append(weighted_mean(rel_err, cell_point_mask).item())
        all_deltas.append(weighted_mean((rel_err < 1.0).float(), cell_point_mask).item())
    
    # Normalize loss by number of cells used
    valid_batches = total_cells_used > 0
    if valid_batches.any():
        loss[valid_batches] = loss[valid_batches] / total_cells_used[valid_batches].float().clamp(min=1)
    
    # Collect metrics
    misc = {
        'cross_frame_local_error': np.mean(all_errors) if all_errors else 0.0,
        'cross_frame_local_delta_1': np.mean(all_deltas) if all_deltas else 0.0,
        'cells_used': total_cells_used.float().mean().item()
    }
    
    return loss.mean(), misc

def compute_losses(
    config,
    pred_points,
    pred_mask,
    pointmap,
    valid_mask,
    sky_mask,
    gt_focal,
    camera_poses=None,  # Add this parameter
    loss_category="synthetic",
    i_step=None,
    num_iterations=None
):
    loss_dict, weight_dict, misc_dict = {}, {}, {}
    gt_metric_scale, gt_metric_shift = None, None
    grad_metric_scale, grad_metric_shift = None, None
    
    # Add monitoring metrics
    misc_dict['monitoring'] = monitoring(pred_points)

    # Calculate each loss component
    for k, v in config['loss'][loss_category].items():
        weight_dict[k] = v['weight']  # Store the weight
        
        # First compute global alignment loss to get scale and shift
        if v['function'] == 'affine_invariant_global_loss':
            loss_dict[k], misc_dict[k], gt_metric_scale, gt_metric_shift, grad_metric_scale, grad_metric_shift = affine_invariant_global_loss(
                pred_points, pointmap, valid_mask, **v.get('params', {})
            )
        # Then compute other single-frame losses    
        elif v['function'] == 'affine_invariant_local_loss':
            loss_dict[k], misc_dict[k] = affine_invariant_local_loss(
                pred_points, pointmap, valid_mask, gt_focal, gt_metric_scale, **v.get('params', {})
            )
        elif v['function'] == 'normal_loss':
            loss_dict[k], misc_dict[k] = normal_loss(pred_points, pointmap, valid_mask)
        elif v['function'] == 'edge_loss':
            loss_dict[k], misc_dict[k] = edge_loss(pred_points, pointmap, valid_mask)
        elif v['function'] == 'mask_bce_loss':
            loss_dict[k], misc_dict[k] = mask_bce_loss(pred_mask, ~sky_mask, sky_mask)
        elif v['function'] == 'mask_l2_loss':
            loss_dict[k], misc_dict[k] = mask_l2_loss(pred_mask, ~sky_mask, sky_mask)
        # Add spatial and temporal gradient losses
        elif v['function'] == 'gradient_loss_spatial':
            loss_dict[k] = gradient_loss_spatial(
                pred_points, pointmap, valid_mask, 
                grad_metric_scale, grad_metric_shift, 
                **v.get('params', {})
            )
            misc_dict[k] = {}
        elif v['function'] == 'gradient_loss_temporal':
            loss_dict[k] = gradient_loss_temporal(
                pred_points, pointmap, valid_mask, 
                grad_metric_scale, grad_metric_shift,
                **v.get('params', {})
            )
            misc_dict[k] = {}
        # Add new cross-frame losses
        elif v['function'] in ['cross_frame_global_loss', 'cross_frame_local_loss']:
            if camera_poses is None:
                # Skip cross-frame losses if camera_poses is None
                loss_dict[k] = torch.tensor(0.0, dtype=pred_points.dtype, device=pred_points.device)
                misc_dict[k] = {}
            elif v['function'] == 'cross_frame_global_loss':
                loss_dict[k], misc_dict[k] = cross_frame_global_loss(
                    pred_points, pointmap, valid_mask, camera_poses,
                    grad_metric_scale, grad_metric_shift, **v.get('params', {})
                )
            elif v['function'] == 'cross_frame_local_loss':  # cross_frame_local_loss
                loss_dict[k], misc_dict[k] = cross_frame_local_loss(
                    pred_points, pointmap, valid_mask, camera_poses,
                    grad_metric_scale, grad_metric_shift, **v.get('params', {})
                )
        else:
            raise ValueError(f'Undefined loss function: {v["function"]}')

    # Compute weighted sum of losses
    weighted_losses = [loss_dict[k] * weight_dict[k] for k in loss_dict.keys()]
    total_loss = sum(weighted_losses)
    
    return total_loss, loss_dict, misc_dict, gt_metric_scale, gt_metric_shift


def get_flat_metrics(loss_dict, misc_dict):
    """
    Flatten nested dictionaries for easy logging.
    
    Args:
        loss_dict: Dictionary of losses
        misc_dict: Dictionary of metrics
        
    Returns:
        flat_metrics: Flattened dictionary with all metrics
    """
    # Flatten loss dict
    flat_loss_dict = {k: v.item() for k, v in loss_dict.items()}
    
    # Flatten misc dict
    flat_misc_dict = {}
    for k, v in misc_dict.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat_misc_dict[f"{k}.{sub_k}"] = sub_v
        else:
            flat_misc_dict[k] = v
    
    # Combine dictionaries
    flat_metrics = {
        **flat_loss_dict,
        **flat_misc_dict,
        'total_loss': sum(flat_loss_dict.values())
    }
    
    return flat_metrics
