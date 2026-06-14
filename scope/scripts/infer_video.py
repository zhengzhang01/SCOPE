import argparse
import os
import cv2
import torch
import numpy as np
import warnings
from typing import List, Tuple, Dict, Union
import time
import re
import tempfile
from tqdm import tqdm
from glob import glob
from pathlib import Path
import multiprocessing as mp

from scope.model import import_model_class, normalize_model_name
from scope.utils.checkpoints import SCOPE_CHECKPOINT_REPO_ID, resolve_checkpoint_path

warnings.filterwarnings("ignore")

DEFAULT_CHECKPOINT = "auto"
MAX_SIZE = 1080
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

def resize_images(images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Resize images to have the longest dimension equal to MAX_SIZE (default 1080)
    while maintaining aspect ratio.
    Args:
        images (torch.Tensor): shape (B, C, H, W), batch of images
    Returns:
        resized_images (torch.Tensor): shape (B, C, H', W'), resized batch of images
        original_size (Tuple[int, int]): original (H, W) before resizing
    """
    B, C, H, W = images.shape
    original_size = (H, W)

    # Skip if already smaller than max_size
    if max(H, W) <= MAX_SIZE:
        return images, original_size

    # Calculate new dimensions
    if H > W:
        new_h, new_w = MAX_SIZE, int(W * MAX_SIZE / H)
    else:
        new_h, new_w = int(H * MAX_SIZE / W), MAX_SIZE

    # Ensure even dimensions for some operations
    new_h = new_h + (new_h % 2)
    new_w = new_w + (new_w % 2)

    # Resize images
    resized_images = torch.nn.functional.interpolate(
        images, (new_h, new_w),
        mode="bicubic", align_corners=False, antialias=True
    )

    return resized_images, original_size

def normalize_image(img: torch.Tensor) -> torch.Tensor:
    """Normalize image with ImageNet mean and std."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(-1, 1, 1)
    return (img - mean) / std

def natural_sort_key(s):
    """Sort strings containing numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def _str2bool(v) -> bool:
    """Robust bool parsing for argparse (fixes `type=bool` pitfall)."""
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")


def _discover_default_image_glob(sequence_dir: str) -> str | None:
    """Auto-select safest glob for known datasets (e.g., 7-Scenes)."""
    # 7-Scenes has both *.color.png and *.depth.png in the same folder; never mix them.
    if glob(os.path.join(sequence_dir, "*.color.png")):
        return "*.color.png"
    return None


def read_image_sequence(sequence_dir: str, image_glob: str | None = None) -> List[str]:
    """Read image sequence from directory (optionally restricted by a glob)."""
    if image_glob is None:
        image_glob = _discover_default_image_glob(sequence_dir)

    if image_glob:
        image_paths = glob(os.path.join(sequence_dir, image_glob))
    else:
        image_paths = (
            glob(os.path.join(sequence_dir, "*.jpg"))
            + glob(os.path.join(sequence_dir, "*.jpeg"))
            + glob(os.path.join(sequence_dir, "*.png"))
        )

    return sorted(image_paths, key=natural_sort_key)


def is_video_file(path: Union[str, Path]) -> bool:
    path = Path(path)
    return path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES


def safe_sequence_name(path: Union[str, Path]) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).stem).strip("._")
    return name or "video"


def extract_video_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> int:
    """Extract a video to a runtime PNG image sequence."""
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    if max_frames is not None and max_frames < 1:
        raise ValueError(f"max_frames must be >= 1 when set, got {max_frames}")

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    saved_frames = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_stride == 0:
                frame_path = output_dir / f"{saved_frames:06d}.png"
                if not cv2.imwrite(str(frame_path), frame):
                    raise IOError(f"Failed to write extracted frame: {frame_path}")
                saved_frames += 1
                if max_frames is not None and saved_frames >= max_frames:
                    break

            frame_index += 1
    finally:
        capture.release()

    if saved_frames == 0:
        raise ValueError(f"No frames were extracted from video file: {video_path}")
    return saved_frames


def _colorize_depth_frame(depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Convert a depth map to an RGB visualization frame."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    disparity = 1.0 / np.clip(depth, 1e-6, None)
    values = disparity[valid]
    vmin, vmax = np.quantile(values, [0.02, 0.98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(values.min()), float(values.max())

    if vmax <= vmin:
        normalized = np.zeros_like(disparity, dtype=np.float32)
    else:
        normalized = (disparity - vmin) / (vmax - vmin)
    normalized = np.where(valid, normalized, 0.0)
    gray = np.ascontiguousarray((normalized.clip(0, 1) * 255).astype(np.uint8))
    colored_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored_bgr[~valid] = 0
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def _make_even_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if height % 2:
        frame = frame[:-1]
    if width % 2:
        frame = frame[:, :-1]
    return np.ascontiguousarray(frame)


def save_visualization_video(
    sequence_name: str,
    frames_rgb: np.ndarray,
    depths: np.ndarray,
    masks: np.ndarray,
    output_dir: Union[str, Path],
    fps: float = 24.0,
) -> Path:
    """Save a side-by-side RGB and predicted-depth visualization video."""
    output_video_path = Path(output_dir) / f"{sequence_name}_scope_vis.mp4"
    tmp_video_path = output_video_path.with_suffix(".tmp.mp4")

    if tmp_video_path.exists():
        tmp_video_path.unlink()

    writer = None
    try:
        for idx, rgb in enumerate(frames_rgb):
            depth_vis = _colorize_depth_frame(depths[idx], masks[idx])
            if depth_vis.shape[:2] != rgb.shape[:2]:
                depth_vis = cv2.resize(depth_vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

            combined_rgb = np.concatenate([rgb, depth_vis], axis=1)
            combined_rgb = _make_even_frame(combined_rgb)
            combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)

            if writer is None:
                height, width = combined_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(tmp_video_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise IOError(f"Failed to open video writer: {tmp_video_path}")

            writer.write(combined_bgr)
    finally:
        if writer is not None:
            writer.release()

    os.replace(tmp_video_path, output_video_path)
    return output_video_path

def get_video_sequences(root_dir: str, image_subfolder: str = None, image_glob: str | None = None) -> List[Tuple[str, str]]:
    """
    Get all video sequences from root directory.
    
    Args:
        root_dir: Root directory containing sequences
        image_subfolder: Optional subfolder name that contains images (e.g., 'rgb' for Sintel).
                        If None, look for images directly in sequence folders.
        image_glob: Optional glob pattern to restrict image files (e.g., '*.color.png' for 7-Scenes).
    """
    sequences = []
    if not os.path.isdir(root_dir):
        print(f"Warning: Input directory {root_dir} does not exist.")
        return sequences

    for item_name in os.listdir(root_dir):
        full_path = os.path.join(root_dir, item_name)
        
        # Skip if not a directory
        if not os.path.isdir(full_path):
            continue
            
        if image_subfolder:
            # Look for images in the specified subfolder
            subfolder_path = os.path.join(full_path, image_subfolder)
            if os.path.isdir(subfolder_path):
                effective_glob = image_glob or _discover_default_image_glob(subfolder_path)
                if effective_glob:
                    image_files = glob(os.path.join(subfolder_path, effective_glob))
                else:
                    image_files = (
                        glob(os.path.join(subfolder_path, "*.jpg"))
                        + glob(os.path.join(subfolder_path, "*.jpeg"))
                        + glob(os.path.join(subfolder_path, "*.png"))
                    )
                if image_files:
                    sequences.append((item_name, subfolder_path))
                else:
                    print(f"Skipping {subfolder_path}: No images found in {image_subfolder} subfolder.")
            else:
                print(f"Skipping {full_path}: No {image_subfolder} subfolder found.")
        else:
            # Look for images directly in sequence folder
            effective_glob = image_glob or _discover_default_image_glob(full_path)
            if effective_glob:
                image_files = glob(os.path.join(full_path, effective_glob))
            else:
                image_files = (
                    glob(os.path.join(full_path, "*.jpg"))
                    + glob(os.path.join(full_path, "*.jpeg"))
                    + glob(os.path.join(full_path, "*.png"))
                )
            if image_files:
                sequences.append((item_name, full_path))
            else:
                print(f"Skipping {full_path}: No images found.")
                
    return sequences

def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs."""
    # Check environment variable for GPU allocation
    gpu_env = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    if gpu_env:
        # If environment variable is set, parse the GPU IDs
        try:
            return [int(x) for x in gpu_env.split(',')]
        except ValueError:
            pass
    
    # Fallback to default behavior
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))

def distribute_sequences(sequences: List[Tuple[str, str]], num_gpus: int) -> List[List[Tuple[str, str]]]:
    """Distribute sequences across GPUs."""
    total_sequences = len(sequences)
    sequences_per_gpu = total_sequences // num_gpus
    remainder = total_sequences % num_gpus
    
    distributed = []
    current_idx = 0
    
    for gpu_idx in range(num_gpus):
        chunk_size = sequences_per_gpu + (1 if gpu_idx < remainder else 0)
        end_idx = current_idx + chunk_size
        gpu_sequences = sequences[current_idx:end_idx]
        distributed.append(gpu_sequences)
        current_idx = end_idx
    
    return distributed

def process_and_save_sequence(
    sequence_name: str,
    sequence_dir: str,
    model,
    output_dir: str,
    device: str = 'cuda',
    resolution_level: int = 9,
    apply_mask: bool = True,
    force_projection: bool = True,
    shared_camera: bool = True,
    use_median_intrinsics: bool = False,
    use_fp16: bool = False,
    infer_method: str = 'infer',
    save_raw_data: bool = False,
    save_vis_video: bool = True,
    vis_fps: float = 24.0,
    image_glob: str | None = None,
    fov_x: float | None = None,
) -> bool:
    """Process a single image sequence and save predictions."""
    try:
        # Define output path
        os.makedirs(output_dir, exist_ok=True)
        output_npz_path = Path(output_dir) / f"{sequence_name}_scope_raw.npz"
        tmp_npz_path = output_npz_path.with_suffix(".tmp.npz")

        # Read image sequence
        image_paths = read_image_sequence(sequence_dir, image_glob=image_glob)
        if not image_paths:
            print(f"Skipping {sequence_name}: No images found.")
            return False

        # --- Load images and prepare batch ---
        print(f"Loading images from {sequence_dir}...")
        frames_list = []
        first_image_shape = None
        for img_path in tqdm(image_paths, desc=f"Loading {sequence_name}"):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to read image: {img_path}. Skipping frame.")
                continue

            if first_image_shape is None:
                first_image_shape = img.shape[:2] # H, W
            elif img.shape[:2] != first_image_shape:
                print(f"Warning: Inconsistent image size in {sequence_name}. Expected {first_image_shape}, got {img.shape[:2]} for {img_path}. Resizing...")
                img = cv2.resize(img, (first_image_shape[1], first_image_shape[0]), interpolation=cv2.INTER_AREA)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1) # C, H, W
            frames_list.append(img_tensor)

        if not frames_list:
            print(f"Skipping {sequence_name}: No valid images could be loaded.")
            return False

        frames_batch = torch.stack(frames_list, dim=0) # T, C, H, W
        original_size = frames_batch.shape[2:] # H, W

        # --- Resize and Normalize ---
        frames_resized, _ = resize_images(frames_batch) # T, C, H', W'
        resized_size = frames_resized.shape[2:] # H', W'
        frames_input = normalize_image(frames_resized) # T, C, H', W'
        frames_vis = np.ascontiguousarray(
            (frames_resized.permute(0, 2, 3, 1).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        )

        # --- SCOPE Inference ---
        print(f"Running SCOPE inference for {sequence_name} (batch size {frames_input.shape[0]})...")
        
        is_moge_model = isinstance(model, import_model_class('moge'))
        
        with torch.no_grad():
            if is_moge_model:
                # Process frames one by one for the MoGe image model.
                all_depths = []
                all_points = []
                all_masks = []
                all_intrinsics = []
                
                print(f"Processing {len(frames_input)} frames individually (MoGe model)...")
                for i, frame in enumerate(tqdm(frames_input)):
                    # Add batch dimension for single frame
                    frame_batch = frame.unsqueeze(0)
                    
                    # Run inference on single frame
                    pred = model.infer(
                        frame_batch.to(device),
                        resolution_level=resolution_level,
                        apply_mask=apply_mask,
                        force_projection=force_projection
                    )
                    
                    # Store results (removing batch dimension)
                    all_depths.append(pred["depth"][0])
                    all_points.append(pred["points"][0])
                    all_masks.append(pred["mask"][0])
                    all_intrinsics.append(pred["intrinsics"][0])
                    
                    # Clear GPU cache periodically
                    if (i + 1) % 10 == 0:
                        torch.cuda.empty_cache()
                
                # Stack results to match the SCOPE output format.
                predictions = {
                    "depth": torch.stack(all_depths),
                    "points": torch.stack(all_points),
                    "mask": torch.stack(all_masks),
                    "intrinsics": torch.stack(all_intrinsics)
                }
            else:
                # SCOPE processes all frames at once with the specified inference method.
                if infer_method == 'infer_simple':
                    predictions = model.infer_simple(
                        frames_input,
                        resolution_level=resolution_level,
                        use_fp16=use_fp16,
                        force_projection=force_projection,
                        device=torch.device(device),
                    )
                else:  # infer_method == 'infer'
                    predictions = model.infer(
                        frames_input,
                        fov_x=fov_x,
                        resolution_level=resolution_level,
                        apply_mask=apply_mask,
                        force_projection=force_projection,
                        use_fp16=use_fp16,
                        frame_shared_params=shared_camera,
                        use_common_intrinsics=use_median_intrinsics,
                        device=torch.device(device),
                    )

        # --- Extract and Save Results ---
        depths = predictions["depth"].cpu().numpy()      # (T, H', W') - Metric Depth
        masks = predictions["mask"].cpu().numpy()        # (T, H', W') - Boolean Mask
        intrinsics = predictions["intrinsics"].cpu().numpy()  # (T, 3, 3) - Normalized Intrinsics

        points = None
        if save_raw_data:
            # Point maps are huge; keep them opt-in for analysis only.
            points = predictions["points"].cpu().numpy()     # (T, H', W', 3) - 3D Points

        print(f"Saving predictions to {output_npz_path}...")
        save_kwargs = dict(
            depths=depths.astype(np.float32),
            masks=masks,
            intrinsics=intrinsics.astype(np.float32),
            original_size=np.array(original_size),  # H, W
            resized_size=np.array(resized_size),    # H', W'
        )
        if points is not None:
            save_kwargs["points"] = points.astype(np.float32)

        # Atomic write: avoid corrupted npz when jobs are preempted.
        np.savez_compressed(tmp_npz_path, **save_kwargs)
        os.replace(tmp_npz_path, output_npz_path)

        if save_vis_video:
            vis_video_path = save_visualization_video(
                sequence_name,
                frames_vis,
                depths,
                masks,
                output_dir,
                fps=vis_fps,
            )
            print(f"Saved visualization video to {vis_video_path}")

        print(f"Saved predictions for {sequence_name}")
        return True

    except Exception as e:
        print(f"Error processing sequence {sequence_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Best-effort cleanup of temp file.
        try:
            if 'tmp_npz_path' in locals() and Path(tmp_npz_path).exists():
                Path(tmp_npz_path).unlink()
        except Exception:
            pass
        return False

def process_sequences_on_gpu(
    gpu_id: int,
    sequences: List[Tuple[str, str]],
    checkpoint_path: str,
    model_name: str,
    output_dir: str,
    resolution_level: int = 9,
    apply_mask: bool = True,
    force_projection: bool = True,
    shared_camera: bool = True,
    use_median_intrinsics: bool = False,
    use_fp16: bool = False,
    infer_method: str = 'infer',
    save_raw_data: bool = False,
    save_vis_video: bool = True,
    vis_fps: float = 24.0,
    image_glob: str | None = None,
    fov_x: float | None = None,
):
    """Process sequences on specified GPU."""
    try:
        # Set GPU device
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'
        print(f"Process started on GPU {gpu_id}")
        
        model_name = normalize_model_name(model_name)
        print(f"GPU {gpu_id}: Loading {model_name} model from {checkpoint_path}...")
        ScopeModel = import_model_class(model_name)
        model = ScopeModel.from_pretrained(checkpoint_path)
        model = model.to(device)
        model.eval()
        
        # Process each sequence assigned to this GPU
        successful_sequences = 0
        for sequence_name, sequence_dir in sequences:
            print(f"\nGPU {gpu_id}: Processing sequence: {sequence_name}")
            success = process_and_save_sequence(
                sequence_name,
                sequence_dir,
                model,
                output_dir,
                device=device,
                resolution_level=resolution_level,
                apply_mask=apply_mask,
                force_projection=force_projection,
                shared_camera=shared_camera,
                use_median_intrinsics=use_median_intrinsics,
                use_fp16=use_fp16,
                infer_method=infer_method,
                save_raw_data=save_raw_data,
                save_vis_video=save_vis_video,
                vis_fps=vis_fps,
                image_glob=image_glob,
                fov_x=fov_x,
            )
            
            if success:
                successful_sequences += 1
                print(f"GPU {gpu_id}: Completed processing {sequence_name}")
            else:
                print(f"GPU {gpu_id}: Failed processing {sequence_name}")
                
        print(f"GPU {gpu_id}: Successfully processed {successful_sequences}/{len(sequences)} sequences")
                
    except Exception as e:
        print(f"GPU {gpu_id}: Process error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"GPU {gpu_id}: Process finished")

def main(argv: List[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.model = normalize_model_name(args.model)
    video_temp_dir = None

    try:
        checkpoint_path = str(resolve_checkpoint_path(args.checkpoint))
        print(f"Using checkpoint: {checkpoint_path}")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Get sequences to process
        input_path = Path(args.input_dir)
        if is_video_file(input_path):
            if args.sequence:
                raise ValueError("--sequence is only supported for directory inputs.")
            sequence_name = safe_sequence_name(input_path)
            video_temp_dir = tempfile.TemporaryDirectory(prefix="scope_video_")
            sequence_dir = Path(video_temp_dir.name) / sequence_name
            frame_count = extract_video_frames(
                input_path,
                sequence_dir,
                frame_stride=args.video_frame_stride,
                max_frames=args.video_max_frames,
            )
            print(f"Extracted {frame_count} frames from {input_path} to {sequence_dir}")
            sequences = [(sequence_name, str(sequence_dir))]
        elif args.sequence:
            # Handle single sequence with image subfolder consideration
            sequence_path = os.path.join(args.input_dir, args.sequence)
            if not os.path.isdir(sequence_path):
                raise ValueError(f"Specified sequence directory not found: {sequence_path}")
                
            # Check if we need to look for an image subfolder
            if args.image_subfolder:
                subfolder_path = os.path.join(sequence_path, args.image_subfolder)
                if os.path.isdir(subfolder_path):
                    sequences = [(args.sequence, subfolder_path)]
                else:
                    raise ValueError(f"Specified sequence {sequence_path} does not have a {args.image_subfolder} subfolder")
            else:
                sequences = [(args.sequence, sequence_path)]
        else:
            sequences = get_video_sequences(args.input_dir, args.image_subfolder, image_glob=args.image_glob)
            if not sequences and read_image_sequence(args.input_dir, image_glob=args.image_glob):
                sequences = [(Path(args.input_dir).name, args.input_dir)]

        if not sequences:
            print(f"No valid image sequences found in {args.input_dir}")
            return

        print(f"Found {len(sequences)} sequences to process in {args.input_dir}")
        
        # GPU Detection Logic - Modified Section to match first script
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        # Always use logical device IDs (0, 1, 2...) regardless of what physical GPUs are assigned
        num_available_gpus = torch.cuda.device_count()
        if num_available_gpus == 0:
            print("No GPUs available. Using CPU instead.")
            # Process on CPU if no GPUs are available
            device = 'cpu'
            ScopeModel = import_model_class(args.model)
            model = ScopeModel.from_pretrained(checkpoint_path)
            model = model.to(device)
            model.eval()
            
            # Process each sequence
            start_time = time.time()
            successful_sequences = 0
            for sequence_name, sequence_dir in sequences:
                print(f"\nProcessing sequence: {sequence_name}")
                success = process_and_save_sequence(
                    sequence_name,
                    sequence_dir,
                    model,
                    args.output_dir,
                    device=device,
                    resolution_level=args.resolution_level,
                    apply_mask=args.apply_mask,
                    force_projection=args.force_projection,
                    shared_camera=args.shared_camera,
                    use_median_intrinsics=args.use_median_intrinsics,
                    use_fp16=args.use_fp16,
                    infer_method=args.infer_method,
                    save_raw_data=args.save_raw_data,
                    save_vis_video=args.save_vis_video,
                    vis_fps=args.vis_fps,
                    image_glob=args.image_glob,
                    fov_x=args.fov_x,
                )
                if success:
                    successful_sequences += 1
            
            total_time = time.time() - start_time
            print(f"\nProcessing completed in {total_time:.2f} seconds")
            print(f"Successfully processed {successful_sequences}/{len(sequences)} sequences.")
            return
            
        # Use logical device IDs from 0 to device_count-1
        if args.gpus is not None:
            gpu_ids = args.gpus
            for gid in gpu_ids:
                if gid < 0 or gid >= num_available_gpus:
                    raise ValueError(f"--gpus contains invalid logical id {gid} (available: 0..{num_available_gpus-1})")
        else:
            gpu_ids = list(range(num_available_gpus))
        print(f"Detected {num_available_gpus} available GPUs")
        print(f"Using GPUs with logical IDs: {gpu_ids}")
            
        # Distribute sequences among GPUs
        distributed_sequences = distribute_sequences(sequences, len(gpu_ids))
        
        # Process sequences on multiple GPUs
        processes = []
        start_time = time.time()
        
        # Initialize multiprocessing
        mp.set_start_method('spawn', force=True)
        
        for idx, (gpu_id, gpu_sequences) in enumerate(zip(gpu_ids, distributed_sequences)):
            print(f"GPU {gpu_id} will process {len(gpu_sequences)} sequences")
            p = mp.Process(
                target=process_sequences_on_gpu,
                args=(
                    gpu_id, 
                    gpu_sequences, 
                    checkpoint_path,
                    args.model,
                    args.output_dir,
                    args.resolution_level,
                    args.apply_mask,
                    args.force_projection,
                    args.shared_camera,
                    args.use_median_intrinsics,
                    args.use_fp16,
                    args.infer_method,
                    args.save_raw_data,
                    args.save_vis_video,
                    args.vis_fps,
                    args.image_glob,
                    args.fov_x,
                )
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        total_time = time.time() - start_time
        print(f"\nMulti-GPU processing completed in {total_time:.2f} seconds")
        print(f"Raw predictions saved to: {args.output_dir}")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        for p in processes:
            if p.is_alive():
                p.terminate()
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise
    finally:
        if video_temp_dir is not None:
            video_temp_dir.cleanup()


def build_parser():
    parser = argparse.ArgumentParser(description='SCOPE video/sequence geometry inference')
    parser.add_argument('--input-dir', type=str, required=True,
                      help='Root directory containing image sequences, a single image-sequence directory, or a video file')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                      help=f'Checkpoint path, Hugging Face repo id, or "auto". Default auto checks local bundle then downloads from {SCOPE_CHECKPOINT_REPO_ID}.')
    parser.add_argument('--output-dir', type=str, default='./scope_predictions',
                      help='Directory to save SCOPE prediction .npz files')
    parser.add_argument('--gpus', type=int, nargs='+',
                      help='List of GPU IDs to use (e.g., 0 1 2). If not specified, use all available GPUs')
    parser.add_argument('--sequence', type=str, default=None,
                      help='Process only the specified sequence name (relative to input-dir)')
    parser.add_argument('--model', type=str, default='scope', choices=['scope', 'moge'],
                      help='Model to run')
    parser.add_argument('--resolution-level', type=int, default=3,
                      help='SCOPE resolution level [0-9], higher means more detailed')
    parser.add_argument('--apply-mask', type=_str2bool, nargs='?', const=True, default=False,
                      help='Whether to apply predicted mask to outputs (MoGe model)')
    parser.add_argument('--force-projection', type=_str2bool, nargs='?', const=True, default=False,
                      help='Whether to enforce projection constraints (MoGe model)')
    parser.add_argument('--shared-camera', type=_str2bool, nargs='?', const=True, default=True,
                      help='Whether all frames share the same camera intrinsics')
    parser.add_argument('--use-median-intrinsics', type=_str2bool, nargs='?', const=True, default=False,
                      help='Whether to use median intrinsics for final reprojection (SCOPE model)')
    parser.add_argument('--image-subfolder', type=str, default=None,
                      help='Subfolder containing images (e.g., "rgb" for Sintel). If not specified, look for images directly in sequence folders.')
    parser.add_argument('--image-glob', type=str, default=None,
                      help="Optional glob to restrict images (e.g. '*.color.png' for 7-Scenes). If omitted, auto-detect '*.color.png' when present.")
    parser.add_argument('--use-fp16', action='store_true',
                      help='Whether to use FP16 precision for SCOPE inference')
    parser.add_argument('--infer-method', type=str, default='infer', choices=['infer', 'infer_simple'],
                      help='Inference method to use for SCOPE (infer or infer_simple)')
    parser.add_argument('--save-raw-data', type=_str2bool, nargs='?', const=True, default=False,
                      help='Whether to save raw pointmap data (very large). Depth/mask/intrinsics are always saved.')
    parser.add_argument('--save-vis-video', type=_str2bool, nargs='?', const=True, default=True,
                      help='Whether to save a side-by-side RGB and predicted-depth visualization video')
    parser.add_argument('--vis-fps', type=float, default=24.0,
                      help='FPS for saved visualization videos')
    parser.add_argument('--fov-x', type=float, default=None,
                      help='Optional fixed horizontal FoV in degrees for SCOPE inference. Useful to stabilize focal.')
    parser.add_argument('--video-frame-stride', type=int, default=1,
                      help='Frame stride when --input-dir is a video file')
    parser.add_argument('--video-max-frames', type=int, default=None,
                      help='Maximum number of frames to extract when --input-dir is a video file')
    return parser

if __name__ == "__main__":
    main()
