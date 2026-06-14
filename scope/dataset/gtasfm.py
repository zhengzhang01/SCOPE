import os
import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop, ColorJitter, GaussianBlur, CenterCrop, generate_pointmap
import random
from typing import Dict, List, Tuple
import glob

class GTASFMPoint(Dataset):
    """Dataset loader for GTASFM dataset with HDF5 format."""
    
    def __init__(
        self,
        data_dir: str = None,
        mode: str = "train",
        images_per_sample: int = 16,
        size: Tuple[int, int] = (518, 518),
        sample_interval: int = 6,
        current_epoch: int = 0,
        duplicate_times: int = 1,
        disparity: bool = True,
        cj_p: float = 0.0,
        cj_s: float = 1.0,
        g_p: float = 0.0,
        g_s: float = 1.0,
        crop_mode: str = "none",  # Options: "random", "center", "none"
    ):
        """
        Initialize the dataset loader for GTASFM.
        
        Args:
            data_dir (str): Directory containing HDF5 files (train or test folder)
            mode (str): Dataset mode ('train' or 'val')
            images_per_sample (int): Number of images to group per sample
            size (tuple): Target size for resizing (width, height)
            sample_interval (int): Interval for sampling frames
            current_epoch (int): Current training epoch for controlled sampling
            duplicate_times (int): Number of times to duplicate the dataset
            disparity (bool): Whether to convert depth to disparity
            cj_p (float): Color jitter probability
            cj_s (float): Color jitter strength
            g_p (float): Gaussian blur probability
            g_s (float): Gaussian blur strength
            crop_mode (str): Method for cropping images ("random", "center", "none")
        """
        self.mode = mode
        self.size = size
        self.images_per_sample = images_per_sample
        self.sample_interval = sample_interval
        self.current_epoch = current_epoch
        self.duplicate_times = max(1, duplicate_times)  # Ensure at least 1
        self.disparity = disparity
        self.cj_p = cj_p
        self.cj_s = cj_s
        self.g_p = g_p
        self.g_s = g_s
        self.crop_mode = crop_mode
        if data_dir is None:
            raise ValueError("GTASFMPoint requires data_dir. Set data.dataset_roots.GTASFM in the training config or pass data_dir explicitly.")
        self.data_dir = data_dir
        
        # Find all HDF5 files in the data directory
        self.hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
        
        if len(self.hdf5_files) == 0:
            raise FileNotFoundError(f"No HDF5 files found in {data_dir}")
        
        # Initialize scenes dictionary
        self.scenes = {}
        self._initialize_scenes()
        
        # Convert scenes dict to sorted list for consistent ordering
        self.scene_names = sorted(self.scenes.keys())
        self.scenes_list = [self.scenes[scene] for scene in self.scene_names]
        
        # Group samples based on current configuration
        self.regroup_samples()
        
        # Initialize transforms
        self._initialize_transforms()
    
    def _initialize_scenes(self):
        """Initialize scenes from HDF5 files."""
        original_scenes = {}
        
        # Process each HDF5 file
        for file_idx, hdf5_path in enumerate(self.hdf5_files):
            # Use filename without extension as scene name
            base_name = os.path.splitext(os.path.basename(hdf5_path))[0]
            
            # Open the HDF5 file to get number of frames
            with h5py.File(hdf5_path, 'r') as h5file:
                # Each scene has image_0, image_1, ... We need to count how many
                keys = list(h5file.keys())
                image_keys = [k for k in keys if k.startswith('image_')]
                num_frames = len(image_keys)
            
            # Create scene data
            scene_data = []
            for frame_idx in range(num_frames):
                # Store file path and frame index for lazy loading
                scene_data.append((hdf5_path, frame_idx))
            
            original_scenes[base_name] = scene_data
        
        # Create duplicates with modified scene names if needed
        for dup_idx in range(self.duplicate_times):
            scene_suffix = f"_dup{dup_idx}" if dup_idx > 0 else ""
            
            for original_scene_name, scene_data in original_scenes.items():
                new_scene_name = original_scene_name + scene_suffix
                self.scenes[new_scene_name] = scene_data
    
    def _initialize_transforms(self):
        """Initialize image transformations."""
        net_w, net_h = self.size
        target_area = net_w * net_h  # 518 * 518
        
        # Basic transforms for all modes
        transforms_list = [
            ColorJitter(p=self.cj_p, strength=self.cj_s),
            GaussianBlur(p=self.g_p, strength=self.g_s),
        ]
        
        # Add resize transform based on crop mode
        if self.crop_mode == "none":
            # For none mode, use area-based resize to maintain aspect ratio
            transforms_list.append(
                Resize(
                    width=net_w,
                    height=net_h,
                    resize_target=True if self.mode == 'train' else False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='area',  # Special method for 'none' crop mode
                    image_interpolation_method=cv2.INTER_CUBIC,
                    target_area=target_area,
                )
            )
        else:
            # For random and center crop modes, use lower_bound resize
            transforms_list.append(
                Resize(
                    width=net_w,
                    height=net_h,
                    resize_target=True if self.mode == 'train' else False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                )
            )
        
        # Add normalization and prepare transforms
        transforms_list.extend([
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        # Add crop transform based on mode for training
        if self.mode == 'train' and self.crop_mode != 'none':
            if self.crop_mode == 'random':
                transforms_list.append(Crop(self.size[0]))
            elif self.crop_mode == 'center':
                transforms_list.append(CenterCrop(self.size[0]))
        
        self.transform = Compose(transforms_list)
    
    def set_epoch(self, epoch: int):
        """Update current epoch and regroup samples."""
        self.current_epoch = epoch
        self.regroup_samples()
    
    def regroup_samples(self):
        """Regroup samples based on current epoch and interval-based sampling."""
        self.samples = []
        
        for scene_name, images in zip(self.scene_names, self.scenes_list):
            # Create deterministic random number generator for this scene and epoch
            rng = random.Random(hash((scene_name, self.current_epoch)))
            
            min_required_images = self.images_per_sample * self.sample_interval
            if len(images) < min_required_images:
                print(f"Warning: Scene {scene_name} has {len(images)} images, "
                    f"which is less than required {min_required_images} images. Skipping.")
                continue
            
            # Process images in intervals
            num_images = len(images)
            sampled_images = []
            
            # Process each interval
            for i in range(0, num_images, self.sample_interval):
                interval_end = min(i + self.sample_interval, num_images)
                interval_images = images[i:interval_end]
                
                if interval_images:
                    # Randomly select one image from this interval
                    selected_image = rng.choice(interval_images)
                    sampled_images.append(selected_image)
            
            # Group sampled images
            num_sampled = len(sampled_images)
            num_full_groups = num_sampled // self.images_per_sample
            remainder = num_sampled % self.images_per_sample
            
            # Create full groups
            for i in range(num_full_groups):
                group = sampled_images[i * self.images_per_sample : (i + 1) * self.images_per_sample]
                self.samples.append((scene_name, group))
            
            # Handle remaining images
            if remainder > 0:
                group = sampled_images[num_full_groups * self.images_per_sample:]
                # Pad with images from the beginning of sampled_images
                padding = sampled_images[:(self.images_per_sample - remainder)]
                group.extend(padding)
                self.samples.append((scene_name, group))
    
    def _load_frame_data(self, hdf5_path, frame_idx):
        """
        Load image, depth, camera intrinsics, and pose from HDF5 file.
        
        Args:
            hdf5_path (str): Path to HDF5 file
            frame_idx (int): Frame index
            
        Returns:
            tuple: (image, depth, camera_intrinsics, camera_pose)
        """
        with h5py.File(hdf5_path, 'r') as h5file:
            # Load image data
            img_name = f"image_{frame_idx}"
            img_data = h5file[img_name][:]
            image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode image from {hdf5_path}, frame {frame_idx}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            
            # Load depth data
            depth_name = f"depth_{frame_idx}"
            depth = h5file[depth_name][:]
            
            # Load camera intrinsics
            K_name = f"K_{frame_idx}"
            camera_intrinsics = h5file[K_name][:].astype(np.float32)
            
            # Load camera pose
            pose_name = f"pose_{frame_idx}"
            camera_pose = h5file[pose_name][:].astype(np.float32)
            
            # Convert pose to 4x4 transformation matrix if it's not already
            if camera_pose.shape != (4, 4):
                # If pose is in a different format (e.g., 7-dim vector with translation + quaternion)
                # you would need to convert it here based on the specific format
                # For now, assuming it's already a 4x4 matrix
                pass
            
        return image, depth, camera_intrinsics, camera_pose
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample of grouped images from the dataset."""
        scene_name, group = self.samples[idx]
        
        images = []
        depths = []
        pointmaps = []
        valid_masks = []
        valid_masks_disparity = []
        image_paths = []
        intrinsics = []
        camera_poses = []
        sky_masks = [] 
        
        # Determine shared crop parameters for the entire group
        if self.mode == 'train' and self.crop_mode != 'none':
            first_item = group[0]
            first_image, _, _, _ = self._load_frame_data(first_item[0], first_item[1])
            
            # Apply transforms up to the last one (which would be crop)
            temp_sample = {'image': first_image}
            for t in self.transform.transforms[:-1]:
                temp_sample = t(temp_sample)
            
            # Store the preprocessed dimensions for crops
            h, w = temp_sample['image'].shape[-2:]
            
            # Get crop parameters
            if self.crop_mode == 'random':
                crop_transform = self.transform.transforms[-1]
                h_start, w_start = crop_transform.get_crop_params(h, w)
            elif self.crop_mode == 'center':
                h_start = (h - self.size[0]) // 2
                w_start = (w - self.size[1]) // 2
        
        # For each image in the group
        for hdf5_path, frame_idx in group:
            # Load data for this frame
            image, depth, camera_intrinsics, camera_pose = self._load_frame_data(hdf5_path, frame_idx)
            
            # Create sample dictionary with intrinsics
            sample = {
                'image': image, 
                'depth': depth,
                'intrinsics': camera_intrinsics,
            }
            
            # Apply transforms except the last one (crop)
            for t in self.transform.transforms[:-1] if (self.mode == 'train' and self.crop_mode != 'none') else self.transform.transforms:
                sample = t(sample)
            
            # Apply crop with consistent parameters if in train mode
            if self.mode == 'train' and self.crop_mode != 'none':
                crop_transform = self.transform.transforms[-1]
                if self.crop_mode == 'random':
                    sample = crop_transform(sample, h_start, w_start)
                elif self.crop_mode == 'center':
                    sample = crop_transform(sample, h_start, w_start)
            
            # Convert to tensors
            transformed_image = torch.from_numpy(sample['image']).float()
            transformed_depth = torch.from_numpy(sample['depth']).float()
            K_tensor = torch.from_numpy(sample['intrinsics']).float()
            
            # Create masks (assuming the same thresholds as TartanAir)
            sky_mask = (transformed_depth > 9999)    
            valid_mask = (transformed_depth > 0) & (transformed_depth <= 80)
            valid_mask_disparity = (transformed_depth > 0)
            
            # Calculate pointmap
            point_map = generate_pointmap(transformed_depth, K_tensor)

            # Take reciprocal of depth if disparity is True
            if self.disparity:
                transformed_depth[transformed_depth >= 1000] = 0
                positive_mask = transformed_depth > 0
                # Only take reciprocal of positive values, keep zeros as zeros
                depth_copy = transformed_depth.clone()
                depth_copy[positive_mask] = 1.0 / depth_copy[positive_mask]
                transformed_depth = depth_copy
            
            # Create a frame identifier for logging and traceability.
            frame_path = f"{os.path.basename(hdf5_path)}:frame_{frame_idx}"
            
            images.append(transformed_image)
            depths.append(transformed_depth)
            pointmaps.append(point_map)
            valid_masks.append(valid_mask)
            valid_masks_disparity.append(valid_mask_disparity)
            sky_masks.append(sky_mask)  
            image_paths.append(frame_path)
            intrinsics.append(K_tensor)
            camera_poses.append(torch.from_numpy(camera_pose).float())
        
        # Stack tensors
        images = torch.stack(images, dim=0)             # [N, 3, H, W]
        depths = torch.stack(depths, dim=0)             # [N, H, W]
        pointmaps = torch.stack(pointmaps, dim=0)       # [N, 3, H, W]
        valid_masks = torch.stack(valid_masks, dim=0)   # [N, H, W]
        valid_masks_disparity = torch.stack(valid_masks_disparity, dim=0)  # [N, H, W]
        sky_masks = torch.stack(sky_masks, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)     # [N, 3, 3]
        camera_poses = torch.stack(camera_poses, dim=0) # [N, 4, 4]
        
        return {
            'image': images,
            'depth': depths,
            'pointmap': pointmaps,
            'valid_mask': valid_masks,
            'valid_mask_disparity': valid_masks_disparity,
            'sky_mask': sky_masks,
            'image_paths': image_paths,
            'intrinsics': intrinsics,
            'camera_poses': camera_poses,
            'scene_name': scene_name,
        }
    
    def __len__(self) -> int:
        """Return the number of grouped samples."""
        return len(self.samples)
