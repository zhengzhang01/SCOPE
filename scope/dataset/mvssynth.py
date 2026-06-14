import os
import cv2
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop, ColorJitter, GaussianBlur, CenterCrop, generate_pointmap
import random
from typing import Dict, List, Tuple

class MVSSynthPoint(Dataset):
    """Dataset loader for MVS Synth with pointmap generation and camera intrinsics handling."""
    
    def __init__(
        self,
        filelist_path: str,
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
        Initialize the dataset loader.
        
        Args:
            filelist_path (str): Path to the txt file containing image/depth/pose paths
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
        # Enable OpenEXR support for OpenCV
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        
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
        self.R_conv = np.array(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], 
            dtype=np.float32
        )        
        
        # Initialize scenes dictionary
        self.scenes = {}
        self._initialize_scenes(filelist_path)
        
        # Convert scenes dict to sorted list for consistent ordering
        self.scene_names = sorted(self.scenes.keys())
        self.scenes_list = [self.scenes[scene] for scene in self.scene_names]
        
        # Group samples based on current configuration
        self.regroup_samples()
        
        # Initialize transforms
        self._initialize_transforms()
    
    def _initialize_scenes(self, filelist_path: str):
        """Initialize scenes from the filelist with duplication."""
        with open(filelist_path, 'r') as f:
            filelist = f.read().splitlines()
        
        # First, collect all original scenes
        original_scenes = {}
        for line in filelist:
            if not line.strip():
                continue
            
            parts = line.strip().split(' ')
            if len(parts) != 3:
                print(f"Warning: Skipping invalid line: {line}")
                continue
                
            img_path, depth_path, pose_path = parts
            
            # Extract scene name from path
            # Example path: /path/to/MVSSynth/GTAV_720/0000/images/0002.png
            path_parts = img_path.split('/')
            
            # Scene name is the directory after GTAV_720
            try:
                gtav_idx = path_parts.index("GTAV_720")
                scene_name = path_parts[gtav_idx + 1]
            except ValueError:
                # Fallback: use the parent directory of 'images'
                img_dir_idx = path_parts.index("images")
                scene_name = path_parts[img_dir_idx - 1]
            
            # Extract frame number from filename
            frame_basename = os.path.basename(img_path)
            frame_num = int(os.path.splitext(frame_basename)[0])
            
            if scene_name not in original_scenes:
                original_scenes[scene_name] = []
            
            original_scenes[scene_name].append((img_path, depth_path, frame_num, pose_path))
        
        # Create duplicates with modified scene names
        for dup_idx in range(self.duplicate_times):
            scene_suffix = f"_dup{dup_idx}" if dup_idx > 0 else ""
            
            for original_scene_name, scene_data in original_scenes.items():
                new_scene_name = original_scene_name + scene_suffix
                
                if new_scene_name not in self.scenes:
                    self.scenes[new_scene_name] = []
                
                # Add all images from the original scene to the new scene
                self.scenes[new_scene_name].extend(scene_data)
        
        # Sort images within each scene by frame number
        for scene_name in self.scenes:
            self.scenes[scene_name].sort(key=lambda x: x[2])
    
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
    
    def _load_camera_pose(self, pose_path):
        """
        Load camera pose and intrinsics from JSON file.
        
        Args:
            pose_path (str): Path to JSON pose file
            
        Returns:
            tuple: (camera_pose, camera_intrinsics)
        """
        try:
            # Load the JSON data
            with open(pose_path, 'r') as f:
                cam_data = json.load(f)
            
            # Extract intrinsics
            c_x = cam_data["c_x"]
            c_y = cam_data["c_y"]
            f_x = cam_data["f_x"]
            f_y = cam_data["f_y"]
            
            # Build intrinsics matrix
            intrinsics = np.array(
                [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]], 
                dtype=np.float32
            )
            
            # Extract extrinsic matrix (world-to-camera)
            extrinsic = np.array(cam_data["extrinsic"], dtype=np.float32)
            
            # Invert to get camera-to-world matrix
            pose = np.linalg.inv(extrinsic)
            
            # Apply coordinate system conversion (important for consistency with OpenCV format)
            pose = self.R_conv @ pose
            
            # Ensure valid pose
            if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
                raise ValueError(f"Invalid pose from {pose_path}")
            
            return pose.astype(np.float32), intrinsics
            
        except Exception as e:
            print(f"Error loading camera pose from {pose_path}: {e}")
            # Return identity pose and default intrinsics if loading fails
            default_intrinsics = np.array([
                [750.0, 0.0, 640.0],
                [0.0, 750.0, 360.0],
                [0.0, 0.0, 1.0]
            ]).astype(np.float32)
            
            default_pose = np.eye(4).astype(np.float32)
            return default_pose, default_intrinsics
    
    def _read_depthmap(self, depth_path):
        """
        Read depth map from EXR file.
        
        Args:
            depth_path (str): Path to EXR depth file
            
        Returns:
            np.ndarray: Depth map in meters (2D array)
        """
        try:
            # Read the EXR depth file
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth is None:
                raise FileNotFoundError(f"Failed to read depth: {depth_path}")
            
            # Clean up infinite values
            depth = depth.astype(np.float32)
            depth[np.isinf(depth)] = 0.0
            
            # In MVS Synth, infinite depth means sky (typically set to inf)
            # This is usually represented in the EXR file, but we need to handle it
            
            return depth 
            
        except Exception as e:
            print(f"Error processing depth from {depth_path}: {e}")
            # Return a zero depth map as fallback
            return np.zeros((540, 960), dtype=np.float32)  # Default size for MVS Synth
    
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
            first_image = cv2.imread(first_item[0])
            first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB) / 255.0
            
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
        for img_path, depth_path, frame_num, pose_path in group:
            # Load and process image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            
            # Load depth from EXR file
            depth = self._read_depthmap(depth_path)
            
            # Load camera parameters
            camera_pose, camera_intrinsics = self._load_camera_pose(pose_path)
            
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
            
            # Create sky mask (depth > 80m)
            sky_mask = (transformed_depth == 0)
            
            # Create valid mask (depth > 0 and <= 80m)
            valid_mask = (transformed_depth > 0) #& (transformed_depth <= 8000)
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
            
            images.append(transformed_image)
            depths.append(transformed_depth)
            pointmaps.append(point_map)
            valid_masks.append(valid_mask)
            valid_masks_disparity.append(valid_mask_disparity)
            sky_masks.append(sky_mask)  
            image_paths.append(img_path)
            intrinsics.append(K_tensor)
            camera_poses.append(torch.from_numpy(camera_pose).float())
        
        # Stack tensors
        images = torch.stack(images, dim=0)             # [N, 3, H, W]
        depths = torch.stack(depths, dim=0)             # [N, H, W]
        pointmaps = torch.stack(pointmaps, dim=0)       # [N, 3, H, W]
        valid_masks = torch.stack(valid_masks, dim=0)   # [N, H, W]
        valid_masks_disparity = torch.stack(valid_masks_disparity, dim=0)  # [N, H, W]
        sky_masks = torch.stack(sky_masks, dim=0)       # [N, H, W]
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
