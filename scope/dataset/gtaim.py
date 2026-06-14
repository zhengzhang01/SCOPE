import os
import cv2
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop, ColorJitter, GaussianBlur, CenterCrop, generate_pointmap
import random
from typing import Dict, List, Tuple

class GTAIMPoint(Dataset):
    """Dataset loader for GTAIM with pointmap generation and camera intrinsics handling."""
    
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
                
            img_path, depth_path, pose_npz_path = parts
            
            # Convert pose NPZ path to pickle path (just change extension)
            pose_pickle_path = os.path.splitext(pose_npz_path)[0] + '.pickle'
            
            # Extract scene name from path
            # For paths like .../GTA_IM/FPS-30/2020-05-20-21-13-13/2020-05-20-21-13-13/00000.jpg
            # We need to extract 2020-05-20-21-13-13 as the scene name
            path_parts = img_path.split('/')
            
            # Extract frame number from filename (e.g., 00001 from 00001.jpg)
            frame_num = int(os.path.splitext(path_parts[-1])[0])
            
            # Extract scene name using the second last directory
            if len(path_parts) >= 2:
                scene_name = path_parts[-2]
            else:
                # Fallback if path structure is unexpected
                scene_name = os.path.dirname(img_path)
            
            if scene_name not in original_scenes:
                original_scenes[scene_name] = []
            
            original_scenes[scene_name].append((img_path, depth_path, frame_num, pose_npz_path, pose_pickle_path))
        
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
    
    def _read_depthmap(self, depth_path, cam_near_clip, cam_far_clip):
        """
        Read depth map for GTA-IM dataset with proper conversion.
        Based on the original read_depthmap function from gta_utils.py
        
        Args:
            depth_path (str): Path to depth PNG file
            cam_near_clip (float): Camera near clip value
            cam_far_clip (float): Camera far clip value
            
        Returns:
            np.ndarray: Depth map in meters (2D array)
        """
        try:
            # Read the depth image - get RGB channels
            depth = cv2.imread(depth_path)
            if depth is None:
                raise FileNotFoundError(f"Failed to read depth: {depth_path}")
            
            # Add zero channel to make RGBA (as per original code)
            depth = np.concatenate(
                (depth, np.zeros_like(depth[:, :, 0:1], dtype=np.uint8)), 
                axis=2
            )
            
            # Reinterpret as uint32 to get single value
            depth.dtype = np.uint32
            
            # Convert to float for processing
            depth_float = depth.astype('float32')
            
            # Create a mask for non-zero values
            valid_mask = (depth_float > 0)
            
            # Initialize the result with zeros
            result = np.zeros_like(depth_float)
            
            # Apply the formula only to valid depth values (avoid divide by zero)
            if np.any(valid_mask):
                result[valid_mask] = 0.05 * 1000 / depth_float[valid_mask]
            
            # Apply the second part of the formula
            depth_processed = (
                cam_near_clip
                * cam_far_clip
                / (cam_near_clip + result * (cam_far_clip - cam_near_clip))
            )
            
            # Get 2D depth map (first channel)
            return depth_processed[:, :, 0]
            
        except Exception as e:
            print(f"Error processing depth from {depth_path}: {e}")
            # Return a zero depth map as fallback
            return np.zeros((1080, 1920), dtype=np.float32)
    
    def _load_camera_params(self, pose_npz_path, pose_pickle_path, frame_num):
        """
        Load camera pose and intrinsics from the GTA-IM dataset.
        
        Args:
            pose_npz_path (str): Path to info_frames.npz file
            pose_pickle_path (str): Path to info_frames.pickle file
            frame_num (int): Frame number to load
            
        Returns:
            tuple: (camera_pose, camera_intrinsics, cam_near_clip, cam_far_clip)
        """
        try:
            # Load camera info from NPZ file
            info_npz = np.load(pose_npz_path)
            
            # Load info from Pickle file to get near/far clip planes
            info_pickle = None
            try:
                with open(pose_pickle_path, 'rb') as f:
                    info_pickle = pickle.load(f)
                
                # Get camera parameters
                cam_near_clip = info_pickle[frame_num]['cam_near_clip']
                cam_far_clip = info_pickle[frame_num].get('cam_far_clip', 800.0)
            except Exception as e:
                print(f"Error loading pickle data: {e}, using default values")
                cam_near_clip = 0.15
                cam_far_clip = 800.0
            
            # Extract intrinsics for the specific frame
            intrinsics = info_npz['intrinsics'][frame_num].astype(np.float32)
            
            # Extract world2cam transform with correct transformation
            world2cam = info_npz['world2cam_trans'][frame_num].astype(np.float32)
            world2cam = world2cam.T  # Important: transpose as per correct implementation
            
            # Invert to get camera-to-world transform (camera pose)
            cam2world = np.linalg.inv(world2cam)
            
            return cam2world, intrinsics, cam_near_clip, cam_far_clip
            
        except Exception as e:
            print(f"Error loading camera params: {e}")
            # Return identity pose and default intrinsics if loading fails
            default_intrinsics = np.array([
                [750.0, 0.0, 960.0],
                [0.0, 750.0, 540.0],
                [0.0, 0.0, 1.0]
            ]).astype(np.float32)  # Reasonable default for GTA-IM
            
            default_pose = np.eye(4).astype(np.float32)
            return default_pose, default_intrinsics, 0.15, 800.0
    
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
        for img_path, depth_path, frame_num, pose_npz_path, pose_pickle_path in group:
            # Load camera parameters with the correct method
            camera_pose, camera_intrinsics, cam_near_clip, cam_far_clip = self._load_camera_params(
                pose_npz_path, pose_pickle_path, frame_num
            )
            
            # Load and process image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            
            # Load depth using the proper GTA-IM depth conversion with actual clip plane values
            depth = self._read_depthmap(depth_path, cam_near_clip, cam_far_clip)
            
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
            
            # Create sky mask (depth > 40m)
            sky_mask = (transformed_depth > 40)
            
            # Create valid mask (depth > 0 and <= 40m)
            valid_mask = (transformed_depth > 0) & (transformed_depth <= 40)
            valid_mask_disparity = (transformed_depth > 0)
            
            # Calculate pointmap
            point_map = generate_pointmap(transformed_depth, K_tensor)

            # Take reciprocal of depth if disparity is True
            if self.disparity:
                # Cap very large depth values
                transformed_depth[transformed_depth >= 1000] = 0
                
                # Only take reciprocal of positive values, keep zeros as zeros
                positive_mask = transformed_depth > 0
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
        
        # Return data dictionary
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