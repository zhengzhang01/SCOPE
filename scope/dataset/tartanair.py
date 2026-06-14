import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop, ColorJitter, GaussianBlur, CenterCrop, generate_pointmap
import random
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R

class TartanAir(Dataset):
    def __init__(self, filelist_path, mode='train', size=(518, 518), p=0.0):
        self.mode = mode
        self.size = size

        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()

        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))

    def __getitem__(self, index):
        img_path, depth_path = self.filelist[index].split(' ')

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Fail read: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        depth = np.load(depth_path)
        if depth is None:
            raise FileNotFoundError(f"Fail read: {depth_path}")

        sample = {'image': image, 'depth': depth}

        sample = self.transform(sample)

        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float()

        sample['valid_mask'] = (sample['depth'] > 0) & (sample['depth'] <= 20)

        sample['image_path'] = img_path

        return sample

    def __len__(self):
        return len(self.filelist)

class TartanAirScene(Dataset):
    """Dataset loader for TartanAir with grouped scenes and interval-based sampling."""
    
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
    ):
        """
        Initialize the dataset loader.
        
        Args:
            filelist_path (str): Path to the txt file containing image/depth pairs
            mode (str): Dataset mode ('train' or 'val')
            images_per_sample (int): Number of images to group per sample
            size (tuple): Target size for resizing (width, height)
            sample_interval (int): Interval for sampling frames
            current_epoch (int): Current training epoch for controlled sampling
            duplicate_times (int): Number of times to duplicate the dataset
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
            
            img_path, depth_path = line.strip().split(' ')
            
            # Extract scene name from path
            # For path like /path/to/TartanAir/office2/Hard/P000/image_left/000000_left.png
            path_parts = img_path.split('/')
            frame_num = int(path_parts[-1].split('_')[0])  # Extract frame number
            
            # Get scene name (e.g., 'office2/Hard/P000/image_left')
            scene_name = '/'.join(path_parts[-5:-1])
            
            if scene_name not in original_scenes:
                original_scenes[scene_name] = []
            
            original_scenes[scene_name].append((img_path, depth_path, frame_num))
        
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
            # Remove frame numbers after sorting
            self.scenes[scene_name] = [(img, depth) for img, depth, _ in self.scenes[scene_name]]
    
    def _initialize_transforms(self):
        """Initialize image transformations."""
        net_w, net_h = self.size
        self.transform = Compose([
            ColorJitter(p=self.cj_p,strength=self.cj_s),
            GaussianBlur(p=self.g_p, strength=self.g_s),
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if self.mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(self.size[0])] if self.mode == 'train' else []))
    
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
        valid_masks = []
        valid_masks_disparity = []
        image_paths = []
        
        # Generate crop positions for the first frame
        first_image = cv2.imread(group[0][0])
        first_depth = np.load(group[0][1])
        if first_image is None:
            raise FileNotFoundError(f"Failed to read image: {group[0][0]}")
        if first_depth is None:
            raise FileNotFoundError(f"Failed to read depth: {group[0][1]}")
        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB) / 255.0
        
        # Apply transforms up to crop to get the resized dimensions
        sample = {'image': first_image, 'depth': first_depth}
        for t in self.transform.transforms[:-1]:  # Only apply first two transforms
            sample = t(sample)
        
        # Get crop positions if in train mode
        h_start = None
        w_start = None
        if self.mode == 'train':
            crop_transform = self.transform.transforms[-1]
            h, w = sample['image'].shape[-2:]
            h_start, w_start = crop_transform.get_crop_params(h, w)
        
        for img_path, depth_path in group:
            # Load and process image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            
            # Load depth from .npy file
            depth = np.load(depth_path)
            if depth is None:
                raise FileNotFoundError(f"Failed to read depth: {depth_path}")
            
            # Apply transforms with fixed crop position
            sample = {'image': image, 'depth': depth}
            for t in self.transform.transforms[:-1]:  # Apply all transforms except crop
                sample = t(sample)
            
            if self.mode == 'train':  # Apply crop with fixed position
                sample = self.transform.transforms[-1](sample, h_start, w_start)
            
            # Convert to tensors
            transformed_image = torch.from_numpy(sample['image']).float()
            transformed_depth = torch.from_numpy(sample['depth']).float()
            
            # Create valid mask (depth > 0 and <= 80m for TartanAir)
            valid_mask = (transformed_depth > 0) & (transformed_depth <= 80)
            
            valid_mask_disparity = (transformed_depth > 0)
            
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
            valid_masks.append(valid_mask)
            valid_masks_disparity.append(valid_mask_disparity)
            image_paths.append(img_path)
        
        # Stack tensors
        images = torch.stack(images, dim=0)       # [N, 3, H, W]
        depths = torch.stack(depths, dim=0)       # [N, H, W]
        valid_masks = torch.stack(valid_masks, dim=0)  # [N, H, W]
        valid_masks_disparity = torch.stack(valid_masks_disparity, dim=0)  # [N, H, W]
        
        return {
            'image': images,
            'depth': depths,
            'valid_mask': valid_masks,
            'valid_mask_disparity': valid_masks_disparity,
            'image_paths': image_paths,
            'scene_name': scene_name,
        }
    
    def __len__(self) -> int:
        """Return the number of grouped samples."""
        return len(self.samples)


class TartanAirPoint(Dataset):
    """Dataset loader for TartanAir with pointmap generation and camera intrinsics handling."""
    
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
            filelist_path (str): Path to the txt file containing image/depth pairs
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
            
            img_path, depth_path = line.strip().split(' ')
            
            # Extract scene name from path
            # For path like /path/to/TartanAir/office2/Hard/P000/image_left/000000_left.png
            path_parts = img_path.split('/')
            frame_num = int(path_parts[-1].split('_')[0])  # Extract frame number
            
            # Get scene name (e.g., 'office2/Hard/P000/image_left')
            scene_name = '/'.join(path_parts[-5:-1])
            
            # Determine if this is left or right camera
            is_left = '_left.' in path_parts[-1]
            camera_type = 'left' if is_left else 'right'
            
            # Create pose file path
            base_dir = '/'.join(path_parts[:-2])  # Strip off 'image_left/000000_left.png'
            pose_path = f"{base_dir}/pose_{camera_type}.txt"
            
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
            # Keep frame_num and pose_path in the data
            self.scenes[scene_name] = [(img, depth, frame_num, pose) for img, depth, frame_num, pose in self.scenes[scene_name]]
    
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
    
    def _load_camera_pose(self, pose_path, frame_num):
        """
        Load camera pose and intrinsics from pose file.
        
        Args:
            pose_path (str): Path to pose file
            frame_num (int): Frame number to extract pose for
            
        Returns:
            tuple: (camera_pose, camera_intrinsics)
        """
        # Default intrinsics for TartanAir (as per the sample code)
        intrinsics = np.array([
            [320.0, 0.0, 320.0],
            [0.0, 320.0, 240.0],
            [0.0, 0.0, 1.0]
        ]).astype(np.float32)
        
        try:
            # Load poses
            poses = np.loadtxt(pose_path)
            
            # Extract pose for specific frame
            pose = poses[frame_num]
            
            # Extract pose components
            x, y, z, qx, qy, qz, qw = pose
            
            # Convert quaternion to rotation matrix
            rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
            
            # Create camera-to-world transform
            c2w = np.eye(4)
            c2w[:3, :3] = rotation
            c2w[:3, 3] = [x, y, z]
            
            # Adjust coordinate system (as in provided code)
            w2c = np.linalg.inv(c2w)
            w2c = w2c[[1, 2, 0, 3]]
            c2w = np.linalg.inv(w2c)
            
            return c2w.astype(np.float32), intrinsics
            
        except Exception as e:
            print(f"Error loading camera pose from {pose_path}: {e}")
            # Return identity pose and default intrinsics if loading fails
            return np.eye(4).astype(np.float32), intrinsics
    
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
            
            # Load depth from .npy file
            depth = np.load(depth_path)
            if depth is None:
                raise FileNotFoundError(f"Failed to read depth: {depth_path}")
            
            # Load camera parameters
            camera_pose, camera_intrinsics = self._load_camera_pose(pose_path, frame_num)
            
            # Create sample dictionary with intrinsics
            sample = {
                'image': image, 
                'depth': depth,
                'intrinsics': camera_intrinsics,  # Add intrinsics to the sample
            }
            
            # Apply all transforms - intrinsics will be adjusted by the transforms
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
            sky_mask = (transformed_depth > 80)    
            # Create valid mask (depth > 0 and <= 80m for TartanAir)
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
