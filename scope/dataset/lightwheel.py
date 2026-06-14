import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop, ColorJitter, GaussianBlur, CenterCrop, generate_pointmap
import random
from typing import Dict, List, Tuple
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
class LightWheelScene(Dataset):
    """Dataset loader for LightWheel with grouped scenes and interval-based sampling."""
    
    def __init__(
        self,
        filelist_path: "dataset/splits/lightwheel.txt",
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
        self.duplicate_times = max(1, duplicate_times)
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
            
            # Extract scene name from path using the second-to-last and third-to-last directories
            path_parts = img_path.split('/')
            scene_name = '/'.join(path_parts[-3:-1])  # Get camera type and sequence ID
            
            # Extract frame timestamp for sorting
            frame_timestamp = float(path_parts[-1].split('.')[0] + '.' + path_parts[-1].split('.')[1])
            
            if scene_name not in original_scenes:
                original_scenes[scene_name] = []
            
            original_scenes[scene_name].append((img_path, depth_path, frame_timestamp))
        
        # Now create duplicates with modified scene names
        for dup_idx in range(self.duplicate_times):
            scene_suffix = f"_dup{dup_idx}" if dup_idx > 0 else ""
            
            for original_scene_name, scene_data in original_scenes.items():
                new_scene_name = original_scene_name + scene_suffix
                
                if new_scene_name not in self.scenes:
                    self.scenes[new_scene_name] = []
                
                # Add all images from the original scene to the new scene
                self.scenes[new_scene_name].extend(scene_data)
        
        # Sort images within each scene by timestamp
        for scene_name in self.scenes:
            self.scenes[scene_name].sort(key=lambda x: x[2])
            # Remove timestamps after sorting
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
        # Get first frame to determine crop position
        first_image = cv2.imread(group[0][0])
        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth_img = cv2.imread(group[0][1], cv2.IMREAD_UNCHANGED)
        first_depth = depth_img[:,:,0] + (depth_img[:,:,1] * 256)
        first_depth = first_depth * 0.01  # Convert to meters
        
        # Apply transforms up to crop
        sample = {'image': first_image, 'depth': first_depth}
        for t in self.transform.transforms[:-1]:
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            
            # Load and process depth
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = depth_img[:,:,0] + (depth_img[:,:,1] * 256)
            depth = depth * 0.01  # Convert to meters
            
            sample = {'image': image, 'depth': depth}
            
            # Apply non-crop transforms
            for t in self.transform.transforms[:-1]:
                sample = t(sample)
                
            # Apply crop with fixed position if in train mode
            if self.mode == 'train':
                sample = self.transform.transforms[-1](sample, h_start, w_start)
            
            # Convert to tensors
            transformed_image = torch.from_numpy(sample['image'])
            transformed_depth = torch.from_numpy(sample['depth'])
            
            # Create valid mask (depth <= 80m)
            valid_mask = (transformed_depth > 0) & (transformed_depth <= 80)
            valid_mask_disparity = (transformed_depth >= 0)
            transformed_depth[transformed_depth >= 630] = 0
            
            # Take reciprocal of depth
            if self.disparity:
                transformed_depth[transformed_depth >= 630] = 0
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


class LightWheelPoint(Dataset):
    """Dataset loader for LightWheel with pointmap generation and camera parameter handling."""
    
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
        info_pickle_paths: List[str] = None,
    ):
        """
        Initialize the dataset loader.
        
        Args:
            filelist_path (str): Path to the txt file containing image/depth pairs
            info_pickle_path (str): Path to pickle file containing metadata
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
        self.duplicate_times = max(1, duplicate_times)
        self.disparity = disparity
        self.cj_p = cj_p
        self.cj_s = cj_s
        self.g_p = g_p
        self.g_s = g_s
        self.crop_mode = crop_mode
        
        # Load metadata from pickle files
        self.metadata = {}
        import pickle  # Ensure pickle is imported here
        
        pickle_paths = info_pickle_paths or []
        
        for pickle_path in pickle_paths:
            try:
                with open(pickle_path, 'rb') as f:
                    meta_info = pickle.load(f)
                    # Convert info list to a dictionary indexed by timestamp for quick lookup
                    for info in meta_info['infos']:
                        token = info['token']
                        timestamp = info['timestamp']
                        scene_token = info['scene_token']
                        composite_key = f"{scene_token}_{timestamp}"
                        self.metadata[composite_key] = info
                print(f"Loaded metadata from {pickle_path}")
            except Exception as e:
                print(f"Warning: Failed to load metadata pickle file {pickle_path}: {e}")
        
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
            # For path like /path/to/lightwheelocc/samples/CAM_BACK/00b638f5-ef65-44f0-95f2-29caae0bc4ea/1710101555.452797.jpeg
            path_parts = img_path.split('/')
            
            # Camera type (e.g., CAM_BACK)
            cam_type = path_parts[-3]
            
            # Scene token/ID (e.g., 00b638f5-ef65-44f0-95f2-29caae0bc4ea)
            scene_token = path_parts[-2]
            
            # Extract timestamp (e.g., 1710168978.394248)
            # Metadata stores timestamps in microseconds without decimal (1710168978394248)
            raw_timestamp = path_parts[-1].split('.jpeg')[0]  # Get timestamp part before .jpeg
            
            # Convert the raw timestamp to the format in metadata (microseconds integer)
            if '.' in raw_timestamp:
                # Convert "1710168978.394248" to 1710168978394248
                parts = raw_timestamp.split('.')
                timestamp = int(parts[0] + parts[1])
            else:
                timestamp = int(raw_timestamp)
            
            # Scene name combines camera type and scene token
            scene_name = f"{cam_type}/{scene_token}"
            
            if scene_name not in original_scenes:
                original_scenes[scene_name] = []
            
            original_scenes[scene_name].append((img_path, depth_path, timestamp, cam_type, scene_token))
        
        # Create duplicates with modified scene names
        for dup_idx in range(self.duplicate_times):
            scene_suffix = f"_dup{dup_idx}" if dup_idx > 0 else ""
            
            for original_scene_name, scene_data in original_scenes.items():
                new_scene_name = original_scene_name + scene_suffix
                
                if new_scene_name not in self.scenes:
                    self.scenes[new_scene_name] = []
                
                # Add all images from the original scene to the new scene
                self.scenes[new_scene_name].extend(scene_data)
        
        # Sort images within each scene by timestamp
        for scene_name in self.scenes:
            self.scenes[scene_name].sort(key=lambda x: float(x[2]))
    
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
    
    def _trans_matrix(self, translation, quaternion):
        """
        Create transformation matrix from translation and quaternion rotation
        using PyQuaternion exactly as in the original code.
        
        Args:
            translation (np.ndarray): Translation vector [3]
            quaternion (Quaternion): PyQuaternion object
        
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        tm = np.eye(4)
        tm[:3, :3] = quaternion.rotation_matrix
        tm[:3, 3] = translation
        return tm

    def _get_camera_params(self, timestamp, cam_type, scene_token):
        """
        Get camera intrinsics and extrinsics from metadata using exact implementation 
        from original LightWheel code.
        
        Args:
            timestamp (int): Timestamp of the frame in microseconds format
            cam_type (str): Camera type (e.g., CAM_BACK)
            scene_token (str): Scene token to distinguish different scenes
            
        Returns:
            tuple: (camera_pose, camera_intrinsics)
        """
        # Default values in case metadata is not available
        default_intrinsics = np.array([
            [809.2209905677063, 0.0, 829.2196003259838],
            [0.0, 809.2209905677063, 481.77842384512485],
            [0.0, 0.0, 1.0]
        ]).astype(np.float32)
        
        default_pose = np.eye(4, dtype=np.float32)
        
        # Create composite key
        composite_key = f"{scene_token}_{timestamp}"
        
        if not self.metadata or composite_key not in self.metadata:
            print(f"Warning: Metadata not found for scene {scene_token}, timestamp {timestamp}")
            return default_pose, default_intrinsics
        
        try:
            # Get metadata for this frame
            frame_info = self.metadata[composite_key]
            
            # Extract camera info
            if 'cams' not in frame_info or cam_type not in frame_info['cams']:
                print(f"Warning: Camera info not found for {cam_type}")
                return default_pose, default_intrinsics
            
            cam_info = frame_info['cams'][cam_type]
            
            # Get intrinsics
            if 'cam_intrinsic' in cam_info:
                intrinsics = np.array(cam_info['cam_intrinsic']).astype(np.float32)
            else:
                intrinsics = default_intrinsics
            
            # Exact implementation from original code
            
            # 1. Create ego-to-global transform using PyQuaternion
            global_from_ego = self._trans_matrix(
                np.array(frame_info['ego2global_translation']),
                Quaternion(frame_info['ego2global_rotation'])  # PyQuaternion handles the order
            )
            
            # 2. Create sensor-to-ego transform
            sensor2ego = self._trans_matrix(
                np.array(cam_info['sensor2ego_translation']),
                Quaternion(cam_info['sensor2ego_rotation'])
            )
            
            # 3. Compute the final camera pose
            camera_pose = np.matmul(global_from_ego, sensor2ego)
            
            return camera_pose.astype(np.float32), intrinsics
            
        except Exception as e:
            print(f"Error getting camera parameters: {e}")
            import traceback
            traceback.print_exc()
            return default_pose, default_intrinsics

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
            if first_image is None:
                raise FileNotFoundError(f"Failed to read image: {first_item[0]}")
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
        
        # Process each image in the group
        for img_path, depth_path, timestamp, cam_type, scene_token in group:
            # Load and process image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            
            # Load and process depth (two-channel PNG format specific to LightWheel)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                raise FileNotFoundError(f"Failed to read depth: {depth_path}")
            
            # Extract depth from two-channel depth image
            # First channel + Second channel * 256
            depth = depth_img[:,:,0] + (depth_img[:,:,1] * 256)
            depth = depth * 0.01  # Convert to meters
            
            # Load camera parameters
            camera_pose, camera_intrinsics = self._get_camera_params(timestamp, cam_type, scene_token)
            
            # Create sample dictionary with image, depth, and intrinsics
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
            
            # Create valid mask (depth > 0 and <= 80m)
            valid_mask = (transformed_depth > 0) & (transformed_depth < 630)
            valid_mask_disparity = (transformed_depth >= 0)
            sky_mask = (transformed_depth >= 630) | (transformed_depth == 0)             
            # Calculate pointmap using the camera intrinsics
            point_map = generate_pointmap(transformed_depth, K_tensor)
            
            # Handle disparity conversion if enabled
            if self.disparity:
                # Save original depth
                transformed_depth[transformed_depth >= 630] = 0
                depth_copy = transformed_depth.clone()
                # Apply reciprocal conversion only to positive depth values
                positive_mask = transformed_depth > 0
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
