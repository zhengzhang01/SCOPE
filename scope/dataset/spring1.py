import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop, ColorJitter, GaussianBlur, CenterCrop, generate_pointmap
import random
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import h5py


class SpringScene(Dataset):
    """Dataset loader for Spring dataset with grouped scenes and interval-based sampling."""
    
    def __init__(
        self,
        filelist_path: str,
        mode: str = "train",
        images_per_sample: int = 16,
        size: Tuple[int, int] = (518, 518),
        sample_interval: int = 3,
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
            filelist_path (str): Path to the txt file containing image/disparity pairs
            mode (str): Dataset mode ('train' or 'val')
            images_per_sample (int): Number of images to group per sample
            size (tuple): Target size for resizing (width, height)
            sample_interval (int): Interval for sampling frames
            current_epoch (int): Current training epoch for controlled sampling
            duplicate_times (int): Number of times to duplicate the dataset
            disparity (bool): Whether to use disparity directly
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
            
            img_path, disp_path = line.strip().split(' ')
            
            # Extract scene name from path (train_frame_right/0001 format)
            path_parts = img_path.split('/')
            scene_name = f"{path_parts[-4]}/{path_parts[-3]}"  # Combine folder names
            
            # Extract frame number from image filename (frame_right_0001.png format)
            frame_num = int(path_parts[-1].split('_')[-1].split('.')[0])
            
            if scene_name not in original_scenes:
                original_scenes[scene_name] = []
            
            original_scenes[scene_name].append((img_path, disp_path, frame_num))
        
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
            self.scenes[scene_name] = [(img, disp) for img, disp, _ in self.scenes[scene_name]]
    
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
        disparities = []
        valid_masks = []
        valid_masks_disparity = []
        image_paths = []
        
        # Get first frame to determine crop position (if in train mode)
        first_image = Image.open(group[0][0]).convert('RGB')
        first_image = np.array(first_image) / 255.0
        
        # Read first disparity
        with h5py.File(group[0][1], "r") as f:
            first_disp = np.array(f["disparity"]).astype(np.float32)
        
        # Apply transforms up to crop
        sample = {'image': first_image, 'depth': first_disp}
        for t in self.transform.transforms[:-1]:
            sample = t(sample)
            
        # Get crop positions if in train mode
        h_start = None
        w_start = None
        if self.mode == 'train':
            crop_transform = self.transform.transforms[-1]
            h, w = sample['image'].shape[-2:]
            h_start, w_start = crop_transform.get_crop_params(h, w)
        
        for img_path, disp_path in group:
            # Read RGB image using PIL
            image = Image.open(img_path).convert('RGB')
            image = np.array(image) / 255.0
            
            # Read disparity using h5py
            with h5py.File(disp_path, "r") as f:
                disp = np.array(f["disparity"]).astype(np.float32)
            
            # Apply non-crop transforms
            sample = {'image': image, 'depth': disp}
            for t in self.transform.transforms[:-1]:
                sample = t(sample)
                
            # Apply crop with fixed position if in train mode
            if self.mode == 'train':
                sample = self.transform.transforms[-1](sample, h_start, w_start)
            
            # Ensure depth/disparity is tensor and float32
            if not isinstance(sample['image'], torch.Tensor):
                sample['image'] = torch.from_numpy(sample['image']).float()
            if not isinstance(sample['depth'], torch.Tensor):
                sample['depth'] = torch.from_numpy(sample['depth']).float()
                
            if not self.disparity:
                positive_mask = sample['depth'] > 0
                # Only take reciprocal of positive values, keep zeros as zeros
                depth_copy = sample['depth'].clone()
                depth_copy[positive_mask] = 1.0 / depth_copy[positive_mask]
                sample['depth'] = depth_copy
         
            # Create valid mask (disparity > 0 and <= max_disp)
            valid_mask = (sample['depth'] > 0) #& (sample['depth'] <= 20)  # Adjust max_disp as needed
            valid_mask_disparity = (sample['depth'] >= 0)
            
            images.append(sample['image'])
            disparities.append(sample['depth'])
            valid_masks.append(valid_mask)
            valid_masks_disparity.append(valid_mask_disparity)
            image_paths.append(img_path)
        
        # Stack tensors
        images = torch.stack(images, dim=0)       # [N, 3, H, W]
        disparities = torch.stack(disparities, dim=0)  # [N, H, W]
        valid_masks = torch.stack(valid_masks, dim=0)  # [N, H, W]
        valid_masks_disparity = torch.stack(valid_masks_disparity, dim=0)  # [N, H, W]
        
        return {
            'image': images,
            'depth': disparities,  # Keep the key as 'depth' for compatibility
            'valid_mask': valid_masks,
            'valid_mask_disparity': valid_masks_disparity,
            'image_paths': image_paths,
            'scene_name': scene_name,
        }
    
    def __len__(self) -> int:
        """Return the number of grouped samples."""
        return len(self.samples)
    

class SpringPoint(Dataset):
    """Dataset loader for Spring dataset with pointmap generation and camera intrinsics handling."""
    
    def __init__(
        self,
        filelist_path: str,
        mode: str = "train",
        images_per_sample: int = 16,
        size: Tuple[int, int] = (518, 518),
        sample_interval: int = 3,
        current_epoch: int = 0,
        duplicate_times: int = 1,
        disparity: bool = True,
        cj_p: float = 0.0,
        cj_s: float = 1.0,
        g_p: float = 0.0,
        g_s: float = 1.0,
        crop_mode: str = "none",  # Options: "random", "center", "none"
        use_right_view: bool = True,  # Whether to use right or left camera view
        cam_data_base: str = None,
    ):
        """
        Initialize the dataset loader.
        
        Args:
            filelist_path (str): Path to the txt file containing image/disparity pairs
            mode (str): Dataset mode ('train' or 'val')
            images_per_sample (int): Number of images to group per sample
            size (tuple): Target size for resizing (width, height)
            sample_interval (int): Interval for sampling frames
            current_epoch (int): Current training epoch for controlled sampling
            duplicate_times (int): Number of times to duplicate the dataset
            disparity (bool): Whether to use disparity directly or convert to depth
            cj_p (float): Color jitter probability
            cj_s (float): Color jitter strength
            g_p (float): Gaussian blur probability
            g_s (float): Gaussian blur strength
            crop_mode (str): Method for cropping images ("random", "center", "none")
            use_right_view (bool): Whether to use right or left camera view
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
        self.use_right_view = use_right_view
        
        self.cam_data_base = cam_data_base
        
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
            if len(parts) >= 2:
                img_path, disp_path = parts[0], parts[1]
            else:
                print(f"Skipping invalid line: {line}")
                continue
            
            # Extract scene name and frame info
            path_parts = img_path.split('/')
            
            try:
                # Extract relevant parts from path
                # For Spring dataset, we need to carefully extract the correct scene ID
                
                path_parts = img_path.split('/')
                
                # Find 'spring' in path
                spring_idx = -1
                for i, part in enumerate(path_parts):
                    if part == 'spring':
                        spring_idx = i
                        break
                
                if spring_idx == -1:
                    # Fallback: try to extract scene ID based on fixed positions
                    if len(path_parts) >= 5:
                        dataset_name = "spring"
                        split = "train"
                        scene_id = path_parts[-4]  # Assuming format */0001/frame_*/frame_*_0001.png
                    else:
                        raise ValueError(f"Cannot parse path: {img_path}")
                else:
                    dataset_name = "spring"
                    split = path_parts[spring_idx + 1]  # Usually 'train'
                    scene_id = path_parts[spring_idx + 2]  # Usually '0001'
                
                # Determine if this is left or right view
                if 'left' in img_path.lower() or 'frame_left' in img_path.lower():
                    view_type = 'left'
                elif 'right' in img_path.lower() or 'frame_right' in img_path.lower():
                    view_type = 'right'
                else:
                    # Try to extract view from filename pattern
                    match = re.search(r'frame_([a-z]+)_\d+\.', img_path.lower())
                    if match:
                        view_type = match.group(1)
                    else:
                        view_type = 'unknown'
                
                # Include view type in scene name to keep left and right views separate
                scene_name = f"{dataset_name}/{split}/{scene_id}/{view_type}"
                
            except Exception as e:
                print(f"Error parsing path {img_path}: {e}")
                continue
            
            # Combine for scene name
            scene_name = f"{dataset_name}/{split}/{scene_id}/{view_type}"
            
            # Extract frame number
            frame_id = path_parts[-1].split('_')[-1].split('.')[0]  # '0001'
            frame_num = int(frame_id)
            
            if self.cam_data_base is None:
                spring_root = os.sep.join(path_parts[:spring_idx]) if spring_idx > 0 else ""
                cam_path_base = os.path.join(spring_root, "train_cam_data", dataset_name, split, scene_id, "cam_data")
            else:
                cam_path_base = os.path.join(self.cam_data_base, dataset_name, split, scene_id, "cam_data")
            
            if scene_name not in original_scenes:
                original_scenes[scene_name] = []
            
            original_scenes[scene_name].append((img_path, disp_path, frame_num, cam_path_base))
        
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
            # Keep all data in the scene
            self.scenes[scene_name] = [(img, disp, frame_num, cam_path) 
                                      for img, disp, frame_num, cam_path in self.scenes[scene_name]]
    
    def _load_camera_parameters(self, cam_path_base, frame_num):
        """
        Load camera intrinsics and extrinsics from files.
        
        Args:
            cam_path_base (str): Base path to camera data folder
            frame_num (int): Frame number
            
        Returns:
            tuple: (camera_pose, camera_intrinsics)
        """
        try:
            # In Spring dataset, frame numbers in image filenames might not directly map to line numbers
            # They could be non-sequential (e.g., 1, 37, 73...) but the cam data is sequential per frame
            # We need to determine the actual line index
            
            # Check if path exists
            if not os.path.exists(cam_path_base):
                print(f"Camera data path does not exist: {cam_path_base}")
                return np.eye(4, dtype=np.float32), np.eye(3, dtype=np.float32)
            
            # Load intrinsics
            intrinsics_path = os.path.join(cam_path_base, "intrinsics.txt")
            all_intrinsics = np.loadtxt(intrinsics_path)
            
            # Calculate frame index - in Spring dataset, frame_num might directly correspond to line index
            # First check if there's a direct mapping (0-indexed in file)
            frame_idx = (frame_num - 1) % all_intrinsics.shape[0]
            
            # Create intrinsics matrix
            K = np.eye(3, dtype=np.float32)
            K[0, 0] = all_intrinsics[frame_idx][0]  # fx
            K[1, 1] = all_intrinsics[frame_idx][1]  # fy
            K[0, 2] = all_intrinsics[frame_idx][2]  # cx
            K[1, 2] = all_intrinsics[frame_idx][3]  # cy
            
            # Load extrinsics
            extrinsics_path = os.path.join(cam_path_base, "extrinsics.txt")
            all_extrinsics = np.loadtxt(extrinsics_path)
            
            # Use same frame index for extrinsics
            frame_idx = min(frame_idx, all_extrinsics.shape[0] - 1)
            cam_ext = all_extrinsics[frame_idx].reshape(4, 4)
            
            # Convert to camera pose (inverse of extrinsics)
            pose = np.linalg.inv(cam_ext).astype(np.float32)
            
            return pose, K
            
        except Exception as e:
            print(f"Error loading camera parameters from {cam_path_base}: {e}")
            # Return identity matrices if loading fails
            return np.eye(4, dtype=np.float32), np.eye(3, dtype=np.float32)
    
    def _initialize_transforms(self):
        """Initialize image transformations based on crop mode."""
        net_w, net_h = self.size
        target_area = net_w * net_h
        
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
                
    def _convert_disparity_to_depth(self, disparity, focal_length):
        """Convert disparity to depth using the formula: depth = baseline * focal_length / disparity."""
        # Create a safe copy to avoid modifying the original
        depth = np.zeros_like(disparity)
        
        # Apply the formula only to positive disparity values
        positive_mask = disparity > 0
        depth[positive_mask] = 0.065 * focal_length / disparity[positive_mask]
        
        return depth
    
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
        # Determine shared crop parameters for the entire group
        first_item = group[0]
        first_image = Image.open(first_item[0]).convert('RGB')
        first_image = np.array(first_image) / 255.0
        
        # Read first disparity
        try:
            with h5py.File(first_item[1], "r") as f:
                first_disp = np.array(f["disparity"]).astype(np.float32)
        except Exception as e:
            print(f"Error loading first disparity from {first_item[1]}: {e}")
            # Create a dummy disparity array of zeros with expected shape
            first_disp = np.zeros((first_image.shape[0], first_image.shape[1]), dtype=np.float32)
        
        # Load first camera parameters
        first_pose, first_K = self._load_camera_parameters(first_item[3], first_item[2])
        
        focal_length = first_K[0, 0]  # Assuming fx is used for conversion
        first_depth = self._convert_disparity_to_depth(first_disp, focal_length)
        
        # Create sample with camera intrinsics
        first_sample = {
            'image': first_image, 
            'depth': first_depth,
            'intrinsics': first_K,
        }
        
        # Apply all transforms except the last one (if it's a crop)
        if self.mode == 'train' and self.crop_mode != 'none':
            for t in self.transform.transforms[:-1]:
                first_sample = t(first_sample)
            
            # Get crop parameters
            h, w = first_sample['image'].shape[-2:]
            
            if self.crop_mode == 'random':
                crop_transform = self.transform.transforms[-1]
                h_start, w_start = crop_transform.get_crop_params(h, w)
            elif self.crop_mode == 'center':
                h_start = (h - self.size[0]) // 2
                w_start = (w - self.size[1]) // 2
        
        # For each image in the group
        for img_path, disp_path, frame_num, cam_path_base in group:
            # Load and process image
            image = Image.open(img_path).convert('RGB')
            image = np.array(image) / 255.0
            
            # Load disparity from h5py file
            try:
                with h5py.File(disp_path, "r") as f:
                    disp = np.array(f["disparity"]).astype(np.float32)
            except Exception as e:
                print(f"Error loading disparity from {disp_path}: {e}")
                # Create a dummy disparity array of zeros with expected shape
                # Use the image shape for this
                disp_h, disp_w = image.shape[:2]
                disp = np.zeros((disp_h, disp_w), dtype=np.float32)
            
            # Load camera parameters
            camera_pose, camera_intrinsics = self._load_camera_parameters(cam_path_base, frame_num)
                    
            focal_length = camera_intrinsics[0, 0]  # Assuming fx is used for conversion
            depth = self._convert_disparity_to_depth(disp, focal_length)
            
            # Create sample with intrinsics
            sample = {
                'image': image, 
                'depth': depth,  # Store disparity in 'depth' key for compatibility
                'intrinsics': camera_intrinsics,
            }
            
            # Apply transforms
            if self.mode == 'train' and self.crop_mode != 'none':
                # Apply all transforms except crop
                for t in self.transform.transforms[:-1]:
                    sample = t(sample)
                
                # Apply crop with fixed position
                crop_transform = self.transform.transforms[-1]
                sample = crop_transform(sample, h_start, w_start)
            else:
                # Apply all transforms
                for t in self.transform.transforms:
                    sample = t(sample)
            
            # Convert to tensors if not already
            if not isinstance(sample['image'], torch.Tensor):
                sample['image'] = torch.from_numpy(sample['image']).float()
            if not isinstance(sample['depth'], torch.Tensor):
                sample['depth'] = torch.from_numpy(sample['depth']).float()
            if not isinstance(sample['intrinsics'], torch.Tensor):
                sample['intrinsics'] = torch.from_numpy(sample['intrinsics']).float()
                
            point_map = generate_pointmap(sample['depth'], sample['intrinsics'])
            # Convert disparity to depth if needed
            if self.disparity:
                # Convert disparity to depth (1/disparity)
                positive_mask = sample['depth'] > 0
                depth_copy = sample['depth'].clone()
                depth_copy[positive_mask] = 1.0 / depth_copy[positive_mask]
                sample['depth'] = depth_copy
                        
            valid_mask = (sample['depth'] > 0)
            valid_mask_disparity = (sample['depth'] >= 0)
            
            # Add to batch
            images.append(sample['image'])
            depths.append(sample['depth'])
            pointmaps.append(point_map)
            valid_masks.append(valid_mask)
            valid_masks_disparity.append(valid_mask_disparity)
            image_paths.append(img_path)
            intrinsics.append(sample['intrinsics'])
            camera_poses.append(torch.from_numpy(camera_pose).float())
        
        # Stack tensors
        images = torch.stack(images, dim=0)             # [N, 3, H, W]
        depths = torch.stack(depths, dim=0)             # [N, H, W]
        pointmaps = torch.stack(pointmaps, dim=0)       # [N, 3, H, W]
        valid_masks = torch.stack(valid_masks, dim=0)   # [N, H, W]
        valid_masks_disparity = torch.stack(valid_masks_disparity, dim=0)  # [N, H, W]
        intrinsics = torch.stack(intrinsics, dim=0)     # [N, 3, 3]
        camera_poses = torch.stack(camera_poses, dim=0) # [N, 4, 4]
        
        return {
            'image': images,
            'depth': depths,
            'pointmap': pointmaps,
            'valid_mask': valid_masks,
            'valid_mask_disparity': valid_masks_disparity,
            'image_paths': image_paths,
            'intrinsics': intrinsics,
            'camera_poses': camera_poses,
            'scene_name': scene_name,
        }
    
    def __len__(self) -> int:
        """Return the number of grouped samples."""
        return len(self.samples)
