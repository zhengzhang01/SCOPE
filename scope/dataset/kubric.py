import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
import random
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image


class KubricScene(Dataset):
    """Dataset loader for Kubric with grouped scenes and interval-based sampling."""
    
    def __init__(
        self,
        filelist_path: str,
        mode: str = "train",
        images_per_sample: int = 16,
        size: Tuple[int, int] = (518, 518),
        sample_interval: int = 3,
        current_epoch: int = 0,
        duplicate_times: int = 1,
        disparity: bool = True
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
        """
        self.mode = mode
        self.size = size
        self.images_per_sample = images_per_sample
        self.sample_interval = sample_interval
        self.current_epoch = current_epoch
        self.duplicate_times = max(1, duplicate_times)
        self.disparity = disparity

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
            
            # Extract scene name from path (scn00000/frames_p0_v0 format)
            path_parts = img_path.split('/')
            scene_name = f"{path_parts[-3]}/{path_parts[-2]}"  # Combine folder names
            
            # Extract frame number from image filename (rgba_00000.png format)
            frame_num = int(path_parts[-1].split('_')[1].split('.')[0])
            
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
        image_paths = []
        
        # Get first frame to determine crop position (if in train mode)
        first_image = Image.open(group[0][0]).convert('RGB')  # Convert RGBA to RGB
        first_image = np.array(first_image) / 255.0
        
        # Read first depth image using tiff format
        first_depth = Image.open(group[0][1])
        first_depth = np.array(first_depth).astype(np.float32)
        
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
            # Read RGBA image and convert to RGB
            image = Image.open(img_path).convert('RGB')
            image = np.array(image) / 255.0
            
            # Read depth image
            depth = Image.open(depth_path)
            depth = np.array(depth).astype(np.float32)
            
            # Apply non-crop transforms
            sample = {'image': image, 'depth': depth}
            for t in self.transform.transforms[:-1]:
                sample = t(sample)
                
            # Apply crop with fixed position if in train mode
            if self.mode == 'train':
                sample = self.transform.transforms[-1](sample, h_start, w_start)
            
            # Ensure depth is tensor and float32
            if not isinstance(sample['image'], torch.Tensor):
                sample['image'] = torch.from_numpy(sample['image']).float()
            if not isinstance(sample['depth'], torch.Tensor):
                sample['depth'] = torch.from_numpy(sample['depth']).float()

            # Create valid mask (depth > 0 and <= 20m)
            valid_mask = (sample['depth'] > 0) & (sample['depth'] <= 80)
            
            # Take reciprocal of depth if disparity is True
            if self.disparity:
                depth_copy = sample['depth'].clone()
                valid_depth_mask = depth_copy > 0
                depth_copy[valid_depth_mask] = 1.0 / depth_copy[valid_depth_mask]
                sample['depth'] = depth_copy
            
            images.append(sample['image'])
            depths.append(sample['depth'])
            valid_masks.append(valid_mask)
            image_paths.append(img_path)
        
        # Stack tensors
        images = torch.stack(images, dim=0)       # [N, 3, H, W]
        depths = torch.stack(depths, dim=0)       # [N, H, W]
        valid_masks = torch.stack(valid_masks, dim=0)  # [N, H, W]
        
        return {
            'image': images,
            'depth': depths,
            'valid_mask': valid_masks,
            'image_paths': image_paths,
            'scene_name': scene_name,
        }
    
    def __len__(self) -> int:
        """Return the number of grouped samples."""
        return len(self.samples)
