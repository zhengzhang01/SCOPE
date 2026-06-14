import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import random
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop

class SAMDScene(Dataset):
    """Dataset loader for SAMD with grouped scenes and interval-based sampling."""
    
    def __init__(
        self,
        dataset_path,
        mode="train",
        images_per_sample=8,
        size=(518, 518),
        num_folders=1,
        normal=False,
        sample_interval=5,  # New parameter for sampling interval
        current_epoch=0,
        depth_threshold_quantile=0.1
    ):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path (str): Base path to the SAMD dataset
            mode (str): Dataset mode ('train' or 'val')
            images_per_sample (int): Number of images to group per sample
            size (tuple): Target size for resizing (width, height)
            num_folders (int): Number of sav_XXX folders to include
            normal (bool): Whether to use depthnormal.txt instead of depth.txt
            sample_interval (int): Interval for sampling frames (e.g., 5 means sample 1 frame from every 5 frames)
            current_epoch (int): Current training epoch for controlled sampling
        """
        self.mode = mode
        self.size = size
        self.images_per_sample = images_per_sample
        self.sample_interval = sample_interval
        self.normal = normal
        self.current_epoch = current_epoch
        self.dataset_path = dataset_path
        self.depth_threshold_quantile = depth_threshold_quantile  

        # Initialize scenes dictionary
        self.scenes = {}
        self._initialize_scenes(num_folders)
        
        # Convert scenes dict to sorted list for consistent ordering
        self.scene_names = sorted(self.scenes.keys())
        self.scenes_list = [self.scenes[scene] for scene in self.scene_names]
        
        # Group samples based on current configuration
        self.regroup_samples()
        
        # Initialize transforms
        self._initialize_transforms()

    def _initialize_scenes(self, num_folders):
        """Initialize scenes from dataset folders."""
        sav_folders = sorted([
            d for d in os.listdir(self.dataset_path)
            if d.startswith('sav_') and os.path.isdir(os.path.join(self.dataset_path, d))
        ])

        if len(sav_folders) < num_folders:
            raise ValueError(
                f"Requested {num_folders} folders, but only found {len(sav_folders)}"
            )

        selected_folders = sav_folders[:num_folders]

        for folder in selected_folders:
            folder_path = os.path.join(self.dataset_path, folder)
            txt_filename = 'depthnormal.txt' if self.normal else 'depth.txt'
            txt_path = os.path.join(folder_path, txt_filename)

            if not os.path.isfile(txt_path):
                raise FileNotFoundError(f"File not found: {txt_path}")

            with open(txt_path, 'r') as f:
                for line in f.read().splitlines():
                    if not line.strip():
                        continue
                    
                    img_path, depth_path = line.strip().split(' ')
                    scene_name = img_path.strip().split(os.sep)[-3]
                    
                    if scene_name not in self.scenes:
                        self.scenes[scene_name] = []
                    self.scenes[scene_name].append((img_path, depth_path))

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

    def set_epoch(self, epoch):
        """Update current epoch and regroup samples."""
        self.current_epoch = epoch
        self.regroup_samples()

    def regroup_samples(self):
        """Regroup samples based on current epoch and interval-based sampling."""
        self.samples = []
        
        for scene_name, images in zip(self.scene_names, self.scenes_list):
            # Create deterministic random number generator for this scene and epoch
            rng = random.Random(hash((scene_name, self.current_epoch)))
            
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

    def __getitem__(self, idx):
        """Get a sample of grouped images from the dataset."""
        scene_name, group = self.samples[idx]
        
        images = []
        depths = []
        valid_masks = []
        image_paths = []

        for img_path, depth_path in group:
            # Load and process image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

            # Load and process depth
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"Depth image not found: {depth_path}")
            
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
            depth = depth.astype(np.float32) / 255.0

            # Apply transforms
            transformed = self.transform({'image': image, 'depth': depth})
            
            # Convert to tensors
            transformed_image = torch.from_numpy(transformed['image']).float()
            transformed_depth = torch.from_numpy(transformed['depth']).float()

            # Create valid mask
            valid_mask = ~torch.isnan(transformed_depth)
            transformed_depth[~valid_mask] = 0.0

            valid_depth = transformed_depth[valid_mask]
            if len(valid_depth) > 0:
                threshold = torch.quantile(valid_depth, self.depth_threshold_quantile)
                valid_mask = valid_mask & (transformed_depth >= threshold)
            
            images.append(transformed_image)
            depths.append(transformed_depth)
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

    def __len__(self):
        """Return the number of grouped samples."""
        return len(self.samples)
