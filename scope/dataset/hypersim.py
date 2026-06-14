import os
import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import random
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop, ColorJitter, GaussianBlur, CenterCrop, generate_pointmap
from typing import Dict, List, Tuple

def hypersim_distance_to_depth(npyDistance):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth


class Hypersim(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        
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
        
    def __getitem__(self, item):
        img_path = self.filelist[item].split(' ')[0]
        depth_path = self.filelist[item].split(' ')[1]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth_fd = h5py.File(depth_path, "r")
        distance_meters = np.array(depth_fd['dataset'])
        depth = hypersim_distance_to_depth(distance_meters)
        
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        sample['valid_mask'] = (torch.isnan(sample['depth']) == 0)
        sample['depth'][sample['valid_mask'] == 0] = 0
        
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)


class HypersimScene(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        
        self.mode = mode
        self.size = size

        self.scenes = {}

        # Read and parse the filelist
        with open(filelist_path, 'r') as f:
            lines = f.read().splitlines()

        for line in lines:
            if not line.strip():
                continue  # Skip empty lines
            img_path, depth_path = line.strip().split(' ')
            # Extract scene name from img_path
            # Assuming path contains 'scenes/{scene_name}/images/...'
            parts = img_path.split(os.sep)
            try:
                scenes_index = parts.index('scene')
                scene_name = parts[scenes_index + 1]
            except (ValueError, IndexError):
                raise ValueError(f"Cannot find scene name in path: {img_path}")
            
            if scene_name not in self.scenes:
                self.scenes[scene_name] = []
            self.scenes[scene_name].append((img_path, depth_path))

        # Convert scenes dict to list for indexing
        self.scene_names = sorted(self.scenes.keys())
        self.scenes_list = [self.scenes[scene] for scene in self.scene_names]
        #self.scenes_list = self.scenes_list[:13]
        
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
        
    def __getitem__(self, idx):
        """
        Returns:
            sample (dict): 
                'image': Tensor of shape [N, 3, H, W]
                'depth': Tensor of shape [N, H, W]
                'valid_mask': Tensor of shape [N, H, W]
                'image_paths': List of image paths
                'num_images': Number of images in the scene
        """
        scene = self.scenes_list[idx]  # List of (img_path, depth_path)
        images = []
        depths = []
        valid_masks = []
        image_paths = []

        for img_path, depth_path in scene:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

            # Load depth
            with h5py.File(depth_path, "r") as depth_fd:
                if 'dataset' not in depth_fd:
                    raise KeyError(f"'dataset' key not found in {depth_path}")
                distance_meters = np.array(depth_fd['dataset'])
            depth = hypersim_distance_to_depth(distance_meters)

            # Apply transforms
            transformed = self.transform({'image': image, 'depth': depth})
            transformed_image = transformed['image']
            transformed_depth = transformed['depth']

            # Convert to torch tensors
            transformed_image = torch.from_numpy(transformed_image).float()  # [3, H, W]
            transformed_depth = torch.from_numpy(transformed_depth).float()  # [H, W]

            # Create valid mask
            valid_mask = ~torch.isnan(transformed_depth)
            transformed_depth[~valid_mask] = 0.0

            images.append(transformed_image)
            depths.append(transformed_depth)
            valid_masks.append(valid_mask)
            image_paths.append(img_path)

        # Stack along new dimension (N, C, H, W)
        images = torch.stack(images, dim=0)       # [N, 3, H, W]
        depths = torch.stack(depths, dim=0)       # [N, H, W]
        valid_masks = torch.stack(valid_masks, dim=0)  # [N, H, W]

        sample = {
            'image': images,
            'depth': depths,
            'valid_mask': valid_masks,
            'image_paths': image_paths,
            'num_images': len(images),
            'idx': idx, 
        }
        
        return sample

    def __len__(self):
        return len(self.scenes_list)


class HypersimGroupedScene(Dataset):
    """Dataset loader for Hypersim with grouped scenes and interval-based sampling."""
    
    def __init__(
        self,
        filelist_path: str,
        mode: str = "train",
        images_per_sample: int = 16,
        size: Tuple[int, int] = (518, 518),
        sample_interval: int = 1,
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
            
            # Extract scene name from path
            path_parts = img_path.split(os.sep)
            try:
                scenes_index = path_parts.index('scenes')
                scene_name = path_parts[scenes_index + 1]
            except (ValueError, IndexError):
                raise ValueError(f"Cannot find scene name in path: {img_path}")
            
            # Extract frame number for sorting (assuming last part of path contains frame number)
            frame_str = path_parts[-1]
            try:
                frame_num = int(frame_str.split('.')[1])
            except (IndexError, ValueError):
                raise ValueError(f"Cannot extract frame number from filename: {frame_str}")
            
            if scene_name not in original_scenes:
                original_scenes[scene_name] = []
            
            original_scenes[scene_name].append((img_path, depth_path, frame_num))
        
        # Now create duplicates with modified scene names
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
        # Get first frame to determine crop position
        first_image = cv2.imread(group[0][0])
        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB) / 255.0
        
        with h5py.File(group[0][1], "r") as depth_fd:
            first_distance = np.array(depth_fd['dataset'])
        first_depth = hypersim_distance_to_depth(first_distance)
       
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
            with h5py.File(depth_path, "r") as depth_fd:
                distance_meters = np.array(depth_fd['dataset'])
            depth = hypersim_distance_to_depth(distance_meters)
            
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
            
            # Create valid mask using isnan
            valid_mask = ~torch.isnan(transformed_depth)
            transformed_depth[~valid_mask] = 0.0
            valid_mask_disparity = valid_mask & (transformed_depth > 0)
            valid_mask = valid_mask & (transformed_depth > 0) #& (transformed_depth <= 80)
            # valid_mask_disparity = valid_mask
            
            # Take reciprocal of depth
            if self.disparity:
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


class HypersimPoint(Dataset):
    """Dataset loader for Hypersim with pointmap generation and camera intrinsics handling."""
    
    def __init__(
        self,
        filelist_path: str,
        mode: str = "train",
        images_per_sample: int = 16,
        size: Tuple[int, int] = (518, 518),
        sample_interval: int = 1,
        current_epoch: int = 0,
        duplicate_times: int = 1,
        disparity: bool = True,
        cj_p: float = 0.0,
        cj_s: float = 1.0,
        g_p: float = 0.0,
        g_s: float = 1.0,
        crop_mode: str = "none",  # Options: "random", "center", "none"
        metadata_root: str = None,
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
        self.duplicate_times = max(1, duplicate_times)
        self.disparity = disparity
        self.cj_p = cj_p
        self.cj_s = cj_s
        self.g_p = g_p
        self.g_s = g_s
        self.crop_mode = crop_mode
        self.filelist_path = filelist_path
        self.metadata_root = metadata_root
        
        # Load camera parameters metadata
        self.camera_params_dict = {}
        self.scenes_info = {}
        self._load_camera_metadata()
        
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
    
    def _load_camera_metadata(self):
        """Load global camera parameters from the metadata file."""
        base_dir = self.metadata_root
        if base_dir is None:
            try:
                with open(self.filelist_path, 'r') as f:
                    first_line = next((line.strip() for line in f if line.strip()), "")
                if first_line:
                    img_path = first_line.split(' ')[0]
                    parts = img_path.split(os.sep)
                    if "scenes" in parts:
                        scene_idx = parts.index("scenes")
                        base_dir = os.sep.join(parts[:scene_idx]) or os.sep
            except FileNotFoundError:
                base_dir = None

        if base_dir is None:
            self.camera_params_dict = {}
            return
        
        # Load global camera parameters CSV
        all_metafile = os.path.join(base_dir, "scenes", "metadata_camera_parameters.csv")
        try:
            self.camera_params_dict = pd.read_csv(all_metafile, index_col="scene_name")
        except FileNotFoundError:
            print(f"Warning: Camera metadata file not found at {all_metafile}. Using default parameters.")
            self.camera_params_dict = {}
            
        # Get scene-specific metadata if needed
        scenes_dir = os.path.join(base_dir, "scenes")
        if os.path.exists(scenes_dir):
            scene_dirs = [d for d in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, d))]
            for scene in scene_dirs:
                detail_dir = os.path.join(scenes_dir, scene, "_detail")
                if os.path.exists(detail_dir):
                    scene_metadata_file = os.path.join(detail_dir, "metadata_scene.csv")
                    if os.path.exists(scene_metadata_file):
                        try:
                            # Get world scale from scene metadata
                            worldscale = pd.read_csv(
                                scene_metadata_file,
                                index_col="parameter_name"
                            ).to_numpy().flatten()[0].astype(np.float32)
                            
                            # Store scene info
                            self.scenes_info[scene] = {"worldscale": worldscale}
                            
                            # Get camera IDs
                            cameras_file = os.path.join(detail_dir, "metadata_cameras.csv")
                            if os.path.exists(cameras_file):
                                camera_ids = pd.read_csv(
                                    cameras_file,
                                    header=None,
                                    skiprows=1
                                ).to_numpy().flatten()
                                self.scenes_info[scene]["camera_ids"] = camera_ids
                        except Exception as e:
                            print(f"Error loading scene metadata for {scene}: {e}")
    
    def _extract_scene_and_camera_from_path(self, path):
        """Extract scene name and camera ID from file path."""
        path_parts = path.split(os.sep)
        
        # Find scene name (assuming format like /path/to/hypersim/ai_001_001/...)
        for i, part in enumerate(path_parts):
            if part == "scenes" and i+1 < len(path_parts):
                scene_name = path_parts[i+1]
                break
        else:
            raise ValueError(f"Cannot extract scene name from path: {path}")
        
        # Extract camera ID from path
        # Format is usually like ".../images/scene_cam_00_final_preview/..."
        for i, part in enumerate(path_parts):
            if part.startswith("scene_cam_") and "_final_" in part:
                camera_id = part.split("_")[2]  # Extract "00" from "scene_cam_00_final_preview"
                camera_id = f"cam_{camera_id}"
                break
        else:
            # Fallback to extracting from parent directory
            camera_id = None
        
        return scene_name, camera_id
    
    def _get_frame_id_from_filename(self, filename):
        """Extract frame ID from filename like frame.0000.tonemap.jpg."""
        parts = os.path.basename(filename).split('.')
        try:
            return int(parts[1])  # Extract the number part
        except (IndexError, ValueError):
            raise ValueError(f"Cannot extract frame number from filename: {filename}")
    
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
            
            # Extract scene name and camera ID from path
            try:
                scene_name, camera_id = self._extract_scene_and_camera_from_path(img_path)
                frame_num = self._get_frame_id_from_filename(img_path)
            except ValueError as e:
                print(f"Warning: {e}")
                continue
            
            # Construct the key for grouping (scene_name/camera_id)
            scene_key = f"{scene_name}/{camera_id}" if camera_id else scene_name
            
            if scene_key not in original_scenes:
                original_scenes[scene_key] = []
            
            # Construct camera pose paths
            scene_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(depth_path)))
            cam_dir = None
            if camera_id:
                cam_dir = os.path.join(scene_base_dir, "_detail", camera_id)
            
            # Store all information needed to load camera parameters later
            original_scenes[scene_key].append({
                "img_path": img_path,
                "depth_path": depth_path,
                "frame_num": frame_num,
                "scene_name": scene_name,
                "camera_id": camera_id,
                "camera_dir": cam_dir
            })
        
        # Create duplicates with modified scene names
        for dup_idx in range(self.duplicate_times):
            scene_suffix = f"_dup{dup_idx}" if dup_idx > 0 else ""
            
            for original_scene_key, scene_data in original_scenes.items():
                new_scene_key = original_scene_key + scene_suffix
                
                if new_scene_key not in self.scenes:
                    self.scenes[new_scene_key] = []
                
                # Add all images from the original scene to the new scene
                self.scenes[new_scene_key].extend(scene_data)
        
        # Sort images within each scene by frame number
        for scene_key in self.scenes:
            self.scenes[scene_key].sort(key=lambda x: x["frame_num"])
    
    def _initialize_transforms(self):
        """Initialize image transformations."""
        net_w, net_h = self.size
        target_area = net_w * net_h  # For "area" resize method
        
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
    
    def _get_camera_intrinsics(self, scene_name):
        """Get camera intrinsics from metadata."""
        if scene_name in self.camera_params_dict.index:
            df_ = self.camera_params_dict.loc[scene_name]
            
            width_pixels = int(df_["settings_output_img_width"])
            height_pixels = int(df_["settings_output_img_height"])
            
            M_proj = np.array([
                [df_["M_proj_00"], df_["M_proj_01"], df_["M_proj_02"], df_["M_proj_03"]],
                [df_["M_proj_10"], df_["M_proj_11"], df_["M_proj_12"], df_["M_proj_13"]],
                [df_["M_proj_20"], df_["M_proj_21"], df_["M_proj_22"], df_["M_proj_23"]],
                [df_["M_proj_30"], df_["M_proj_31"], df_["M_proj_32"], df_["M_proj_33"]],
            ])
            
            # Convert OpenGL projection matrix to camera intrinsics
            K00 = M_proj[0, 0] * width_pixels / 2.0
            K01 = -M_proj[0, 1] * width_pixels / 2.0
            K02 = (1.0 - M_proj[0, 2]) * width_pixels / 2.0
            K11 = M_proj[1, 1] * height_pixels / 2.0
            K12 = (1.0 + M_proj[1, 2]) * height_pixels / 2.0
            
            return np.array([
                [K00, K01, K02],
                [0.0, K11, K12],
                [0.0, 0.0, 1.0]
            ]).astype(np.float32)
        else:
            # Default intrinsics if metadata not available
            print(f"Warning: Camera intrinsics not found for scene {scene_name}, using default")
            return np.array([
                [886.81, 0.0, 512.0],
                [0.0, 886.81, 384.0],
                [0.0, 0.0, 1.0]
            ]).astype(np.float32)
    
    def _load_camera_pose(self, camera_dir, frame_num, scene_name):
        """Load camera pose for a specific frame."""
        if camera_dir and os.path.exists(camera_dir):
            try:
                # Load camera position and orientation
                pos_file = os.path.join(camera_dir, "camera_keyframe_positions.hdf5")
                ori_file = os.path.join(camera_dir, "camera_keyframe_orientations.hdf5")
                
                if os.path.exists(pos_file) and os.path.exists(ori_file):
                    with h5py.File(pos_file, "r") as f:
                        positions = f["dataset"][:]
                    
                    with h5py.File(ori_file, "r") as f:
                        orientations = f["dataset"][:]
                    
                    # Get position and orientation for this frame
                    position = positions[frame_num]
                    orientation = orientations[frame_num]
                    
                    # Apply worldscale if available
                    worldscale = 1.0
                    if scene_name in self.scenes_info and "worldscale" in self.scenes_info[scene_name]:
                        worldscale = self.scenes_info[scene_name]["worldscale"]
                    
                    # Create camera-to-world transform
                    T_cam2world = np.eye(4)
                    T_cam2world[:3, :3] = orientation
                    
                    # Apply flipping for coordinate system adjustment
                    T_cam2world[:3, :3] = T_cam2world[:3, :3] @ np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                    
                    # Apply world scale to position
                    T_cam2world[:3, 3] = position * worldscale
                    
                    return T_cam2world.astype(np.float32)
            except Exception as e:
                print(f"Error loading camera pose: {e}")
        
        # Return identity matrix if pose couldn't be loaded
        print(f"Warning: Camera pose not found for frame {frame_num}, using identity")
        return np.eye(4).astype(np.float32)
    
    def _distance_to_depth(self, distance, intrinsics, width, height):
        """Convert distance to depth using camera intrinsics."""
        # Extract focal length as average of fx and fy
        focal = (intrinsics[0, 0] + intrinsics[1, 1]) / 2.0
        
        # Create image plane coordinates
        image_plane_x = np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width).reshape(
            1, width).repeat(height, 0).astype(np.float32)[:, :, None]
        
        image_plane_y = np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height).reshape(
            height, 1).repeat(width, 1).astype(np.float32)[:, :, None]
        
        image_plane_z = np.full([height, width, 1], focal, np.float32)
        
        image_plane = np.concatenate([image_plane_x, image_plane_y, image_plane_z], axis=2)
        
        # Convert distance to depth
        depth = distance / np.linalg.norm(image_plane, axis=2) * focal
        
        return depth
    
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
            
            # Handle remaining images if any
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
        
        # Get details of first item for pre-processing
        first_item = group[0]
        first_img_path = first_item["img_path"]
        first_depth_path = first_item["depth_path"]
        first_scene_name = first_item["scene_name"]
        
        # Load first image to determine crop parameters
        first_image = cv2.imread(first_img_path)
        if first_image is None:
            raise FileNotFoundError(f"Failed to read image: {first_img_path}")
        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB) / 255.0
        
        # Get camera intrinsics for this scene
        first_intrinsics = self._get_camera_intrinsics(first_scene_name)
        
        # Load depth data
        with h5py.File(first_depth_path, "r") as f:
            first_distance = np.array(f["dataset"])
        
        # Convert distance to depth
        h, w = first_distance.shape
        first_depth = self._distance_to_depth(first_distance, first_intrinsics, w, h)
        
        # Determine shared crop parameters for the entire group
        if self.mode == 'train' and self.crop_mode != 'none':
            # Apply transforms up to the last one (which would be crop)
            temp_sample = {
                'image': first_image, 
                'depth': first_depth,
                'intrinsics': first_intrinsics
            }
            
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
        
        # Process each item in the group
        for item in group:
            img_path = item["img_path"]
            depth_path = item["depth_path"]
            frame_num = item["frame_num"]
            item_scene_name = item["scene_name"]
            camera_id = item["camera_id"]
            camera_dir = item["camera_dir"]
            
            # Load and process image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            
            # Get camera intrinsics
            camera_intrinsics = self._get_camera_intrinsics(item_scene_name)
            
            # Load camera pose
            camera_pose = self._load_camera_pose(camera_dir, frame_num, item_scene_name)
            
            # Load and process depth
            try:
                with h5py.File(depth_path, "r") as f:
                    distance = np.array(f["dataset"])
                
                # Get dimensions
                h, w = distance.shape
                
                # Convert distance to depth
                depth = self._distance_to_depth(distance, camera_intrinsics, w, h)
            except Exception as e:
                print(f"Error loading depth from {depth_path}: {e}")
                # Provide empty depth with same shape as image
                h, w = image.shape[:2]
                depth = np.zeros((h, w), dtype=np.float32)
            
            # Create sample dictionary
            sample = {
                'image': image, 
                'depth': depth,
                'intrinsics': camera_intrinsics
            }
            
            # Apply transforms
            if self.mode == 'train' and self.crop_mode != 'none':
                # Apply all transforms except the crop
                for t in self.transform.transforms[:-1]:
                    sample = t(sample)
                
                # Apply crop with consistent parameters
                crop_transform = self.transform.transforms[-1]
                if self.crop_mode == 'random':
                    sample = crop_transform(sample, h_start, w_start)
                elif self.crop_mode == 'center':
                    sample = crop_transform(sample, h_start, w_start)
            else:
                # Apply all transforms
                for t in self.transform.transforms:
                    sample = t(sample)
            
            # Convert to tensors
            transformed_image = torch.from_numpy(sample['image']).float()
            transformed_depth = torch.from_numpy(sample['depth']).float()
            K_tensor = torch.from_numpy(sample['intrinsics']).float()
            
            # Create valid masks
            valid_mask = ~torch.isnan(transformed_depth) & (transformed_depth > 0) & (transformed_depth <= 80)
            valid_mask_disparity = ~torch.isnan(transformed_depth) & (transformed_depth > 0)
            sky_mask = (transformed_depth > 80)
            # Replace NaN values with zeros
            transformed_depth[torch.isnan(transformed_depth)] = 0.0
            
            # Calculate pointmap using the updated intrinsics
            point_map = generate_pointmap(transformed_depth, K_tensor)
            
            # Take reciprocal of depth if disparity is True
            if self.disparity:
                positive_mask = transformed_depth > 0
                # Only take reciprocal of positive values, keep zeros as zeros
                depth_copy = transformed_depth.clone()
                depth_copy[positive_mask] = 1.0 / depth_copy[positive_mask]
                transformed_depth = depth_copy
            
            # Append to lists
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
