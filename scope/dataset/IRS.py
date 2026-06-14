import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop, ColorJitter, GaussianBlur, CenterCrop, generate_pointmap
import random
from typing import Dict, List, Tuple
import numpy as np
import OpenEXR
import Imath


def exr2hdr(exrpath: str) -> np.ndarray:
    """Convert EXR file to HDR numpy array."""
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    
    if CNum > 1:
        Channels = ['R', 'G', 'B']
        CNum = 3
    else:
        Channels = ['G']
    
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in Channels]
    hdr = np.zeros((Size[1], Size[0], CNum), dtype=np.float32)
    
    if CNum == 1:
        hdr[:,:,0] = np.reshape(Pixels[0], (Size[1], Size[0]))
    else:
        hdr[:,:,0] = np.reshape(Pixels[0], (Size[1], Size[0]))
        hdr[:,:,1] = np.reshape(Pixels[1], (Size[1], Size[0]))
        hdr[:,:,2] = np.reshape(Pixels[2], (Size[1], Size[0]))
    
    return hdr


def load_exr(filename: str) -> np.ndarray:
    """Load EXR file and return numpy array."""
    hdr = exr2hdr(filename)
    h, w, c = hdr.shape
    if c == 1:
        hdr = np.squeeze(hdr)
    return hdr


class IRSScene(Dataset):
    """Dataset loader for IRS with grouped scenes and interval-based sampling."""
    
    def __init__(
        self,
        filelist_path: str,
        mode: str = "train",
        images_per_sample: int = 16,
        size: Tuple[int, int] = (518, 518),
        sample_interval: int = 3,
        current_epoch: int = 0,
        duplicate_times: int = 1,
        disparity: bool = False,
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
            disparity (bool): Whether to convert depth to disparity
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
            
            # Extract scene name from path (Store/Supermarket_Dark format)
            path_parts = img_path.split('/')
            scene_name = '/'.join(path_parts[-3:-1])  # Get scene name from second-to-last and third-to-last directories
            
            # Extract frame number from image filename (l_2304.png format)
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
        
        # Get first frame to determine crop position (if in train mode)
        first_image = cv2.imread(group[0][0])
        if first_image is None:
            raise FileNotFoundError(f"Failed to read image: {group[0][0]}")
        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB) / 255.0
        
        # Read first depth image
        first_depth = load_exr(group[0][1])
        if first_depth is None:
            raise FileNotFoundError(f"Failed to read depth: {group[0][1]}")
        
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
            # Read and process RGB image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            
            # Read depth image using EXR loader
            depth = load_exr(depth_path)
            if depth is None:
                raise FileNotFoundError(f"Failed to read depth: {depth_path}")
            
            # Apply non-crop transforms
            sample = {'image': image, 'depth': depth}
            for t in self.transform.transforms[:-1]:
                sample = t(sample)
            
            # Apply crop with fixed position if in train mode
            if self.mode == 'train':
                sample = self.transform.transforms[-1](sample, h_start, w_start)
                
            sample['image'] = torch.from_numpy(sample['image']).float()
            sample['depth'] = torch.from_numpy(sample['depth']).float()
            
            valid_mask_disparity = (sample['depth'] > 0)
                        
            # Take reciprocal of depth 
            if not self.disparity:
                positive_mask = sample['depth'] > 0
                # Only take reciprocal of positive values, keep zeros as zeros
                depth_copy = sample['depth'].clone()
                depth_copy[positive_mask] = 1.0 / depth_copy[positive_mask]
                sample['depth'] = depth_copy
                
            valid_mask = (sample['depth'] > 0) #& (sample['depth'] <= 20)
            
            images.append(sample['image'])
            depths.append(sample['depth'])
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
    
    
class IRSPoint(Dataset):
    """Dataset loader for IRS with grouped scenes, camera parameters, and pointmap generation."""
    
    def __init__(
        self,
        filelist_path: str,
        mode: str = "train",
        images_per_sample: int = 16,
        size: Tuple[int, int] = (518, 518),
        sample_interval: int = 3,
        current_epoch: int = 0,
        duplicate_times: int = 1,
        disparity: bool = False,
        cj_p: float = 0.0,
        cj_s: float = 1.0,
        g_p: float = 0.0,
        g_s: float = 1.0,
        crop_mode: str = "none",  # Options: "random", "center", "none"
        dataset_root: str = None,
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
        
        self.base_path = dataset_root
        
        # Extract base path from filelist if possible
        if self.base_path is None and os.path.exists(filelist_path):
            with open(filelist_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    img_path = first_line.split(' ')[0]
                    extracted_idx = img_path.find('/extracted/')
                    if extracted_idx != -1:
                        self.base_path = img_path[:extracted_idx+10]  # Include '/extracted/'
        if self.base_path is None:
            self.base_path = ""
        
        self.camera_params_root = os.path.join(self.base_path, "Auxiliary", "Auxiliary", "CameraPos")
        
        # Cache for camera parameters and available camera folders
        self.camera_intrinsics_cache = {}
        self.camera_baseline_cache = {}
        self.camera_poses_cache = {}
        self.available_camera_folders = {}
        
        # Scan for available camera parameter folders
        self._scan_camera_folders()
        
        # Load default camera parameters
        self.default_intrinsics = np.array([
            [480.0, 0.0, 479.5],
            [0.0, 480.0, 269.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.default_baseline = 0.1
        
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
    
    def _scan_camera_folders(self):
        """Scan for available camera parameter folders in the dataset."""
        if not os.path.exists(self.camera_params_root):
            print(f"Warning: Camera parameters root directory not found: {self.camera_params_root}")
            return
        
        # Scan each category folder (Home, Office, Restaurant, Store)
        for category in ['Home', 'Office', 'Restaurant', 'Store']:
            category_path = os.path.join(self.camera_params_root, category)
            if not os.path.exists(category_path):
                continue
            
            # List all scene folders in this category
            try:
                scene_folders = [f for f in os.listdir(category_path) 
                               if os.path.isdir(os.path.join(category_path, f))]
                
                for scene_folder in scene_folders:
                    # Check if Camera.txt exists
                    camera_file = os.path.join(category_path, scene_folder, "Camera.txt")
                    if os.path.exists(camera_file):
                        # Store in available folders dictionary
                        folder_key = f"{category}/{scene_folder}"
                        self.available_camera_folders[folder_key] = {
                            'path': os.path.join(category_path, scene_folder),
                            'camera_file': camera_file,
                            'trace_file': os.path.join(category_path, scene_folder, "Trace.txt")
                        }
            except Exception as e:
                print(f"Error scanning camera folders in {category_path}: {e}")
        
        print(f"Found {len(self.available_camera_folders)} camera parameter folders")
    
    def _find_best_matching_camera_folder(self, main_category, scene_folder):
        """Find the best matching camera folder for a given scene."""
        exact_key = f"{main_category}/{scene_folder}"
        
        # Check for exact match first
        if exact_key in self.available_camera_folders:
            return exact_key
        
        # Try some common variations for scene folders
        # e.g., "DinerEnvironment_Night" might have parameters in "DinerEnvironment_Dark"
        variations = []
        
        # Check for variants with different lighting conditions
        lighting_conditions = ["_Dark", "_Night", "_NightFall", "_BL2", "_T1", "_T2", "_LensFlares"]
        
        # Remove lighting condition suffix if present
        base_name = scene_folder
        for condition in lighting_conditions:
            if scene_folder.endswith(condition):
                base_name = scene_folder[:-len(condition)]
                break
        
        # Generate variations with different lighting conditions
        for condition in lighting_conditions:
            variation = f"{main_category}/{base_name}{condition}"
            if variation in self.available_camera_folders:
                variations.append(variation)
        
        # Also check for base name without any lighting condition
        base_variation = f"{main_category}/{base_name}"
        if base_variation in self.available_camera_folders:
            variations.append(base_variation)
        
        # Return the first matching variation or None if none found
        return variations[0] if variations else None
    
    def _extract_scene_info(self, path):
        """Extract main category and scene folder from an image path."""
        # Expected path format: /path/to/IRS/extracted/Home/Home/ModernClassicInterior/l_480.png
        parts = path.split('/')
        
        # Find main category and scene folder
        for i, part in enumerate(parts):
            if part in ['Home', 'Office', 'Restaurant', 'Store'] and i+2 < len(parts):
                main_category = part
                scene_folder = parts[i+2]  # Skip the duplicated category folder
                return main_category, scene_folder
                
        # If we can't determine, return None
        return None, None
    
    def _get_camera_intrinsics(self, main_category, scene_folder):
        """Load camera intrinsics for the given scene."""
        # Check cache first
        cache_key = f"{main_category}/{scene_folder}"
        if cache_key in self.camera_intrinsics_cache and cache_key in self.camera_baseline_cache:
            return self.camera_intrinsics_cache[cache_key], self.camera_baseline_cache[cache_key]
        
        # Find the best matching camera folder
        camera_key = self._find_best_matching_camera_folder(main_category, scene_folder)
        
        # Default values
        intrinsics = self.default_intrinsics.copy()
        baseline = self.default_baseline
        
        if camera_key:
            camera_file = self.available_camera_folders[camera_key]['camera_file']
            
            # Try to load camera parameters
            try:
                with open(camera_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 4:
                        # Parse first 3 lines for intrinsics matrix
                        for i in range(3):
                            values = lines[i].strip().split()
                            if len(values) == 3:
                                intrinsics[i, 0] = float(values[0])
                                intrinsics[i, 1] = float(values[1])
                                intrinsics[i, 2] = float(values[2])
                        
                        # Parse 4th line for baseline
                        baseline = float(lines[3].strip())
                
            except Exception as e:
                print(f"Error loading camera parameters from {camera_file}: {e}")
                print(f"Using default camera parameters for {cache_key}")
        # else:
        #     print(f"No matching camera parameters found for {cache_key}")
        #     print(f"Using default camera parameters")
        
        intrinsics[0][2] += 0.5
        intrinsics[1][2] += 0.5
        
        # Cache results
        self.camera_intrinsics_cache[cache_key] = intrinsics
        self.camera_baseline_cache[cache_key] = baseline
        
        return intrinsics, baseline
    
    def _get_camera_poses(self, main_category, scene_folder, num_frames):
        """Load camera poses for the given scene."""
        # Check cache first
        cache_key = f"{main_category}/{scene_folder}"
        if cache_key in self.camera_poses_cache:
            poses = self.camera_poses_cache[cache_key]
            # If we have enough poses cached, return them
            if len(poses) >= num_frames:
                return poses[:num_frames]
            # Otherwise, we'll need to extend with identity matrices
            else:
                return poses + [np.eye(4, dtype=np.float32) for _ in range(num_frames - len(poses))]
        
        # Find the best matching camera folder
        camera_key = self._find_best_matching_camera_folder(main_category, scene_folder)
        
        # Initialize poses with identity matrices
        poses = [np.eye(4, dtype=np.float32) for _ in range(num_frames)]
        
        if camera_key:
            trace_file = self.available_camera_folders[camera_key]['trace_file']
            
            # Try to load camera poses
            if os.path.exists(trace_file):
                try:
                    with open(trace_file, 'r') as f:
                        lines = f.readlines()
                        
                        # Each pose takes 5 lines (4 rows + empty line)
                        pose_lines = []
                        current_pose = []
                        
                        for line in lines:
                            line = line.strip()
                            if line:  # Non-empty line
                                current_pose.append(line)
                                if len(current_pose) == 4:  # We have all 4 rows of a pose
                                    pose_lines.append(current_pose)
                                    current_pose = []
                            else:  # Empty line, reset current_pose
                                current_pose = []
                        
                        # Process all poses
                        for i, pose_data in enumerate(pose_lines):
                            if i < num_frames:
                                pose = np.eye(4, dtype=np.float32)
                                for j, line in enumerate(pose_data):
                                    values = line.split()
                                    if len(values) == 4:
                                        pose[j, 0] = float(values[0])
                                        pose[j, 1] = float(values[1])
                                        pose[j, 2] = float(values[2])
                                        pose[j, 3] = float(values[3])
                                        
                                # Define the axis mapping from IRS to OpenCV
                                # IRS: X=forward, -Y=right, Z=up
                                # OpenCV: X=right, Y=down, Z=forward
                                R_transform = np.array([
                                    [0, -1, 0],   # IRS -Y → OpenCV X (right)
                                    [0, 0, -1],    # IRS Z → OpenCV Y (down, since IRS Z is up)
                                    [1, 0, 0]     # IRS X → OpenCV Z (forward)
                                ], dtype=np.float32)
                                
                                # Extract the rotation matrix and translation vector
                                R_irs = pose[:3, :3]
                                t_irs = pose[3, :3]
                                
                                # Apply the transformation
                                R_opencv = np.matmul(R_transform, R_irs)
                                t_opencv = np.matmul(R_transform, t_irs)
                                
                                # Construct the new pose matrix
                                pose_opencv = np.eye(4, dtype=np.float32)
                                pose_opencv[:3, :3] = R_opencv
                                pose_opencv[:3, 3] = t_opencv
                                
                                poses[i] = pose_opencv
                                
                    # print(f"Loaded {len(pose_lines)} camera poses for {cache_key} from {camera_key}")
                except Exception as e:
                    print(f"Error loading camera poses from {trace_file}: {e}")
                    print(f"Using identity matrices for camera poses for {cache_key}")
            else:
                print(f"Trace file not found at {trace_file}")
                print(f"Using identity matrices for camera poses for {cache_key}")
        # else:
        #     print(f"No matching camera parameters found for {cache_key}")
        #     print(f"Using identity matrices for camera poses")
        
        # Cache results
        self.camera_poses_cache[cache_key] = poses
        
        return poses
    
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
            
            # Extract scene name from path (Store/Store/Supermarket_Dark format)
            path_parts = img_path.split('/')
            
            # Find the main category indices
            main_indices = [i for i, part in enumerate(path_parts) 
                           if part in ['Home', 'Office', 'Restaurant', 'Store']]
            
            if len(main_indices) >= 2:  # We have at least 2 category folders
                idx = main_indices[0]  # Use the first occurrence
                scene_name = '/'.join(path_parts[idx:idx+3])  # Include both category folders and the scene
            else:
                # Fallback to last 3 directories
                scene_name = '/'.join(path_parts[-3:])
            
            # Extract main category and scene folder
            main_category, scene_folder = self._extract_scene_info(img_path)
            
            # Extract frame number from image filename (l_480.png format)
            try:
                frame_num = int(path_parts[-1].split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                # Fallback: try to extract any number in the filename
                import re
                numbers = re.findall(r'\d+', path_parts[-1])
                frame_num = int(numbers[0]) if numbers else 0
            
            if scene_name not in original_scenes:
                original_scenes[scene_name] = []
            
            original_scenes[scene_name].append({
                "img_path": img_path,
                "depth_path": depth_path,
                "frame_num": frame_num,
                "main_category": main_category,
                "scene_folder": scene_folder
            })
        
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
                self.scenes[new_scene_name].sort(key=lambda x: x["frame_num"])
    
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
    
    def _convert_disparity_to_depth(self, disparity, baseline, focal_length):
        """Convert disparity to depth using the formula: depth = baseline * focal_length / disparity."""
        # Create a safe copy to avoid modifying the original
        depth = np.zeros_like(disparity)
        
        # Apply the formula only to positive disparity values
        positive_mask = disparity > 0
        depth[positive_mask] = baseline * focal_length / disparity[positive_mask]
        
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
        sky_masks = [] 
        
        # Get representative item for camera parameters
        first_item = group[0]
        main_category = first_item["main_category"]
        scene_folder = first_item["scene_folder"]
        # Load camera intrinsics and baseline
        camera_intrinsics, baseline = self._get_camera_intrinsics(main_category, scene_folder)
        
        # Load camera poses for all frames in the scene
        scene_data = self.scenes[scene_name]
        camera_poses_list = self._get_camera_poses(main_category, scene_folder, len(scene_data))
        
        # Get first frame to determine crop position (if in train mode)
        first_img_path = first_item["img_path"]
        first_depth_path = first_item["depth_path"]
        first_frame_num = first_item["frame_num"]
        
        # Load first image
        first_image = cv2.imread(first_img_path)
        if first_image is None:
            raise FileNotFoundError(f"Failed to read image: {first_img_path}")
        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB) / 255.0
        
        # Read first depth image
        try:
            first_disparity = load_exr(first_depth_path)
            if first_disparity is None:
                raise ValueError("Depth data is None")
        except Exception as e:
            print(f"Failed to read depth: {first_depth_path}, Error: {e}")
            # Create a blank depth image
            h, w, _ = first_image.shape
            first_disparity = np.zeros((h, w), dtype=np.float32)
        
        # Convert disparity to depth
        focal_length = camera_intrinsics[0, 0]  # Assuming fx is used for conversion
        first_depth = self._convert_disparity_to_depth(first_disparity, baseline, focal_length)
        
        # Determine shared crop parameters for the entire group if using crop mode
        h_start, w_start = None, None
        if self.mode == 'train' and self.crop_mode != 'none':
            # Apply transforms up to the crop
            temp_sample = {
                'image': first_image, 
                'depth': first_depth,
                'intrinsics': camera_intrinsics.copy()
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
            
            # Get camera pose for this frame
            # Find the index of this frame in the original scene list
            frame_indices = [i for i, x in enumerate(scene_data) if x["frame_num"] == frame_num]
            if frame_indices:
                frame_idx = frame_indices[0]
                if frame_idx < len(camera_poses_list):
                    camera_pose = camera_poses_list[frame_idx]
                else:
                    camera_pose = np.eye(4, dtype=np.float32)
            else:
                camera_pose = np.eye(4, dtype=np.float32)
            
            # Load and process image
            try:
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError("Image data is None")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            except Exception as e:
                print(f"Failed to read image: {img_path}, Error: {e}")
                # Use a black image as fallback
                image = np.zeros_like(first_image)
            
            # Read disparity image using EXR loader
            try:
                disparity = load_exr(depth_path)
                if disparity is None:
                    raise ValueError("Depth data is None")
            except Exception as e:
                print(f"Failed to read depth: {depth_path}, Error: {e}")
                # Create a blank depth image
                h, w = image.shape[:2] if len(image.shape) >= 2 else (first_image.shape[0], first_image.shape[1])
                disparity = np.zeros((h, w), dtype=np.float32)
            
            # Convert disparity to depth
            depth = self._convert_disparity_to_depth(disparity, baseline, focal_length)
            
            # Create sample dictionary with intrinsics
            sample = {
                'image': image, 
                'depth': depth,
                'intrinsics': camera_intrinsics.copy()
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
            sample_image = torch.from_numpy(sample['image']).float()
            sample_depth = torch.from_numpy(sample['depth']).float()
            K_tensor = torch.from_numpy(sample['intrinsics']).float()
            
            # Generate pointmap
            point_map = generate_pointmap(sample_depth, K_tensor)
            
            # Create valid masks
            valid_mask_disparity = (sample_depth > 0)
            # Define threshold for valid depth (can be adjusted)
            valid_mask = (sample_depth > 0) & (sample_depth <= 80)
            sky_mask = (sample_depth > 80)
            # Convert to disparity if needed
            if self.disparity:
                sample_depth[sample_depth >= 1000] = 0
                positive_mask = sample_depth > 0
                depth_copy = sample_depth.clone()
                depth_copy[positive_mask] = 1.0 / depth_copy[positive_mask]
                sample_depth = depth_copy
                        
            # Store processed tensors
            images.append(sample_image)
            depths.append(sample_depth)
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
            'scene_name': scene_name,
        }
    
    def __len__(self) -> int:
        """Return the number of grouped samples."""
        return len(self.samples)
