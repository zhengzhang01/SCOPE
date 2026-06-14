import os
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from scipy import ndimage
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop

class Diode(Dataset):
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
        paths = self.filelist[item].split(' ')
        img_path = paths[0]
        depth_path = paths[1]
        depth_mask_path = paths[2]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = np.load(depth_path)
        mask = np.load(depth_mask_path).astype(np.float32)  # Ensure mask is float for later processing
        
        # Prepare the sample
        sample = {'image': image, 'depth': depth, 'mask': mask}
        # Apply transformations
        sample = self.transform(sample)

        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float().squeeze()
        sample['mask'] = torch.from_numpy(sample['mask']).bool()  # Ensure mask is a boolean tensor

        depth_np = sample['depth'].cpu().numpy()
        dx = ndimage.sobel(depth_np, 0)  # horizontal derivative
        dy = ndimage.sobel(depth_np, 1)  # vertical derivative
        grad = np.abs(dx) + np.abs(dy)
        grad_mask = torch.from_numpy(grad <= 0.3)  # True for valid pixels

        # Generate valid mask and combine with gradient mask
        valid_mask_from_depth = (torch.isnan(sample['depth']) == 0)
        sample['valid_mask'] = valid_mask_from_depth & sample['mask'] & grad_mask

        # Set invalid depth values to 0
        sample['depth'][sample['valid_mask'] == 0] = 0

        sample['image_path'] = img_path.split('metric_depth/')[-1]
        sample['depth_path'] = depth_path.split('metric_depth/')[-1]
        
        return sample

    def __len__(self):
        return len(self.filelist)


class DiodeScene(Dataset):
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
            img_path, depth_path, mask_path = line.strip().split(' ')
            # Extract scene name from img_path
            # Assuming path contains 'scene_{scene_name}/scan_...'
            parts = img_path.split(os.sep)
            try:
                if 'indoors' in parts:
                    scene_index = parts.index('indoors')
                    scene_name = parts[scene_index + 1]  # Get the folder under 'indoors'
                elif 'outdoor' in parts:
                    scene_index = parts.index('outdoor')
                    scene_name = parts[scene_index + 1]  # Get the folder under 'outdoors'
                else:
                    raise ValueError(f"Neither 'indoors' nor 'outdoor' found in path: {img_path}")
            except (ValueError, IndexError):
                raise ValueError(f"Cannot find scene name in path: {img_path}")
            
            if scene_name not in self.scenes:
                self.scenes[scene_name] = []
            self.scenes[scene_name].append((img_path, depth_path, mask_path))

        # Convert scenes dict to list for indexing
        self.scene_names = sorted(self.scenes.keys())
        self.scenes_list = [self.scenes[scene] for scene in self.scene_names]

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
                'depth_paths': List of depth paths
                'num_images': Number of images in the scene
        """
        scene = self.scenes_list[idx]  # List of (img_path, depth_path, mask_path)
        images = []
        depths = []
        valid_masks = []
        image_paths = []
        depth_paths = []

        for img_path, depth_path, mask_path in scene:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

            # Load depth and mask
            depth = np.load(depth_path)
            mask = np.load(mask_path).astype(np.float32)  # Ensure mask is float for later processing

            # Prepare the sample
            sample = {'image': image, 'depth': depth, 'mask': mask}

            # Apply transformations
            transformed = self.transform(sample)
            transformed_image = transformed['image']
            transformed_depth = transformed['depth']
            transformed_mask = transformed['mask']

            # Convert to torch tensors
            transformed_image = torch.from_numpy(transformed_image).float()  # [3, H, W]
            transformed_depth = torch.from_numpy(transformed_depth).float().squeeze()  # [H, W]
            transformed_mask = torch.from_numpy(transformed_mask).bool()  # [H, W] Ensure mask is boolean

            # Calculate depth gradients
            depth_np = transformed_depth.cpu().numpy()
            dx = ndimage.sobel(depth_np, 0)  # horizontal derivative
            dy = ndimage.sobel(depth_np, 1)  # vertical derivative
            grad = np.abs(dx) + np.abs(dy)
            grad_mask = torch.from_numpy(grad <= 0.3)  # True for valid pixels

            # Create valid mask and combine with gradient mask
            valid_mask_from_depth = (torch.isnan(transformed_depth) == 0)
            valid_mask = valid_mask_from_depth & transformed_mask & grad_mask

            # Set invalid depth values to 0
            transformed_depth[valid_mask == 0] = 0.0

            # Collect data
            images.append(transformed_image)
            depths.append(transformed_depth)
            valid_masks.append(valid_mask)
            image_paths.append(img_path)
            depth_paths.append(depth_path)

        # Stack along new dimension (N, C, H, W)
        images = torch.stack(images, dim=0)       # [N, 3, H, W]
        depths = torch.stack(depths, dim=0)       # [N, H, W]
        valid_masks = torch.stack(valid_masks, dim=0)  # [N, H, W]

        sample = {
            'image': images,
            'depth': depths,
            'valid_mask': valid_masks,
            'image_paths': image_paths,
            'depth_paths': depth_paths,
            'num_images': len(images) ,
            'idx': idx,
        }
        
        return sample

    def __len__(self):
        return len(self.scenes_list)
