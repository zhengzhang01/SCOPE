import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
import numpy as np
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet


class DDAD(Dataset):
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
        # Parse the file paths
        paths = self.filelist[item].split(' ')
        img_path = paths[0]
        depth_path = paths[1]

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        # Read depth map
        depth = np.load(depth_path)

        # Prepare the sample
        sample = {'image': image, 'depth': depth}
        sample = self.transform(sample)

        # Convert to torch tensors
        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float().squeeze()

        # Generate valid mask from the depth map
        valid_mask = (torch.isnan(sample['depth']) == 0)
        sample['valid_mask'] = valid_mask

        # Set invalid depth values to 0
        sample['depth'][sample['valid_mask'] == 0] = 0

        # Store the relative paths for later use
        sample['image_path'] = img_path.split('ddad/')[-1]
        sample['depth_path'] = depth_path.split('ddad/')[-1]

        return sample

    def __len__(self):
        return len(self.filelist)


class DDADScene(Dataset):
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

            # Extract scene name from the img_path (e.g., 000150)
            # Assuming path contains 'ddad/val/{scene_name}/...'
            parts = img_path.split(os.sep)
            try:
                scene_index = parts.index('val') + 1  # The folder after 'val' is the scene name
                scene_name = parts[scene_index]
            except (ValueError, IndexError):
                raise ValueError(f"Cannot find scene name in path: {img_path}")

            if scene_name not in self.scenes:
                self.scenes[scene_name] = []
            self.scenes[scene_name].append((img_path, depth_path))

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
        scene = self.scenes_list[idx]  # List of (img_path, depth_path)
        images = []
        depths = []
        valid_masks = []
        image_paths = []
        depth_paths = []

        for img_path, depth_path in scene:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

            # Load depth map
            depth = np.load(depth_path)

            # Prepare the sample
            sample = {'image': image, 'depth': depth}
            transformed = self.transform(sample)

            # Convert to torch tensors
            transformed_image = torch.from_numpy(transformed['image']).float()  # [3, H, W]
            transformed_depth = torch.from_numpy(transformed['depth']).float().squeeze()  # [H, W]

            # Create valid mask
            valid_mask = (torch.isnan(transformed_depth) == 0)

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
            'num_images': len(images),
            'idx': idx, 
        }

        return sample

    def __len__(self):
        return len(self.scenes_list)
