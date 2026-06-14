import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet
from PIL import Image

class BonnScene(Dataset):
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
            parts = img_path.split(os.sep)
            try:
                scene_name = parts[-3]  # Assuming scene name is the third last component
            except IndexError:
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
            depth = self.depth_read(depth_path)

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

    def depth_read(self, filename):
        """
        Loads depth map from a PNG file and returns it as a numpy array.
        """
        depth_png = np.asarray(Image.open(filename))
        # Make sure we have a proper 16-bit depth map
        if np.max(depth_png) <= 255:
            raise ValueError(f"Depth map is not 16-bit: {filename}")
        depth = depth_png.astype(np.float64) / 5000.0
        #depth[depth_png == 0] = np.nan
        return depth
