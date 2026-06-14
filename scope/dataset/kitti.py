import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet


class KITTI(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        if mode != 'val':
            raise NotImplementedError
        
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
        ])
    
    def __getitem__(self, item):
        img_path = self.filelist[item].split(' ')[0]
        depth_path = self.filelist[item].split(' ')[1]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32')
        
        sample = self.transform({'image': image, 'depth': depth})
        
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['depth'] = sample['depth'] / 256.0  # convert in meters
        
        sample['valid_mask'] = sample['depth'] > 0
        
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)


class KITTIScene(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        if mode != 'val':
            raise NotImplementedError("Only 'val' mode is implemented for now.")
        
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
            # Assuming path contains 'kitti/{date_drive}/image_02/data/...'
            parts = img_path.split(os.sep)
            try:
                # The second-to-last part in the path is the scene name
                scene_name = parts[-4]  # "2011_09_26_drive_0002_sync"
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
        ])
        
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
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32')

            # Apply transforms
            transformed = self.transform({'image': image, 'depth': depth})
            transformed_image = transformed['image']
            transformed_depth = transformed['depth'] / 256.0  # Convert depth to meters

            # Convert to torch tensors
            transformed_image = torch.from_numpy(transformed_image).float()  # [3, H, W]
            transformed_depth = torch.from_numpy(transformed_depth).float()  # [H, W]

            # Create valid mask
            valid_mask = transformed_depth > 0

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
