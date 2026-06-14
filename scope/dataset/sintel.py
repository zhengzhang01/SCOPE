import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet

# class SintelScene(Dataset):
#     def __init__(self, filelist_path, mode, size=(518, 518)):
#         if mode != 'val':
#             raise NotImplementedError("Only 'val' mode is implemented for now.")
        
#         self.mode = mode
#         self.size = size
#         self.scenes = {}

#         # Read and parse the filelist
#         with open(filelist_path, 'r') as f:
#             lines = f.read().splitlines()

#         for line in lines:
#             if not line.strip():
#                 continue  # Skip empty lines
#             img_path, depth_path = line.strip().split(' ')
#             # Extract scene name from img_path
#             parts = img_path.split(os.sep)
#             try:
#                 # The second-to-last part in the path is the scene name
#                 scene_name = parts[-2]  # For example, "alley_1"
#             except IndexError:
#                 raise ValueError(f"Cannot find scene name in path: {img_path}")
            
#             if scene_name not in self.scenes:
#                 self.scenes[scene_name] = []
#             self.scenes[scene_name].append((img_path, depth_path))

#         # Convert scenes dict to list for indexing
#         self.scene_names = sorted(self.scenes.keys())
#         self.scenes_list = [self.scenes[scene] for scene in self.scene_names]
        
#         net_w, net_h = size
#         self.transform = Compose([
#             Resize(
#                 width=net_w,
#                 height=net_h,
#                 resize_target=True if mode == 'train' else False,
#                 keep_aspect_ratio=True,
#                 ensure_multiple_of=14,
#                 resize_method='lower_bound',
#                 image_interpolation_method=cv2.INTER_CUBIC,
#             ),
#             NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             PrepareForNet(),
#         ])
        
#     def __getitem__(self, idx):
#         """
#         Returns:
#             sample (dict): 
#                 'image': Tensor of shape [N, 3, H, W]
#                 'depth': Tensor of shape [N, H, W]
#                 'valid_mask': Tensor of shape [N, H, W]
#                 'image_paths': List of image paths
#                 'num_images': Number of images in the scene
#         """
#         scene = self.scenes_list[idx]  # List of (img_path, depth_path)
#         images = []
#         depths = []
#         valid_masks = []
#         image_paths = []

#         for img_path, depth_path in scene:
#             # Load image
#             image = cv2.imread(img_path)
#             if image is None:
#                 raise FileNotFoundError(f"Image not found: {img_path}")
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

#             # Load depth
#             with open(depth_path, "rb") as f:
#                 check = np.fromfile(f, dtype=np.float32, count=1)[0]
#                 if check != 202021.25:
#                     raise ValueError(f"Wrong tag in depth file: {depth_path}")
#                 width = np.fromfile(f, dtype=np.int32, count=1)[0]
#                 height = np.fromfile(f, dtype=np.int32, count=1)[0]
#                 depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))

#             # Apply transforms
#             transformed = self.transform({'image': image, 'depth': depth})
#             transformed_image = transformed['image']
#             transformed_depth = transformed['depth'] 

#             # Convert to torch tensors
#             transformed_image = torch.from_numpy(transformed_image).float()  # [3, H, W]
#             transformed_depth = torch.from_numpy(transformed_depth).float()  # [H, W]

#             # Create valid mask
#             valid_mask = transformed_depth > 0

#             images.append(transformed_image)
#             depths.append(transformed_depth)
#             valid_masks.append(valid_mask)
#             image_paths.append(img_path)

#         # Stack along new dimension (N, C, H, W)
#         images = torch.stack(images, dim=0)       # [N, 3, H, W]
#         depths = torch.stack(depths, dim=0)       # [N, H, W]
#         valid_masks = torch.stack(valid_masks, dim=0)  # [N, H, W]

#         sample = {
#             'image': images,
#             'depth': depths,
#             'valid_mask': valid_masks,
#             'image_paths': image_paths,
#             'num_images': len(images),
#             'idx': idx,
#         }
        
#         return sample

#     def __len__(self):
#         return len(self.scenes_list)

# Define the tag for cam file format validation
TAG_FLOAT = 202021.25

class SintelScene(Dataset):
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
            parts = img_path.split(os.sep)
            try:
                # The second-to-last part in the path is the scene name
                scene_name = parts[-2]  # For example, "alley_1"
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
                'intrinsics': Tensor of shape [N, 3, 3]
                'extrinsics': Tensor of shape [N, 4, 4] (c2w)
        """
        scene = self.scenes_list[idx]  # List of (img_path, depth_path)
        images = []
        depths = []
        valid_masks = []
        image_paths = []
        intrinsics = []
        extrinsics = []

        for img_path, depth_path in scene:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

            # Load depth
            with open(depth_path, "rb") as f:
                check = np.fromfile(f, dtype=np.float32, count=1)[0]
                if check != 202021.25:
                    raise ValueError(f"Wrong tag in depth file: {depth_path}")
                width = np.fromfile(f, dtype=np.int32, count=1)[0]
                height = np.fromfile(f, dtype=np.int32, count=1)[0]
                depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))

            # Derive pose path from img_path
            # Replace "clean" with "camdata_left" and change extension from "png" to "cam"
            pose_path = img_path.replace('clean', 'camdata_left').replace('.png', '.cam')
            
            # Load camera pose
            intrinsic, extrinsic = self.cam_read(pose_path)
            
            # Convert w2c extrinsic to c2w by extending to 4x4 and taking inverse
            extrinsic_4x4 = np.eye(4)
            extrinsic_4x4[:3, :] = extrinsic
            extrinsic_c2w = np.linalg.inv(extrinsic_4x4)

            # Apply transforms
            transformed = self.transform({'image': image, 'depth': depth})
            transformed_image = transformed['image']
            transformed_depth = transformed['depth'] 

            # Convert to torch tensors
            transformed_image = torch.from_numpy(transformed_image).float()  # [3, H, W]
            transformed_depth = torch.from_numpy(transformed_depth).float()  # [H, W]
            intrinsic_tensor = torch.from_numpy(intrinsic).float()  # [3, 3]
            extrinsic_c2w_tensor = torch.from_numpy(extrinsic_c2w).float()  # [4, 4]

            # Create valid mask
            valid_mask = transformed_depth > 0

            images.append(transformed_image)
            depths.append(transformed_depth)
            valid_masks.append(valid_mask)
            image_paths.append(img_path)
            intrinsics.append(intrinsic_tensor)
            extrinsics.append(extrinsic_c2w_tensor)

        # Stack along new dimension (N, C, H, W)
        images = torch.stack(images, dim=0)       # [N, 3, H, W]
        depths = torch.stack(depths, dim=0)       # [N, H, W]
        valid_masks = torch.stack(valid_masks, dim=0)  # [N, H, W]
        intrinsics = torch.stack(intrinsics, dim=0)  # [N, 3, 3]
        extrinsics = torch.stack(extrinsics, dim=0)  # [N, 4, 4]

        sample = {
            'image': images,
            'depth': depths,
            'valid_mask': valid_masks,
            'image_paths': image_paths,
            'num_images': len(images),
            'intrinsics': intrinsics,
            'poses': extrinsics,
            'idx': idx,
        }
        
        return sample

    def __len__(self):
        return len(self.scenes_list)
    
    def cam_read(self, filename):
        """ Read camera data, return (M,N) tuple.
        
        M is the intrinsic matrix, N is the extrinsic matrix, so that
        
        x = M*N*X,
        where x is a point in homogeneous image pixel coordinates, and X is a
        point in homogeneous world coordinates.
        """
        with open(filename, 'rb') as f:
            check = np.fromfile(f, dtype=np.float32, count=1)[0]
            assert check == TAG_FLOAT, f'cam_read:: Wrong tag in cam file (should be: {TAG_FLOAT}, is: {check}). Big-endian machine?'
            M = np.fromfile(f, dtype='float64', count=9).reshape((3, 3))
            N = np.fromfile(f, dtype='float64', count=12).reshape((3, 4))
        return M, N
