import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
import os.path as osp
from scope.dataset.transform import Resize, NormalizeImage, PrepareForNet
from PIL import Image

class ScanNetScene(Dataset):
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
            
            # Extract scene name and frame ID from paths
            img_parts = img_path.split(os.sep)
            scene_name = img_parts[-3]  # Assuming format like "scannet/scans_train/scene0001/color/000001.jpg"
            frame_id = os.path.splitext(img_parts[-1])[0]  # Extract frame number without extension
            
            # Construct paths for pose and intrinsics
            # Assuming standard ScanNet structure:
            # scannet_dir/
            #   ├── scans_train/
            #   │   ├── scene0001/
            #   │   │   ├── color/
            #   │   │   │   ├── 0.jpg, 1.jpg, ...
            #   │   │   ├── depth/
            #   │   │   │   ├── 0.png, 1.png, ...
            #   │   │   ├── pose/
            #   │   │   │   ├── 0.txt, 1.txt, ...
            #   │   │   ├── intrinsic/
            #   │   │   │   ├── intrinsic_color.txt
            #   │   │   │   ├── intrinsic_depth.txt
            
            # Get the base directory by removing 'color/frame_id.jpg' from img_path
            scene_dir = os.path.dirname(os.path.dirname(img_path))
            
            # Construct pose path
            pose_path = osp.join(scene_dir, 'pose', f"{frame_id}.txt")
            
            # Intrinsic paths (same for all frames in a scene)
            color_intrinsic_path = osp.join(scene_dir, 'intrinsic', 'intrinsic_color.txt')
            depth_intrinsic_path = osp.join(scene_dir, 'intrinsic', 'intrinsic_depth.txt')
            
            if scene_name not in self.scenes:
                # First time seeing this scene, load intrinsics
                try:
                    color_intrinsic = np.loadtxt(color_intrinsic_path)[:3, :3].astype(np.float32)
                    depth_intrinsic = np.loadtxt(depth_intrinsic_path)[:3, :3].astype(np.float32)
                    
                    # Only add scene if intrinsics are valid
                    if np.isfinite(color_intrinsic).all() and np.isfinite(depth_intrinsic).all():
                        self.scenes[scene_name] = {
                            'frames': [],
                            'color_intrinsic': color_intrinsic,
                            'depth_intrinsic': depth_intrinsic
                        }
                    else:
                        print(f"Warning: Invalid intrinsics for scene {scene_name}, skipping")
                        continue
                except Exception as e:
                    print(f"Error loading intrinsics for scene {scene_name}: {str(e)}")
                    continue
            
            # Add frame to scene
            self.scenes[scene_name]['frames'].append({
                'img_path': img_path,
                'depth_path': depth_path,
                'pose_path': pose_path
            })

        # Filter out scenes with no valid frames
        self.scenes = {k: v for k, v in self.scenes.items() if v['frames']}
        
        # Convert scenes dict to list for indexing
        self.scene_names = sorted(self.scenes.keys())
        self.scenes_list = [self.scenes[scene] for scene in self.scene_names]

        # Define transformations
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
        scene_data = self.scenes_list[idx]
        
        # Get intrinsics (same for all frames in the scene)
        color_intrinsic = scene_data['color_intrinsic']
        depth_intrinsic = scene_data['depth_intrinsic']
        
        # Process frames
        frames = scene_data['frames']
        images = []
        depths = []
        valid_masks = []
        image_paths = []
        poses = []
        intrinsics = []

        for frame in frames:
            img_path = frame['img_path']
            depth_path = frame['depth_path']
            pose_path = frame['pose_path']
            
            # Load image
            try:
                image = Image.open(img_path).convert('RGB')
                image = np.array(image) / 255.0  # Normalize to [0, 1]

                # Load depth
                depth_png = np.asarray(Image.open(depth_path))
                if np.max(depth_png) <= 255:
                    print(f"Warning: Depth map might not be valid: {depth_path}, max value: {np.max(depth_png)}")
                depth = depth_png.astype(np.float64) / 1000.0  # Convert to meters
                
                # Load pose
                try:
                    pose = np.loadtxt(pose_path).reshape(4, 4).astype(np.float32)
                    if not np.isfinite(pose).all():
                        print(f"Warning: Invalid pose in {pose_path}, using identity")
                        pose = np.eye(4, dtype=np.float32)
                except Exception as e:
                    print(f"Error loading pose from {pose_path}: {str(e)}")
                    pose = np.eye(4, dtype=np.float32)
                
                # Use depth intrinsics for 3D calculations
                K = depth_intrinsic.copy()
                
                # Apply transforms
                transformed = self.transform({'image': image, 'depth': depth})
                transformed_image = transformed['image']
                transformed_depth = transformed['depth']
                
                transformed_image = torch.from_numpy(transformed_image).float()  # [3, H, W]
                transformed_depth = torch.from_numpy(transformed_depth).float()  # [H, W]

                # Create valid mask
                valid_mask = ~torch.isnan(transformed_depth)
                transformed_depth[~valid_mask] = 0.0
                            
                # Convert to tensors
                K_tensor = torch.from_numpy(K)
                pose_tensor = torch.from_numpy(pose)
                
                # Add to lists
                images.append(transformed_image)
                depths.append(transformed_depth)
                valid_masks.append(valid_mask)
                image_paths.append(img_path)
                intrinsics.append(K_tensor)
                poses.append(pose_tensor)
                
            except Exception as e:
                print(f"Error processing frame {img_path}: {str(e)}")
                continue

        # Check if we have any valid frames
        if not images:
            # Create dummy data if no valid frames
            dummy_image = torch.zeros((3, self.size[1], self.size[0]), dtype=torch.float32)
            dummy_depth = torch.zeros((self.size[1], self.size[0]), dtype=torch.float32)
            dummy_mask = torch.zeros((self.size[1], self.size[0]), dtype=torch.bool)
            dummy_K = torch.eye(3, dtype=torch.float32)
            dummy_pose = torch.eye(4, dtype=torch.float32)
            
            images = [dummy_image]
            depths = [dummy_depth]
            valid_masks = [dummy_mask]
            image_paths = ["dummy_path"]
            intrinsics = [dummy_K]
            poses = [dummy_pose]
            print(f"Warning: No valid frames for scene {self.scene_names[idx]}, returning dummy data")

        # Stack along new dimension
        images = torch.stack(images, dim=0)       # [N, 3, H, W]
        depths = torch.stack(depths, dim=0)       # [N, H, W]
        valid_masks = torch.stack(valid_masks, dim=0)  # [N, H, W]
        intrinsics = torch.stack(intrinsics, dim=0)  # [N, 3, 3]
        poses = torch.stack(poses, dim=0)  # [N, 4, 4]

        sample = {
            'image': images,
            'depth': depths,
            'valid_mask': valid_masks,
            'image_paths': image_paths,
            'intrinsics': intrinsics,  # Camera intrinsics
            'poses': poses,            # Camera-to-world poses
            'num_images': len(images),
            'idx': idx,
            'scene_name': self.scene_names[idx]
        }

        return sample

    def __len__(self):
        return len(self.scenes_list)
