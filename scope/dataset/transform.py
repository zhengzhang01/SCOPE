import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import random
from PIL import Image

class Resize(object):
    """Resize sample to given size (width, height) with integrated intrinsics adjustment.
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
        target_area=None,  # New parameter for 'none' crop mode
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size.
                "minimal": Scale as least as possible.
                "area": Scale to maintain aspect ratio while achieving target area.
                Defaults to "lower_bound".
            target_area (int, optional):
                Target area for 'area' resize method. Default is width*height.
        """
        self.__width = width
        self.__height = height
        self.__target_area = target_area if target_area is not None else (width * height * 4 / 3)

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # For the special 'area' method (none crop mode)
        if self.__resize_method == "area":
            # Calculate the scale to achieve target area while maintaining aspect ratio
            current_area = width * height
            scale = np.sqrt(self.__target_area / current_area)
            
            # Calculate new dimensions while maintaining aspect ratio
            new_height = self.constrain_to_multiple_of(height * scale)
            new_width = self.constrain_to_multiple_of(width * scale)
            
            # Fine-tune to get closer to target area
            area_error = abs(new_width * new_height - self.__target_area)
            best_error = area_error
            best_width, best_height = new_width, new_height
                        
            return (best_width, best_height)
        
        # Original logic for other methods
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        # Save original size before resize
        original_height, original_width = sample["image"].shape[:2]
        
        # Calculate new size
        width, height = self.get_size(original_width, original_height)

        # Adjust camera intrinsics if present in the sample
        if "intrinsics" in sample:
            K = sample["intrinsics"].copy()
            
            # Calculate scaling factors
            scale_y = height / original_height
            scale_x = width / original_width
            
            # Handle all elements of the intrinsics matrix, including skew
            # K = [ fx  s  cx ]
            #     [ 0  fy  cy ]
            #     [ 0   0   1 ]
            # Adjust focal length
            K[0, 0] *= scale_x  # fx
            K[1, 1] *= scale_y  # fy
            K[0, 1] *= scale_x  # s (skew)
            # Adjust principal point
            K[0, 2] *= scale_x  # cx
            K[1, 2] *= scale_y  # cy
            
            sample["intrinsics"] = K

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            if "semseg_mask" in sample:
                sample["semseg_mask"] = F.interpolate(torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...], (height, width), mode='nearest').numpy()[0, 0]
                
            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

        return sample


class NormalizeImage(object):
    """Normalize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std
        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])
        
        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)
            
        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(sample["semseg_mask"])

        return sample


class Crop(object):
    """Crop sample for batch-wise training with integrated intrinsics adjustment.
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.h_start = None
        self.w_start = None

    def get_crop_params(self, h, w):
        """Generate random crop parameters."""
        assert h >= self.size[0] and w >= self.size[1], 'Wrong size'
        h_start = np.random.randint(0, h - self.size[0] + 1)
        w_start = np.random.randint(0, w - self.size[1] + 1)
        return h_start, w_start

    def __call__(self, sample, h_start=None, w_start=None):
        h, w = sample['image'].shape[-2:]
        
        # Use provided positions or generate new ones
        if h_start is None or w_start is None:
            self.h_start, self.w_start = self.get_crop_params(h, w)
        else:
            self.h_start, self.w_start = h_start, w_start
            
        h_end = self.h_start + self.size[0]
        w_end = self.w_start + self.size[1]
        
        sample['image'] = sample['image'][:, self.h_start:h_end, self.w_start:w_end]
        
        if "depth" in sample:
            sample["depth"] = sample["depth"][self.h_start:h_end, self.w_start:w_end]
        
        if "mask" in sample:
            sample["mask"] = sample["mask"][self.h_start:h_end, self.w_start:w_end]
            
        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"][self.h_start:h_end, self.w_start:w_end]
        
        # Adjust camera intrinsics if present in the sample
        if "intrinsics" in sample:
            K = sample["intrinsics"].copy()
            # Adjust principal point by subtracting crop start
            K[0, 2] -= self.w_start  # cx
            K[1, 2] -= self.h_start  # cy
            sample["intrinsics"] = K
            
        return sample


class CenterCrop(object):
    """Crop sample at the center with integrated intrinsics adjustment.
    """
    
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, sample, h_start=None, w_start=None):
        h, w = sample['image'].shape[-2:]
        
        # Calculate center crop coordinates
        if h_start is None or w_start is None:
            h_start = (h - self.size[0]) // 2
            w_start = (w - self.size[1]) // 2
        
        h_end = h_start + self.size[0]
        w_end = w_start + self.size[1]
        
        sample['image'] = sample['image'][:, h_start:h_end, w_start:w_end]
        
        if "depth" in sample:
            sample["depth"] = sample["depth"][h_start:h_end, w_start:w_end]
        
        if "mask" in sample:
            sample["mask"] = sample["mask"][h_start:h_end, w_start:w_end]
            
        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"][h_start:h_end, w_start:w_end]
        
        # Adjust camera intrinsics if present in the sample
        if "intrinsics" in sample:
            K = sample["intrinsics"].copy()
            # Adjust principal point by subtracting crop start
            K[0, 2] -= w_start  # cx
            K[1, 2] -= h_start  # cy
            sample["intrinsics"] = K
            
        return sample


class ColorJitter(object):
    def __init__(self, p=0.0, strength=1.0):
        """
        Args:
            p (float): probability of applying this augmentation [0.0, 1.0]
            strength (float): overall strength multiplier for all color adjustments
        """
        self.p = p
        hue_value = min(0.1 * strength, 0.5)  # Cap hue at 0.5
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2 * strength,
            contrast=0.2 * strength,
            saturation=0.2 * strength, 
            hue=hue_value)

    def __call__(self, sample):
        if random.random() < self.p:
            sample["image"] = np.array(self.color_jitter(Image.fromarray(
                (sample["image"] * 255).astype(np.uint8)))) / 255.
        return sample


class GaussianBlur(object):
    """Enhanced Gaussian blur with unified strength control.
    """
    def __init__(self, p=0.5, strength=1.0):
        """
        Args:
            p (float): probability of applying blur [0.0, 1.0]
            strength (float): overall strength multiplier for blur effect
        """
        self.p = p
        self.sigma_min = 0.1 * strength
        self.sigma_max = 2.0 * strength
        
        computed_kernel = 2 * round(3 * self.sigma_max) + 1
        self.kernel_size = min(computed_kernel, 31)
        
    def __call__(self, sample):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma_min, self.sigma_max)
            sample["image"] = cv2.GaussianBlur(
                sample["image"],
                (self.kernel_size, self.kernel_size),
                sigma
            )
        return sample


def generate_pointmap(depth, K):
    """
    Generate pointmap (XYZ coordinates) from depth map and camera intrinsics.
    Handles non-zero skew parameter in camera matrix.
    
    Args:
        depth (torch.Tensor): Depth map [H, W]
        K (torch.Tensor): Camera intrinsics matrix [3, 3]
    
    Returns:
        torch.Tensor: Pointmap [3, H, W]
    """
    H, W = depth.shape
    
    # Create pixel coordinate grid (with pixel centers at x+0.5, y+0.5)
    y, x = torch.meshgrid(torch.arange(H, device=depth.device),
                          torch.arange(W, device=depth.device),
                          indexing='ij')
    x = x.float() + 0.5
    y = y.float() + 0.5
    
    # Extract camera intrinsics parameters
    fx = K[0, 0]  # Focal length x
    fy = K[1, 1]  # Focal length y
    cx = K[0, 2]  # Principal point x
    cy = K[1, 2]  # Principal point y
    s = K[0, 1]   # Skew parameter
    
    # Calculate Z (depth)
    Z = depth
    
    # Calculate X and Y with skew correction
    # In the presence of skew, we need to modify the x-coordinate calculation
    # to account for the non-perpendicular axes
    if torch.abs(s) > 1e-10:  # Only apply skew correction if skew is non-negligible
        # The skew affects how the y-coordinate influences the x-coordinate
        X = ((x - cx) - s * (y - cy) / fy) * Z / fx
        Y = (y - cy) * Z / fy
    else:
        # Standard pinhole model (no skew)
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
    
    # Stack to create pointmap
    pointmap = torch.stack([X, Y, Z], dim=0)
    
    return pointmap