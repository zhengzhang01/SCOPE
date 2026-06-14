import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import random
from PIL import Image


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """Rezise the sample to ensure the given size. Keeps aspect ratio.

    Args:
        sample (dict): sample
        size (tuple): image size

    Returns:
        tuple: new size
    """
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # resize
    sample["image"] = cv2.resize(
        sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    )

    sample["disparity"] = cv2.resize(
        sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    )
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize(object):
    """Resize sample to given size (width, height).
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
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

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
        # determine new height and width
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
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

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
                # sample["semseg_mask"] = cv2.resize(
                #     sample["semseg_mask"], (width, height), interpolation=cv2.INTER_NEAREST
                # )
                sample["semseg_mask"] = F.interpolate(torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...], (height, width), mode='nearest').numpy()[0, 0]
                
            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                # sample["mask"] = sample["mask"].astype(bool)

        # print(sample['image'].shape, sample['depth'].shape)
        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
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
    """Crop sample for batch-wise training. Image is of shape CxHxW.
    Can use fixed positions for consistent cropping across video frames.
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
            
        return sample, (self.h_start, self.w_start)
    
    
class CenterCrop(object):
    """Crop sample at the center. Image is of shape CxHxW."""
    
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
            
        return sample, (h_start, w_start)
    
    
class ColorJitter(object):
    def __init__(self, p=0.0, strength=1.0):
        """
        Args:
            p (float): probability of applying this augmentation [0.0, 1.0]
            strength (float): overall strength multiplier for all color adjustments
                            strength=1.0 corresponds to original settings
                            strength=2.0 would double all ranges
        Base ratios:
            brightness = 0.2 * strength
            contrast = 0.2 * strength
            saturation = 0.2 * strength
            hue = 0.1 * strength
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
    
    Base settings (strength=1.0):
        kernel_size = 7
        sigma_min = 0.1
        sigma_max = 2.0
    
    For any strength value:
        sigma_min = 0.1 * strength
        sigma_max = 2.0 * strength
        kernel_size = 2 * round(3 * sigma_max) + 1
    """
    def __init__(self, p=0.5, strength=1.0):
        """
        Args:
            p (float): probability of applying blur [0.0, 1.0]
            strength (float): overall strength multiplier for blur effect
                            strength=1.0 corresponds to original settings
                            strength=2.0 would double sigma range and adjust kernel
        """
        self.p = p
        # Calculate sigma range based on strength
        self.sigma_min = 0.1 * strength
        self.sigma_max = 2.0 * strength
        
        # Automatically adjust kernel size based on max sigma
        # Making it odd number for symmetric blur
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
    
    Args:
        depth (torch.Tensor): Depth map [H, W]
        K (torch.Tensor): Camera intrinsics matrix [3, 3]
    
    Returns:
        torch.Tensor: Pointmap [3, H, W]
    """
    H, W = depth.shape
    # Create pixel coordinate grid
    y, x = torch.meshgrid(torch.arange(H, device=depth.device),
                          torch.arange(W, device=depth.device),
                          indexing='ij')
    x = x.float()
    y = y.float()
    
    # Convert pixel coordinates to 3D points
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Calculate XYZ coordinates
    Z = depth
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    
    # Stack to create pointmap
    pointmap = torch.stack([X, Y, Z], dim=0)
    
    return pointmap

def adjust_intrinsics_for_resize(K, original_size, new_size):
    """
    Adjust camera intrinsics for resize operation.
    
    Args:
        K (np.ndarray): Original camera intrinsics matrix [3, 3]
        original_size (tuple): Original image size (H, W)
        new_size (tuple): New image size after resize (H, W)
    
    Returns:
        np.ndarray: Adjusted camera intrinsics
    """
    # Calculate scaling factors
    scale_y = new_size[0] / original_size[0]
    scale_x = new_size[1] / original_size[1]
    
    K_new = K.copy()
    # Adjust focal length
    K_new[0, 0] *= scale_x  # fx
    K_new[1, 1] *= scale_y  # fy
    # Adjust principal point
    K_new[0, 2] *= scale_x  # cx
    K_new[1, 2] *= scale_y  # cy
    
    return K_new

def adjust_intrinsics_for_crop(K, crop_start):
    """
    Adjust camera intrinsics for crop operation.
    
    Args:
        K (np.ndarray): Camera intrinsics matrix before crop [3, 3]
        crop_start (tuple): Starting position of crop (y_start, x_start)
    
    Returns:
        np.ndarray: Adjusted camera intrinsics
    """
    h_start, w_start = crop_start
    
    K_new = K.copy()
    # Adjust principal point by subtracting crop start
    K_new[0, 2] -= w_start  # cx
    K_new[1, 2] -= h_start  # cy
    
    return K_new
