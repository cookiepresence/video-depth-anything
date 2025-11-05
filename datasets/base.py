import pathlib
from itertools import chain

from typing import Optional, Literal, Callable

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as v2


def enumerate_paths(src):
    '''flatten out a nested dictionary into an iterable
    
    DIODE metadata is a nested dictionary;
    One could easily query a particular scene and scan, but sequentially
    enumerating files in a nested dictionary is troublesome. This function
    recursively traces out and aggregates the leaves of a tree.
    '''
    if isinstance(src, list):
        return src
    elif isinstance(src, dict):
        acc = []
        for k, v in src.items():
            k = pathlib.Path(k)
            _sub_paths = enumerate_paths(v)
            _sub_paths = list(map(lambda x: k / x, _sub_paths))
            acc.append(_sub_paths)
        return list(chain.from_iterable(acc))
    else:
        raise ValueError('do not accept data type {}'.format(type(src)))


class Resize(torch.nn.Module):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width: int,
        height: int,
        resize_target: Optional[bool] = True,
        keep_aspect_ratio: Optional[bool] = False,
        ensure_multiple_of: int = 1,
        resize_method: Literal['lower_bound', 'upper_bound', 'minimal'] = "lower_bound",
        image_interpolation_method = cv2.INTER_AREA,
    ):
        """
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

        if not self.__keep_aspect_ratio:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        match self.__resize_method:
            case "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height

                new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
                new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
            case "upper_bound":
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height

                new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
                new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)

            case "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height

                new_height = self.constrain_to_multiple_of(scale_height * height)
                new_width = self.constrain_to_multiple_of(scale_width * width)
            case _:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])
        # resize sample
        sample["image"] = cv2.resize(sample["image"], (width, height), interpolation=self.__image_interpolation_method)

        if self.__resize_target:
            if "depth" in sample:
                sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)

            if "disparity" in sample:
                sample["disparity"] = cv2.resize(sample["disparity"], (width, height), interpolation=cv2.INTER_NEAREST)
                
            if "mask" in sample:
                sample["mask"] = cv2.resize(sample["mask"].astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        
        return sample


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_len, min_depth, max_depth, crop_size, augmentations, garg_crop, eigen_crop, is_video_dataset):
        self.sequence_len: int = sequence_len
        self.min_depth: int = min_depth
        self.max_depth: int = max_depth
        self.crop_size: int = crop_size
        self.garg_crop: bool = garg_crop
        self.eigen_crop: bool = eigen_crop

        assert not (self.garg_crop and self.eigen_crop), "both garg crop and eigen crop should not be true"

        # assume that we are training, turn it off incase of evaluations
        self.training = True
        # assume that we need to send video frames instead of image frames
        self.is_video_dataset = is_video_dataset

        self.augmentations: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] = lambda x: x
        # if augmentations is None:
        #     # Define augmentation pipeline
        #     if self.is_video_dataset:
        #         self.augmentations = v2.Compose([
        #             v2.RandomHorizontalFlip(p=0.5),
        #         ])
        #     else:
        #         # borrowed from: https://github.com/aharley/pips2/blob/master/datasets/pointodysseydataset.py#L132
        #         self.color_augmentations = A.ReplayCompose([
        #             A.GaussNoise(p=0.2),
        #             A.OneOf([
        #                 A.MotionBlur(p=0.2),
        #                 A.MedianBlur(blur_limit=3, p=0.1),
        #                 A.Blur(blur_limit=3, p=0.1),
        #             ], p=0.2),
        #             A.OneOf([
        #                 A.CLAHE(clip_limit=2),
        #                 A.Sharpen(),
        #                 A.Emboss(),
        #             ], p=0.2),
        #             A.RGBShift(p=0.5),
        #             A.RandomBrightnessContrast(p=0.5),
        #             A.RandomGamma(p=0.5),
        #             A.HueSaturationValue(p=0.3),
        #             A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
        #         ], p=0.8)
        #         self.augmentations = v2.Compose([
        #             v2.RandomHorizontalFlip(p=0.5),
        #         ])
        # else:
        #     self.augmentations = augmentations

        self.resize_transform = Resize(
            width=self.crop_size,
            height=self.crop_size,
            resize_target=True,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        )

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def tensors(self, im, depth, mask) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training:
            resized_outputs = self.resize_transform({
                "image": im,
            })
            im = resized_outputs['image']
        else:
            resized_outputs = self.resize_transform({
                "image": im,
                "depth": depth,
                "mask": mask
            })
            im = resized_outputs['image']
            depth = resized_outputs['depth']
            mask = resized_outputs['mask']
        
        im = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])(im)  
        depth = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])(depth)
        mask = v2.Compose([v2.ToImage(), v2.ToDtype(torch.bool)])(mask)

        # remove channel dims from the image
        depth = depth.squeeze(0)
        mask = mask.squeeze(0)

        # ensure that all depth is within ranges
        # mask = torch.logical_and(mask, torch.logical_and(depth > self.min_depth, depth < self.max_depth))
        
        return im, depth, mask
    
    def apply_augmentations(self, image, disparity, mask):
        """Apply data augmentations for robustness using torchvision.transforms.v2"""
        if not self.training:
            # apply garg/eigen crop during evaluation
            # constants were found through experimental trial and error
            eval_mask = torch.zeros_like(disparity)
            h, w = disparity.shape
            if self.garg_crop:
                eval_mask[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)] = 1
            elif self.eigen_crop:
                eval_mask[int(0.3324324 * h):int(0.91351351 * h), int(0.0359477 * w):int(0.96405229 * w)] = 1
            else:
                eval_mask = torch.ones_like(disparity)
            mask = torch.logical_and(mask, eval_mask)
            return image, disparity, mask
        
        # Apply same geometric transforms to all modalities
        # For v2 transforms, we can pass multiple tensors
        # TODO: remove this and replace with proper augmentations
        augmented = self.augmentations({
            'image': image,
            'disparity': disparity, 
            'mask': mask
        })
    
        return augmented['image'], augmented['disparity'], augmented['mask']

    def depth_to_disparity(self, depth):
        epsilon = 1e-5  # small value to prevent division by zero

        # Normalize disparity to [0, 1] for pseudo-invariance
        max_depth = torch.max(depth)
        min_depth = torch.min(depth)

        # we can simplify the formula
        # psuedo-depth = 1/depth
        # and then normalize it all in one go, to prevent weird infinities
        depth_mask = (depth == min_depth)
        disparity = torch.zeros_like(depth)
        disparity = torch.where(depth > min_depth,
                                ((max_depth - depth) / (max_depth - min_depth)) * (min_depth + epsilon) / (depth + epsilon),
                                1)
        return disparity
