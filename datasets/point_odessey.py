from typing import Literal, Optional, Any, override

import pathlib
import json

import albumentations as A
import torch
import torchvision.transforms.v2 as v2
import numpy as np
from PIL import Image

import datasets.base as base

_VALID_SPLITS = ('sample', 'train', 'val', 'test')

# the directory is organised as follows:
# -- split
#    |-- <scene name>
#        |-- rgbs
#            |-- rgb_<:05d>.jpg
#        |-- depths
#            |-- depth_<:05d>.png

class PointOdessey(base.BaseDataset):
    def __init__(self,
                 data_root: pathlib.Path,
                 splits='train',
                 is_video_dataset: bool =True,
                 sequence_len: int = 32,
                 crop_size: int = 518,
                 augmentations: Optional[Any] = None):
        super().__init__(
            max_depth=65536.0,
            min_depth=0.0,
            is_video_dataset=is_video_dataset,
            sequence_len=sequence_len,
            crop_size=crop_size,
            augmentations=augmentations,
            garg_crop=False,
            eigen_crop=False
        )

        if not self.is_video_dataset:
            assert sequence_len == 1, "sequence len > 1 despite being image dataset"

        self.data_root: pathlib.Path = pathlib.Path(data_root)

        self.shape: tuple[int, int] = (540, 960)

        if type(splits) == str:
            self.splits: tuple[str] = (splits,)

        self.scenes: list[tuple[str | pathlib.Path, str | pathlib.Path]] = []
        self.scenes_count: int = 0
        for split in self.splits:
            # for every intermediate split
            for scene in (self.data_root / split).glob("*/"):
                # open info.npz and get number of the frames in the scene
                info_file = np.load(scene / 'info.npz')
                num_scenes: int = info_file['valids'][0]
                # TODO: read other info reg. disparity?
                # we do not read all files directly as this is cheaper on the fs
                self.scenes.extend([
                    (scene / 'rgbs' / f'rgb_{idx:05d}.jpg',
                     scene / 'depths' / f'depth_{idx:05d}.png') for idx in range(num_scenes)
                ])
                self.scenes_count += num_scenes
                # add blank scenes if we do not have anything
                if (num_scenes % self.sequence_len) != 0:
                    rem_scenes = self.sequence_len - (num_scenes % self.sequence_len)
                    self.scenes.extend([
                        ('blank', 'blank') for _ in range(rem_scenes)
                    ])

    def __len__(self) -> int:
        return len(self.scenes) // self.sequence_len

    def get_image_count(self) -> int:
        return self.scenes_count

    @override
    def depth_to_disparity(self, depth: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-5  # small value to prevent division by zero

        # Normalize disparity to [0, 1] for pseudo-invariance
        max_depth = torch.max(depth)
        min_depth = torch.min(depth)

        # we can simplify the formula
        # psuedo-depth = 1/depth
        # and then normalize it all in one go, to prevent weird infinities
        # pointodessey is special: depths 0 are actually invalid depths. no clue who came up with this braindead scheme.
        disparity = torch.where(depth > min_depth,
                                ((max_depth - depth) / (max_depth - min_depth))
                                * (min_depth + epsilon) / (depth + epsilon),
                                0)
        return disparity

    @override                
    def __getitem__(self, idx: int):
        selected_scenes: list[tuple[str | pathlib.Path, str | pathlib.Path]] = self.scenes[idx * self.sequence_len
                                      :(idx + 1) * self.sequence_len]
        if self.is_video_dataset:
            data = [self.__getitem_helper__(img, depth) for img, depth in selected_scenes]
            
            images, depths, masks = zip(*data)
            images = torch.stack(images)
            depths = torch.stack(depths)
            masks = torch.stack(masks)
                    
            return images, depths, masks
        else:
            assert len(selected_scenes) == 1
            scene = selected_scenes[0]
            return self.__getitem_helper__(*scene)

    def __getitem_helper__(self, img: str | pathlib.Path, depth: str | pathlib.Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        match (img, depth):
            case ('blank', 'blank'):
                # print(f"adding padding scenes!")
                # load images and masks as black
                image = np.zeros((*self.shape, 3), dtype=np.float32)
                depth_img = np.zeros(self.shape, dtype=np.float32)
                mask = np.zeros(self.shape, dtype=np.float32)
            case ('blank', _) | (_, 'blank'):
                assert False
            case (_, _):
                image = Image.open(img).convert('RGB')
                depth_img = Image.open(depth)
                
                image = np.array(image)
                depth_img = np.array(depth_img)
                mask = np.ones_like(depth_img)

        image, depth, mask = self.tensors(image, depth_img, mask)
        # do we normalize predictions here?
        # depth /= 65536.0
        # mask[(depth <= self.min_depth) | (depth > self.max_depth)] = 0
        disparity = self.depth_to_disparity(depth)
        mask[(disparity <= 0) | (disparity > 1)] = 0
        image, disparity, mask = self.apply_augmentations(image, disparity, mask)

        return image, disparity, mask

