import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
import torch
from torchvision import tv_tensors
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_url,
)
from torchvision.transforms import functional as F
from tqdm import tqdm


import datasets.base as base


def download_and_extract_file(filename, raw_url, data_root):
    logging.info("%s", raw_url + filename)
    download_and_extract_archive(
        raw_url + filename,
        download_root=data_root,
        extract_root=data_root / "raw",
        md5=None,
    )
        

class KITTIDepth(base.BaseDataset):
    data_root: Path
    depth_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
    depth_md5 = "7d1ce32633dc2f43d9d1656a1f875e47"
    raw_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
    raw_filenames_url = "https://raw.githubusercontent.com/torch-uncertainty/dataset-metadata/main/download/kitti/raw_filenames.json"
    raw_filenames_md5 = "e5b7fad5ecd059488ef6c02dc9e444c1"
    _num_samples = {
        "train": 42949,
        "val": 3426,
        "test": ...,
    }

    def __init__(
        self,
        data_root: str | Path,
        splits: Literal["train", "val"],
        min_depth: float = 0.0,
        # capped max_depth to match Eigen et. al's experiments
        max_depth: float = 80.0,
        augmentations = None,
        download: bool = False,
        remove_unused: bool = False,
        crop_size: int = 518,
        sequence_len: int = 32
    ) -> None:
        logging.info(
            "KITTIDepth is copyrighted by the Karlsruhe Institute of Technology "
            "(KIT) and the Toyota Technological Institute at Chicago (TTIC). "
            "By using KITTIDepth, you agree to the terms and conditions of the "
            "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License. "
            "This means that you must attribute the work in the manner specified "
            "by the authors, you may not use this work for commercial purposes "
            "and if you alter, transform, or build upon this work, you may "
            "distribute the resulting work only under the same license."
        )

        super().__init__(
            min_depth=min_depth,
            max_depth=max_depth,
            sequence_len=32,
            crop_size=crop_size,
            augmentations=augmentations,
            eigen_crop=False,
            garg_crop=True # following MDE toolkit setup
        )
        self.data_root = Path(data_root)

        if splits not in ["train", "val"]:
            raise ValueError(f"split must be one of ['train', 'val']. Got {split}.")

        self.split = splits

        if not self.check_split_integrity("left_depth"):
            if download:
                self._download_depth()
            else:
                raise FileNotFoundError(
                    f"KITTI {split} split not found or incomplete. Set download=True to download it."
                )

        if not self.check_split_integrity("left_img"):
            if download:
                self._download_raw(remove_unused)
            else:
                raise FileNotFoundError(
                    f"KITTI {split} split not found or incomplete. Set download=True to download it."
                )

        self._make_dataset()

    def check_split_integrity(self, folder: str) -> bool:
        split_path = self.data_root / self.split
        return (
            split_path.is_dir()
            and len(list((split_path / folder).glob("*.png"))) == self._num_samples[self.split]
        )

    def __getitem__(self, index: int) -> tuple:
        """Get the sample at the given index.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a depth map.
        """
        image = Image.open(self.samples[index]).convert("RGB")
        depth = Image.open(self.targets[index])

        image = np.array(image)
        depth = np.array(depth)
        mask = np.ones_like(depth)

        image, depth, mask = self.tensors(image, depth, mask)
        # convert values to metres for more accurate predictions
        depth /= 256.0
        mask[(depth <= self.min_depth) | (depth >= self.max_depth)] = 0
        disparity = self.depth_to_disparity(torch.where(mask, depth, 0.0))
        image, disparity, mask = self.apply_augmentations(image, disparity, mask)

        return image, disparity, mask

    def __len__(self) -> int:
        """The number of samples in the dataset."""
        return self._num_samples[self.split] // self.sequence_len

    def _make_dataset(self) -> None:
        self.samples = sorted((self.data_root / self.split / "left_img").glob("*.png"))
        self.targets = sorted((self.data_root / self.split / "left_depth").glob("*.png"))

    def _download_depth(self) -> None:
        """Download and extract the depth annotation dataset."""
        if not (self.data_root / "tmp").exists():
            download_and_extract_archive(
                self.depth_url,
                download_root=self.data_root,
                extract_root=self.data_root / "tmp",
                md5=self.depth_md5,
            )

        logging.info("Re-structuring the depth annotations...")

        if (self.data_root / "train" / "left_depth").exists():
            shutil.rmtree(self.data_root / "train" / "left_depth")

        (self.data_root / "train" / "left_depth").mkdir(parents=True, exist_ok=False)

        depth_files = list((self.data_root).glob("**/tmp/train/**/image_02/*.png"))
        logging.info("Train files...")
        for file in tqdm(depth_files):
            exp_code = file.parents[3].name.split("_")
            filecode = "_".join([exp_code[0], exp_code[1], exp_code[2], exp_code[4], file.name])
            shutil.copy(file, self.data_root / "train" / "left_depth" / filecode)

        if (self.data_root / "val" / "left_depth").exists():
            shutil.rmtree(self.data_root / "val" / "left_depth")

        (self.data_root / "val" / "left_depth").mkdir(parents=True, exist_ok=False)

        depth_files = list((self.data_root).glob("**/tmp/val/**/image_02/*.png"))
        logging.info("Validation files...")
        for file in tqdm(depth_files):
            exp_code = file.parents[3].name.split("_")
            filecode = "_".join([exp_code[0], exp_code[1], exp_code[2], exp_code[4], file.name])
            shutil.copy(file, self.data_root / "val" / "left_depth" / filecode)

        shutil.rmtree(self.data_root / "tmp")

    
    def _download_raw(self, remove_unused: bool) -> None:
        """Download and extract the raw dataset."""
        download_url(
            self.raw_filenames_url,
            self.data_root,
            "raw_filenames.json",
            self.raw_filenames_md5,
        )
        with (self.data_root / "raw_filenames.json").open() as file:
            raw_filenames = json.load(file)

        # attempt to parallelize the process
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(download_and_extract_file, filename, self.raw_url, self.data_root): filename 
                       for filename in raw_filenames}
    
            for future in tqdm(as_completed(futures), total=len(raw_filenames)):
                try:
                    filename = future.result()
                except Exception as e:
                    logging.error(f"Failed to download {futures[future]}: {e}")

        logging.info("Re-structuring the raw data...")

        samples_to_keep = list((self.data_root / "train" / "left_depth").glob("*.png"))

        if (self.data_root / "train" / "left_img").exists():
            shutil.rmtree(self.data_root / "train" / "left_img")

        (self.data_root / "train" / "left_img").mkdir(parents=True, exist_ok=False)

        logging.info("Train files...")
        for sample in tqdm(samples_to_keep):
            filecode = sample.name.split("_")
            first_level = "_".join([filecode[0], filecode[1], filecode[2]])
            second_level = "_".join(
                [
                    filecode[0],
                    filecode[1],
                    filecode[2],
                    "drive",
                    filecode[3],
                    "sync",
                ]
            )
            raw_path = (
                self.data_root / "raw" / first_level / second_level / "image_02" / "data" / filecode[4]
            )
            shutil.copy(raw_path, self.data_root / "train" / "left_img" / sample.name)

        samples_to_keep = list((self.data_root / "val" / "left_depth").glob("*.png"))

        if (self.data_root / "val" / "left_img").exists():
            shutil.rmtree(self.data_root / "val" / "left_img")

        (self.data_root / "val" / "left_img").mkdir(parents=True, exist_ok=False)

        logging.info("Validation files...")
        for sample in tqdm(samples_to_keep):
            filecode = sample.name.split("_")
            first_level = "_".join([filecode[0], filecode[1], filecode[2]])
            second_level = "_".join(
                [
                    filecode[0],
                    filecode[1],
                    filecode[2],
                    "drive",
                    filecode[3],
                    "sync",
                ]
            )
            raw_path = (
                self.data_root / "raw" / first_level / second_level / "image_02" / "data" / filecode[4]
            )
            shutil.copy(raw_path, self.data_root / "val" / "left_img" / sample.name)

        if remove_unused:
            shutil.rmtree(self.data_root / "raw")
