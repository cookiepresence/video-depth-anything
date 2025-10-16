import json
import pathlib
from itertools import chain

import torch
import torchvision.transforms.v2 as v2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import datasets.base as base
'''
The json metadata for DIODE is laid out as follows:
train:
    outdoor:
        scene_000xx:
            scan_00yyy:
                - 000xx_00yyy_indoors_300_010
                - 000xx_00yyy_indoors_300_020
                - 000xx_00yyy_indoors_300_030
        scene_000kk:
            _analogous_
val:
    _analogous_
test:
    _analogous_
'''

_VALID_SPLITS = ('train', 'val', 'test')
_VALID_SCENE_TYPES = ('indoors', 'outdoor')


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


def check_and_tuplize_tokens(tokens, valid_tokens):
    if not isinstance(tokens, (tuple, list)):
        tokens = (tokens, )
    for split in tokens:
        assert split in valid_tokens
    return tokens


def plot_depth_map(dm, validity_mask, idx):
    validity_mask = validity_mask > 0
    MIN_DEPTH = 0.5
    MAX_DEPTH = min(300, np.percentile(dm, 99))
    dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
    dm = np.log(dm, where=validity_mask)
    # dm = (dm - np.min(dm)) / (MAX_DEPTH - MIN_DEPTH)

    dm = np.ma.masked_where(~validity_mask, dm)

    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    plt.imshow(dm, cmap=cmap, vmax=np.log(MAX_DEPTH))
    plt.savefig(f"artifacts/{idx}_gt.png")


def plot_normal_map(normal_map):
    normal_viz = normal_map[:, ::, :]

    normal_viz = normal_viz + np.equal(np.sum(normal_viz, 2,
    keepdims=True), 0.).astype(np.float32)*np.min(normal_viz)

    normal_viz = (normal_viz - np.min(normal_viz))/2.
    plt.axis('off')
    plt.imshow(normal_viz)


class DIODE(base.BaseDataset):
    def __init__(self,
                 data_root,
                 meta_fname='diode_meta.json',
                 splits='train',
                 scene_types=['indoors', 'outdoor'],
                 sequence_len: int = 32,
                 crop_size: int = 518,
                 augmentations = None):
        # TODO: ensure that these depths are accurate
        super().__init__(
            max_depth=310,
            min_depth=0,
            sequence_len=sequence_len,
            crop_size=crop_size,
            augmentations=augmentations,# use default augmentations
            garg_crop=False,
            eigen_crop=False
        )
        self.data_root = pathlib.Path(data_root)
        self.splits = check_and_tuplize_tokens(
            splits, _VALID_SPLITS
        )
        self.scene_types = check_and_tuplize_tokens(
            scene_types, _VALID_SCENE_TYPES
        )
        with open(meta_fname, 'r') as f:
            self.meta = json.load(f)

        imgs = []
        for split in self.splits:
            split_p = pathlib.Path(split)
            for scene_type in self.scene_types:
                _curr = enumerate_paths(self.meta[split][scene_type])
                _curr = map(lambda x: (split_p / scene_type / x), _curr)
                imgs.extend(list(_curr))
        self.imgs = imgs

        self.sequence_len = sequence_len

    def __len__(self):
        return len(self.imgs) // self.sequence_len

    def __get_individual_frame__(self, index):
        im = self.imgs[index]
        im_fname = self.data_root / f'{im}.png'
        de_fname = self.data_root / f'{im}_depth.npy'
        de_mask_fname = self.data_root / f'{im}_depth_mask.npy'

        im = np.array(Image.open(self.data_root / im_fname))
        de = np.load(de_fname).squeeze()
        de_mask = np.load(de_mask_fname)

        # plot_depth_map(de, de_mask, index)

        im, de, de_mask = self.tensors(im, de, de_mask)
        disparity = self.depth_to_disparity(de)
        im, disparity, de_mask = self.apply_augmentations(im, disparity, de_mask)

        return im, disparity, de_mask
        # More TODOS:
        # -- fix eval pipeline to test if the evals work
        # -- run evals on video and image models to compare
        # -- email hpc for dataset space

    def __getitem__(self, index):
        img_index = index * self.sequence_len
        frames = [self.__get_individual_frame__(i) for i in range(img_index, img_index + self.sequence_len)]
        im, de, de_mask = zip(*frames)

        # ensure that we have proper tensors
        im = torch.stack(im)
        de = torch.stack(de)
        de_mask = torch.stack(de_mask)

        return im, de, de_mask
