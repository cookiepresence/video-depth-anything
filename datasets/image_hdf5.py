import pathlib

import h5py
import torch

class ImageHdf5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path: pathlib.Path):
        self.file_path = file_path
        self.file = h5py.File(file_path, "r")
        self.length = len(self.file["images"])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file = h5py.File(self.file_path, "r")

        img = torch.from_numpy(file["images"][idx])
        disparity_pred = torch.from_numpy(file["disparity_preds"][idx])
        mask = torch.from_numpy(file["masks"][idx])

        file.close()
        return img, disparity_pred, mask
