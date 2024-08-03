"""
Class for loading a dataset of images
"""
import os

import torch

from torch.utils.data import Dataset

import torchvision.transforms as transforms

from PIL import Image

opj = os.path.join


class ImageDataset(Dataset):
    """
    Class for loading up a folder of images in (x, y) pairs, where y is a HR
    patch from the image and x is the downsampled LR version.
    """

    def __init__(self, root_dir: str, patch_size: int):
        """
        Params
        ------
        root_dir: str
            Directory containing folders `train` and `test`
        patch_size: int
            Size of the random crop taken for y

        Returns
        -------
        None
        """

        self.patch_size = patch_size
        self.root_dir = root_dir
        self.image_paths = [
            opj(root_dir, x) for x in os.listdir(root_dir) if '.DS' not in x
        ]

    def __len__(self):
        return len(self.image_paths)

    def transform(self, full_image: Image) -> (torch.Tensor, torch.Tensor):
        """
        Function to preprocess an image for training
        - Convert to a torch.Tensor
        - Downsample input to obtain (x, y) pair
        - Normalize to [-1, 1] range

        Params
        ------
        full_image: PIL Image to downsample

        Returns
        -------
        x: torch.Tensor
            LR input image
        y: torch.Tensor
            HR ground truth image
        """

        # Randomly crop part of the input image
        crop_func = transforms.RandomCrop(self.patch_size)

        # Downsample crop by factor of 4
        resize_func = transforms.Resize(self.patch_size // 4)

        to_tensor = transforms.ToTensor()

        full_image = to_tensor(full_image)

        y = crop_func(full_image)
        x = resize_func(y)

        # Normalize to [-1, 1] range
        x = (x * 2) - 1.
        y = (y * 2) - 1.

        return x, y

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(Image.open(self.image_paths[idx]))
