from typing import Sequence, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from itertools import chain, zip_longest

import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from albumentations import BasicTransform

from wsi_io.imagereader import ImageReader
from utils.conf_helpers import DataloaderConf


@dataclass
class DefaultDataModule(pl.LightningDataModule):

    train_dataloader_conf: DataloaderConf
    val_dataloader_conf: DataloaderConf
    test_dataloader_conf: Optional[DataloaderConf] = None

    def __post_init__(self):
        super().__init__()

    def train_dataloader(self) -> DataLoader:
        return instantiate(self.train_dataloader_conf)

    def val_dataloader(self) -> DataLoader:
        return instantiate(self.val_dataloader_conf)

    def test_dataloader(self) -> DataLoader:
        return instantiate(self.test_dataloader_conf)


@dataclass
class CAMELYON16RandomPatchDataSet(Dataset):
    path: str
    spacing: float
    spacing_tolerance: float
    patch_size: Tuple[int, int]
    n_patches_per_wsi: int
    transforms: Optional[BasicTransform]
    train: str
    train_frac: float

    def __post_init__(self):

        def merge_scan_paths(*scan_types: Sequence) -> Tuple[np.array, np.array]:
            '''
            practically: merge n lists of unequal size, also based on `self.train_frac`:
            [(a, a), (b, b)], [(c,),(d,)] into
            [(a, c), (a,)],   [(b, d), (b,)]

            This function is obviously extremely overengineered.
            '''
            assert self.train in ('train', 'validation')

            paths = list(map(self._find_image_mask_pairs_paths, scan_types))
            for i, path in enumerate(paths):

                length, frac = (ln := len(path[0])), round(ln*self.train_frac)
                paths[i] = [
                    elem[(slice(frac) if self.train == 'train' else slice(frac, length))]
                    for elem in path
                ]

            return tuple(map(
                lambda x: np.asarray(list(filter(None, chain.from_iterable(zip_longest(*x))))),
                zip(*paths)
            ))


        self.image_paths, self.mask_paths = (
            self.__find_image_mask_pairs_paths(pattern='test')
            if self.train == 'test'
            else merge_scan_paths('normal', 'tumor')
        )

        self.n_wsi = len(self.image_paths)
        self._length = self.n_wsi * self.n_patches_per_wsi


    def _find_image_mask_pairs_paths(self, pattern: str) -> Tuple[np.array, np.array]:
        image_paths, mask_paths = (
            list(Path(self.path).glob(f'{modality}/*{pattern}*.tif'))
            for modality in ('images', 'tissue_masks')
        )

        def assert_matching_paths(image_paths, mask_paths):
            '''asserting that image and mask names line up'''
            fault = False
            mask_name_set = set(map(lambda path: path.stem[:-7], mask_paths))
            for image_name in map(lambda path: path.stem, image_paths):
                try:
                    mask_name_set.remove(image_name)
                except KeyError:
                    fault = True
                    print(f"Error: {image_name} does not have an associated tissue mask!")
            for mask_name in iter(mask_name_set): # if any mask is left-over
                fault = True
                print(f"Error: {mask_name} does not have an associated WSI!")
            if fault:
                raise ValueError('WSI/Mask mismatch.')

        assert_matching_paths(image_paths, mask_paths)

        return tuple(
            np.sort(list(map(str, modality_paths)))
            for modality_paths in (image_paths, mask_paths)
        )

    def __cascade_sampler(self, image: ImageReader, mask: ImageReader) -> np.array:
        rng = np.random.default_rng()

        start_level = -1
        size = mask.shapes[start_level]
        idx = np.array([0, 0])

        target_spacing = image.refine(self.spacing)
        assert target_spacing <= mask.spacings[start_level]

        for spacing in mask.spacings[start_level::-1]:
            idx *= 2

            options = np.asarray(np.where(mask.read(spacing, *idx, *size, normalized=True))[:2])
            idx += rng.choice(options, axis=-1)

            if np.isclose(spacing, target_spacing):
                break

            size = (2, 2)

        # NOP if spacing == target_spacing
        discrepancy = round(spacing / target_spacing)
        idx = idx * discrepancy + rng.integers(discrepancy, size=2)

        return image.read_center(target_spacing, *idx, *self.patch_size)

    def __getitem__(self, index: int) -> np.array:
        wsi_index = index % self.n_wsi

        image, mask = (
            ImageReader(modality_paths[wsi_index], self.spacing_tolerance)
            for modality_paths in (self.image_paths, self.mask_paths)
        )

        patch = self.__cascade_sampler(image, mask)

        return patch if self.transforms is None else self.transforms(image=patch)['image']

    def __len__(self) -> int:
        return self._length
