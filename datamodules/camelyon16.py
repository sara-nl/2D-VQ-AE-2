from __future__ import annotations

import bisect
import functools
from functools import reduce
from operator import xor
from dataclasses import dataclass
from itertools import chain, product, starmap, zip_longest
from pathlib import Path
from typing import Optional, Tuple
from collections.abc import Sequence

import numpy as np
import pytorch_lightning as pl
from albumentations import BasicTransform
from hydra.utils import instantiate
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from utils.conf_helpers import DataloaderConf
from wsi_io.imagereader import ImageReader


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
    train: str  # TODO: replace with Enum
    train_frac: float

    def __post_init__(self):
        modality_folders = ('images', 'tissue_masks')
        modality_postfixes = ('', '_tissue')

        find_pairs = functools.partial(
            _find_image_mask_pairs_paths,
            path=self.path,
            modality_folders=modality_folders,
            modality_postfixes=modality_postfixes
        )

        self.image_paths, self.tissue_mask_paths = (
            find_pairs(pattern='test')
            if self.train == 'test'
            else _train_val_split_paths(
                modality_arrays=tuple(  # type: ignore
                    find_pairs(pattern=pattern)
                    for pattern in ('normal', 'tumor')
                ),
                split_frac=self.train_frac,
                mode=self.train
            )
        )

        self.n_wsi = len(self.image_paths)
        self._length = self.n_wsi * self.n_patches_per_wsi

    def __cascade_sampler(self, image: ImageReader, mask: ImageReader) -> np.ndarray:

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
        discrepancy = round(spacing / target_spacing)  # noqa
        idx = idx * discrepancy + rng.integers(discrepancy, size=2)

        return image.read_center(target_spacing, *idx, *self.patch_size, normalized=True)

    def __getitem__(self, index: int) -> np.ndarray:
        wsi_index = index % self.n_wsi

        image, mask = (
            ImageReader(modality_paths[wsi_index], self.spacing_tolerance)
            for modality_paths in (self.image_paths, self.tissue_mask_paths)
        )

        patch = self.__cascade_sampler(image, mask)

        return patch if self.transforms is None else self.transforms(image=patch)['image']

    def __len__(self) -> int:
        return self._length


class CAMELYON16SlicePatchDataSet(Dataset):
    """
    A dataset object of non-overlapping patches from Camelyon16
    """

    def __init__(
        self,
        path: str,
        spacing: float,
        spacing_tolerance: float,
        patch_size: Tuple[int, int],
        transforms: Optional[BasicTransform],
        train: str,  # TODO: replace with enum
        train_frac: float,
        **throwaway_kwargs
    ):
        modality_folders = ('images', 'masks')
        modality_postfixes = ('', '_mask')

        find_pairs = functools.partial(
            _find_image_mask_pairs_paths,
            path=path,
            modality_folders=modality_folders,
            modality_postfixes=modality_postfixes
        )

        self.image_paths, self.mask_paths = (
            find_pairs(pattern='test')
            if train == 'test'
            else _train_val_split_paths(
                modality_arrays=tuple(  # type: ignore
                    find_pairs(pattern=pattern)
                    for pattern in ('normal', 'tumor')
                ),
                split_frac=train_frac,
                mode=train
            )
        )

        self.spacing = spacing
        self.spacing_tolerance = spacing_tolerance
        self.patch_size = patch_size
        self.transforms = transforms

    def __len__(self):
        return self._cum_lengths[-1]

    @functools.cached_property
    def _lengths(self) -> np.ndarray:  # np.ndarray[...]
        return self._sizes.prod(axis=-1)

    @functools.cached_property
    def _cum_lengths(self) -> np.ndarray:
        return np.cumsum(self._lengths)

    @functools.cached_property
    def _sizes(self) -> np.ndarray:  # np.ndarray[..., 2]
        return np.asarray(
            [
                (img := ImageReader(img_path, self.spacing_tolerance)).shapes[
                    img.level(self.spacing)]
                for img_path in self.image_paths
            ]
        ) // self.patch_size

    @functools.lru_cache(maxsize=1)
    def _get_paths(self, index):
        return self.image_paths[index], self.mask_paths[index]

    def __getitem__(self, index) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get a non-overlapping image patch of size self.image_patch

        :return:
            - Image patch
            - Image patch labels
            -
            - Patch indices w.r.t. the original image
        """
        img_index: np.int = bisect.bisect(self._cum_lengths, index)
        patch_index = index - (self._cum_lengths[img_index-1] if img_index != 0 else 0)

        patch_indices = np.asarray((
            patch_index // self._sizes[img_index, 1],
            patch_index %  self._sizes[img_index, 1]  # noqa[E222]
        ))

        paths = self._get_paths(img_index)
        patch, labels = (
            ImageReader(path, self.spacing_tolerance).read(
                self.spacing,
                *(patch_indices * self.patch_size),  # row, col
                *self.patch_size,  # height, width,
                normalized=True
            )
            for path in paths
        )

        return (  # type: ignore
            *((patch, labels)
              if self.transforms is None
              else self.transforms(image=patch, mask=labels)).values(),
            (img_index, patch_indices, *paths)
        )


def _train_val_split_paths(
    modality_arrays: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    split_frac: float,
    mode: str,  # train or validation
) -> Tuple[np.ndarray, np.ndarray]:  # Tuple[images, masks]
    """
    practically: merge n lists of (possibly unequal) size, also based on `split_frac`:
    [(a, a), (b, b)], [(c,),(d,)] into
    [(a, c), (a,)],   [(b, d), (b,)]

    This function is obviously extremely overengineered.
    """
    assert mode in ('train', 'validation')

    temp = []
    for images, masks in modality_arrays:

        length, split_index = (
            (ln := len(images)),
            tf if 0 < (tf := round(ln * split_frac)) < ln
            else 1 if tf == 0
            else tf - 1
        )

        temp.append(
            [
                elem[(slice(split_index) if mode == 'train' else slice(split_index, length))]
                for elem in (images, masks)
            ]
        )

    return tuple(  # type: ignore
        map(
            lambda x: np.asarray(list(filter(None, chain.from_iterable(zip_longest(*x))))),
            zip(*temp)
        )
    )


def _find_image_mask_pairs_paths(
    path: str,
    pattern: str,
    modality_folders: Sequence[str] = ('images', 'masks', 'tissue_masks'),
    modality_postfixes: Sequence[str] = ('', '_mask', '_tissue')
) -> Sequence[np.ndarray]:
    assert len(modality_folders) == len(modality_postfixes)

    paths = [
        list(Path(path).glob(f'{modality}/*{pattern}*.tif'))
        for modality in modality_folders
    ]

    def assert_matching_paths(modality_paths, postfixes):
        """asserting that image and mask names line up"""

        def refine_scan_path(scan_path, postfix):
            if len(postfix) == 0:
                return scan_path.stem

            if not scan_path.stem[-len(postfix):] == postfix:
                raise ValueError(f"{scan_path.stem} does not have postfix {postfix}")

            return scan_path.stem[:-len(postfix)]

        scan_names = [
            set(refine_scan_path(scan_path, postfix) for scan_path in modality_path)
            for modality_path, postfix in zip(modality_paths, postfixes)
        ]

        # Practically: (A ^ B) ^ (A ^ C) ^ ...
        difference = reduce(xor, (starmap(xor, product((scan_names[0],), scan_names[1:]))))

        if len(difference) != 0:
            raise ValueError(
                f"Scan(s) {difference} do not have all associated modalities. "
                f"Please check these are present in all {modality_folders} subfolders"
                f"with name postfixes {modality_postfixes}"
            )

    if len(paths) > 1:
        assert_matching_paths(paths, modality_postfixes)

    return tuple(  # type: ignore
        np.sort(list(map(str, modality_paths)))
        for modality_paths in paths
    )
