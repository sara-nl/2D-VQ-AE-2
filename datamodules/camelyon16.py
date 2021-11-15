import bisect
import functools
from dataclasses import dataclass
from itertools import chain, zip_longest
from pathlib import Path
from typing import Optional, Tuple

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
    train: str
    train_frac: float

    def __post_init__(self):
        self.image_paths, self.mask_paths = (
            _find_image_mask_pairs_paths(self.path, pattern='test')
            if self.train == 'test'
            else _train_val_split_paths(
                modality_arrays=tuple(  # type: ignore
                    _find_image_mask_pairs_paths(self.path, pattern=pattern)
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
            for modality_paths in (self.image_paths, self.mask_paths)
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
        train: str,
        train_frac: float,
        **throwaway_kwargs
    ):
        self.image_paths, _ = (
            _find_image_mask_pairs_paths(self.path, pattern='test')
            if train == 'test'
            else _train_val_split_paths(
                modality_arrays=tuple(  # type: ignore
                    _find_image_mask_pairs_paths(path, pattern=pattern)
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

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a non-overlapping image patch of size self.image_patch

        :param index:
        :return:
            - Image patch
            - Patch indices w.r.t. the original image
        """
        img_index: np.int = bisect.bisect_left(self._cum_lengths, index)
        patch_index = index - (self._cum_lengths[img_index-1] if index != 0 else 0)

        patch_indices = np.asarray((
            patch_index // self._sizes[img_index, 1],
            patch_index %  self._sizes[img_index, 0]  # noqa[E222]
        )) * self.patch_size

        return (  # type: ignore
            ImageReader(self.image_paths[img_index], self.spacing_tolerance).read(
                self.spacing,
                *patch_indices,  # row, col
                *self.patch_size,  # height, width,
                normalized=True
            ),
            patch_indices
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
    modalities: Tuple[str, str] = ('images', 'tissue_masks')
) -> Tuple[np.ndarray, np.ndarray]:
    image_paths, mask_paths = (
        list(Path(path).glob(f'{modality}/*{pattern}*.tif'))
        for modality in modalities
    )

    def assert_matching_paths(image_paths_, mask_paths_):
        """asserting that image and mask names line up"""
        fault = False
        mask_name_set = set(map(lambda path_: path_.stem[:-7], mask_paths_))
        for image_name in map(lambda path_: path_.stem, image_paths_):
            try:
                mask_name_set.remove(image_name)
            except KeyError:
                fault = True
                print(f"Error: {image_name} does not have an associated tissue mask!")
        for mask_name in iter(mask_name_set):  # if any mask is left-over
            fault = True
            print(f"Error: {mask_name} does not have an associated WSI!")
        if fault:
            raise ValueError('WSI/Mask mismatch.')

    assert_matching_paths(image_paths, mask_paths)

    return tuple(  # type: ignore
        np.sort(list(map(str, modality_paths)))
        for modality_paths in (image_paths, mask_paths)
    )
