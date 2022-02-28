import bisect
import functools
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import chain, product, starmap, zip_longest
from operator import xor, attrgetter
from pathlib import Path
from typing import Optional, Tuple

import albumentations.pytorch
import h5py
import numpy as np
import torch
from albumentations import BasicTransform
from torch.utils.data import Dataset, DataLoader

from wsi_io.imagereader import ImageReader


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

        self.image_paths, self.tissue_mask_paths = map(np.asarray, (
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
        ))

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
        modality_folders = ('images', 'masks', 'tissue_masks')
        modality_postfixes = ('', '_mask', '_tissue')

        find_pairs = functools.partial(
            _find_image_mask_pairs_paths,
            path=path,
            modality_folders=modality_folders,
            modality_postfixes=modality_postfixes
        )

        self.image_paths, self.mask_paths, self.tissue_mask_paths = map(np.asarray, (
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
        ))

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
        return self.image_paths[index], self.mask_paths[index], self.tissue_mask_paths[index]

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
        patch_index = index - (self._cum_lengths[img_index - 1] if img_index != 0 else 0)

        patch_indices = np.asarray((
            patch_index // self._sizes[img_index, 1],
            patch_index % self._sizes[img_index, 1]  # noqa[E222]
        ))

        paths = self._get_paths(img_index)[:2]  # img, mask, _
        patch, label = (
            ImageReader(path, self.spacing_tolerance).read(
                self.spacing,
                *(patch_indices * self.patch_size),  # row, col
                *self.patch_size,  # height, width,
                normalized=True
            )
            for path in paths
        )

        return (  # type: ignore
            *((patch, label)
              if self.transforms is None
              else (
                (transformed := self.transforms(image=patch, mask=label))['image'],
                transformed['mask']
            )),
            (img_index, patch_indices, *paths)
        )


class CAMELYON16EmbeddingsDataset(Dataset):
    """
    Dataset which loads the CAMELYON16 embeddings from a .hdf5 file:
    """

    def __init__(
            self,
            path: str,
            transforms: Optional[BasicTransform],
            train: str,  # TODO: replace with enum
            train_frac: float
    ):
        hdf5_database = h5py.File(path, mode='r')
        assert 'images' in hdf5_database and 'masks' in hdf5_database
        img_db, mask_db = hdf5_database['images'], hdf5_database['masks']

        def get_scans_from_db(pattern: str):
            return tuple(zip(*(
                tuple(map(np.asarray, (img_db[key], mask_db[key + '_mask'])))
                for key in np.sort(list(img_db.keys()))
                if pattern in key
            )))

        # Getting all the h5 files back to numpy requires calling np.asarray on each individual value,
        # which is the reason that the iteration logic below is such a mess
        self.images, self.masks = map(
            # need casting to np.64
            lambda arrays: tuple(arr.astype(np.promote_types(np.int64, arr.dtype), casting='safe') for arr in arrays),
            (
                get_scans_from_db(pattern='test')
                if train == 'test'
                else _train_val_split_paths([
                    get_scans_from_db(modality)
                    for modality in ('normal', 'tumor')
                ], split_frac=train_frac, mode=train)
            )
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]

        return (
            (image, mask) if self.transforms is None
            else (
                (transformed := self.transforms(image=image, mask=mask))['image'],
                transformed['mask']
            )
        )


def _train_val_split_paths(
        modality_arrays: Sequence[Tuple[np.ndarray, np.ndarray]],  # Sequence[Tuple[images, masks]]
        split_frac: float,
        mode: str,  # train or validation
) -> Tuple[Tuple[np.ndarray, ...], ...]:  # Tuple[images, masks]
    """
    practically: merge n lists of (possibly unequal) size, also based on `split_frac`:
    [(a, a), (b, b)], [(c,),(d,)] into
    [(a, c), (a,)],   [(b, d), (b,)]

    'Modality' refers to types of scans, e.g.: cancer/non-cancer

    This function is obviously extremely overengineered.
    """
    assert mode in ('train', 'validation')

    def slice_arrays():
        for modality_array in modality_arrays:
            length, split_index = (
                (ln := len(modality_array[0])),
                tf if 0 < (tf := round(ln * split_frac)) < ln
                else 1 if tf == 0
                else tf - 1
            )

            # don't remove the tuple() call, otherwise the zip in the return statement breaks,
            # because the generator variables reference out-of-date values
            yield tuple(
                # return the first n slices if train, else the last m-n slices
                image_or_mask[(slice(split_index) if mode == 'train' else slice(split_index, length))]
                for image_or_mask in modality_array
            )

    # Practically: slice_arrays() returns an iterable over ((images_1, masks_1), (images_2, masks_2), ...)
    # First, create iterator over (((images_1_1,...), (images_2_1,...), ...), ((masks_1_1,...), (masks_2_1,...), ...)
    # Then, interleave these, by first zip_longest, and then chain:
    # ((images_1_1, images_2_1, ..., images_1_2, images_2_2, ...), (masks_1_1, masks_2_1, ..., etc))
    return tuple(
        map(
            lambda x: tuple(filter(lambda y: y is not None, chain.from_iterable(zip_longest(*x)))),
            zip(*slice_arrays())
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


def collate_unequal_sized_slides(batch: Sequence[Tuple[np.ndarray, ...]], mode: str = 'random_crop'):
    """batch: Sequence[Tuple[Image, Mask1, Mask2, ...]]"""
    assert mode in ('random_crop',)

    def crop_to_smallest_size(arrays_to_merge):
        if all(isinstance(arr, np.ndarray) for arr in arrays_to_merge):
            stack = lambda arrays: torch.as_tensor(np.stack(arrays))  # noqa[E731]
        elif all(isinstance(arr, torch.Tensor) for arr in arrays_to_merge):
            stack = torch.stack
        else:
            raise ValueError("All input arrays must either be numpy arrays or torch Tensors")

        def get_slices(arr: Sequence[np.ndarray]):
            residuals = (shape := np.asarray(tuple(map(attrgetter('shape'), arr)))) - shape.min(axis=0)
            crop_start_stop = np.moveaxis(np.asarray((
                # residuals + 1 to make `high` inclusive in np.random.randint
                (start_idx := np.random.randint(residuals + 1)),
                # replace 0's in crop_stop with None, because otherwise you get array[0:0]
                np.where((stop_idx := start_idx - residuals) == 0, None, stop_idx)
            )), source=0, destination=-1)  # move first axis to last

            return (
                tuple(slice(start_, stop_) for start_, stop_ in idx)
                for idx in crop_start_stop
            )

        return stack(tuple(
            array[array_slice]
            for array, array_slice
            in zip(arrays_to_merge, get_slices(arrays_to_merge))
        ))

    return tuple(crop_to_smallest_size(arrays) for arrays in zip(*batch))


if __name__ == '__main__':
    print("starting")
    dataset = CAMELYON16EmbeddingsDataset(
        path='/home/robertsc/2D-VQ-AE-2/vq_ae/multirun/2021-11-01/13-43-06/0/encodings.hdf5',
        transforms=albumentations.pytorch.ToTensorV2(),
        train='train',
        train_frac=0.95,
    )
    print(dataset[0])

    dl = DataLoader(dataset, batch_size=4, collate_fn=collate_unequal_sized_slides)

    dli = iter(dl)
    out = next(dli)
    breakpoint()
    out = next(dli)
    breakpoint()
