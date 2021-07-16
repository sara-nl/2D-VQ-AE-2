from typing import Callable
from argparse import Namespace, ArgumentParser
from functools import partial

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from wsi_io.imagereader import ImageReader
from utils.argparse_helpers import attr_setter_, booltype


class CAMELYON16DataModule(pl.LightningDataModule):
    def __init__(self, args: Namespace):
        super().__init__()

        self._parse_input_args_(args)
        self.init_train_dataset, self.init_test_dataset = (
            partial(CAMELYON16RandomPatchDataSet, args=args, train=True),
            partial(CAMELYON16RandomPatchDataSet, args=args, train=False)
        )


    def _parse_input_args_(self, args: Namespace) -> None:
        attr_setter_(
            self,
            args,
            ('batch_size', lambda x: x > 0),
            ('train_frac', lambda x: 0 <= x <= 1),
            ('batch_size', lambda x: x > 0),
            ('num_workers', lambda x: x >= 0),
            'shuffle_dataset',
            'drop_last',
            ('prefetch_factor', lambda x: x > 0)
        )

    @classmethod
    def add_datamodule_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # DataModule specific arguments
        parser = parent_parser.add_argument_group('Camelyon16 DataModule')
        parser.add_argument('--train-frac', default=0.95, type=int)

        parser = parent_parser.add_argument_group('Camelyon16 DataLoader')
        # Dataloader specific arguments
        parser.add_argument('--batch-size', default=16, type=int)
        parser.add_argument('--num-workers', default=6, type=int)
        parser.add_argument('--shuffle-dataset', default=True, type=booltype)
        parser.add_argument('--drop-last', default=True, type=booltype)
        parser.add_argument('--prefetch-factor', default=2, type=int)

        # Dataset specific arguments
        parser = CAMELYON16RandomPatchDataSet.add_dataset_specific_args(parser)

        return parent_parser

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            dataset = self.init_train_dataset()
            train_len, val_len = (
                (tl := round(len(dataset) * self.train_frac)),
                len(dataset) - tl
            )

            assert val_len > 0, "Validation dataset split length is 0!"

            # train/val split
            self.train_dataset, self.val_dataset = random_split(dataset, [train_len, val_len])

        if stage == 'test' or stage is None:
            self.test_dataset = self.init_test_dataset()


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True if self.num_workers > 0 else False
        )


class CAMELYON16RandomPatchDataSet(Dataset):
    def __init__(
        self,
        args: Namespace,
        train: bool = True,
        transforms: Callable = None,
    ):
        super().__init__()

        self._parse_input_args_(args)

        # TODO: find a way to put these into args
        self.train = train
        self.transforms = transforms

        is_training_pattern = '[normal|tumor]' if self.train else '[test]'

        image_paths, mask_paths = (
            list(self.data_dir.glob(f'{modality}/?{is_training_pattern}*.tif'))
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

        self.image_paths, self.mask_paths = (
            np.sort(list(map(str, modality_paths)))
            for modality_paths in (image_paths, mask_paths)
        )

        self.n_wsi = len(self.image_paths)
        self._length = self.n_wsi * self.n_patches_per_wsi

    def _parse_input_args_(self, args: Namespace):
        attr_setter_(
            self,
            args,
            'data_dir',
            ('spacing', lambda x: x > 0),
            ('spacing_tolerance', lambda x: x >= 0),
            ('patch_size', lambda x: all(elem > 0 for elem in x)),
            ('n_patches_per_wsi', lambda x: x > 0),
        )

    @classmethod
    def add_dataset_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group('Camelyon16 Dataset')

        # Dataset specific arguments
        parser.add_argument('--data-dir', type=Path, required=True)
        parser.add_argument('--spacing', default=0.5, type=float)
        parser.add_argument('--spacing-tolerance', default=0.15, type=float)
        parser.add_argument('--patch-size', default=[128, 128], nargs=2, type=int)
        parser.add_argument('--n-patches-per-wsi', default=1000, type=int)

        return parent_parser

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

        image = ImageReader(self.image_paths[wsi_index], self.spacing_tolerance)
        mask = ImageReader(self.mask_paths[wsi_index], self.spacing_tolerance)

        patch = self.__cascade_sampler(image, mask)
        return patch if self.transforms is None else self.transforms(patch)

    def __len__(self) -> int:
        return self._length
