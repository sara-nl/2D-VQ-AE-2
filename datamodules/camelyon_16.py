from typing import Callable, Sequence
from itertools import chain


import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset
from pathlib import Path

from wsi_io.imagereader import ImageReader


class CAMELYON16RandomPatchDataSet(Dataset):
    def __init__(
        self,
        data_dir: Path,
        spacing: float,
        spacing_tolerance: float,
        patch_size, # TODO: fix signature
        train: bool = True,
        transforms: Callable = None,
    ):

        self.spacing = spacing
        self.spacing_tolerance = spacing_tolerance

        is_training_pattern = '[normal|tumor]' if train else '[test]'

        image_paths, mask_paths = (
            list(data_dir.glob(f'{modality}/?{is_training_pattern}*.tif'))
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

        self._length = len(self.image_paths)

    def __cascade_sampler(self, image: ImageReader, mask: ImageReader):
        pass

    def __getitem__(self, index):
        image = ImageReader(self.image_paths[index], self.spacing_tolerance)
        mask = ImageReader(self.mask_paths[index], self.spacing_tolerance)

        patch = self.__cascade_sampler(image, mask)

        return patch if self.transforms is None else self.transforms(patch)

    def __len__(self):
        return self._length


if __name__ == '__main__':
    data_dir = Path('/project/robertsc/examode/CAMELYON16/')
    dataset = CAMELYON16RandomPatchDataSet(data_dir, 0.5, 0.02, (128,128))
    dataset[1]